import copy

import numpy as np
from einops import rearrange
from functools import partial

import torch
import torch.nn as nn

from model.sub_module.swin_block import PatchMerging, PatchEmbed, BasicBlock


class SwinTransformer(nn.Module):
    """ Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, args, img_size=224, patch_size=4, decoder_num_patches=49, num_bins=3, mask_ratio=0.50,
                 embed_dim=[96, 192, 384, 768], depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, out_indices=(0, 1, 2, 3),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs):
        super().__init__()

        self.args = args
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = decoder_num_patches
        self.num_layers = len(depths)
        self.embed_dim = embed_dim[0]
        self.out_indices = out_indices

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=num_bins, embed_dim=embed_dim[0],
            norm_layer=norm_layer)

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.swin_block = nn.ModuleList()
        for i in range(self.num_layers):
            blk = BasicBlock(
                dim=int(embed_dim[0] * 2 ** i),
                input_resolution=(patches_resolution[0] // (2 ** i), patches_resolution[1] // (2 ** i)),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i < self.num_layers - 1) else None)
            self.swin_block.append(blk)

        self.norm_layer = norm_layer(embed_dim[-1])

        if args.phase == "pretrain":
            if args.pr_phase == "rec" or args.pr_phase == "rec+con" or args.pr_phase == "rec-n":
                self.mask_ratio = mask_ratio
                self.stage1_output_decode = nn.Conv2d(embed_dim[0], embed_dim[-1], kernel_size=8, stride=8)
                self.stage2_output_decode = nn.Conv2d(embed_dim[1], embed_dim[-1], kernel_size=4, stride=4)
                self.stage3_output_decode = nn.Conv2d(embed_dim[2], embed_dim[-1], kernel_size=2, stride=2)

        if args.phase == "finetune_semseg" or args.phase == "finetune_flow":
            self.fpn = nn.Sequential(
                nn.Conv2d(embed_dim[-1], embed_dim[-1], kernel_size=3, stride=2),
                nn.BatchNorm2d(embed_dim[-1]),
                nn.GELU(),
            )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, x_org, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        B = x.shape[0]  # batch_size: 64
        L = self.num_patches  # 196
        # B, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))  # 49

        if self.args.masking_strategy == "random":
            noise = torch.rand(B, L, device=x.device)  # noise: (2,196)  # noise in [0, 1]
        else:
            with torch.no_grad():
                sum_events = abs(torch.sum(x_org, dim=1))
                density = nn.AvgPool2d(32, 32)(sum_events)
            density = density.flatten(1)
            if self.args.masking_strategy == "density":
                noise = density
            elif self.args.masking_strategy == "anti-density":
                noise = -density
            else:
                raise ValueError

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # 从小到大找noise的下标  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # 从小到大找ids_shuffle的下标，相当于noise当前位置的排位

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]  # [64,49]  # 保留ids_shuffle前面的，即小的noise的下标
        # x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, L], device=x.device)  # [64,196]
        mask[:, :len_keep] = 0  # (2,196)
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep, mask, ids_restore

    def apply_mask(self, x, mask, patches_resolution):
        # mask out some patches according to the random mask
        B, N, C = x.shape
        H, W = patches_resolution
        mask = mask[:1].clone()  # we use the same mask for the whole batch (1,49)
        up_ratio = N // mask.shape[1]  # 3136/49=64  # 3136/196=16
        assert up_ratio * mask.shape[1] == N
        num_repeats = int(up_ratio ** 0.5)  # 8  # 4
        if up_ratio > 1:  # mask_size != patch_embed_size
            Mh, Mw = [sz // num_repeats for sz in patches_resolution]  # 7, 7  # 14, 14
            mask = mask.reshape(1, Mh, 1, Mw, 1)  # (1,7,1,7,1)  # (1,14,1,14,1)
            mask = mask.expand(-1, -1, num_repeats, -1, num_repeats)  # (1,7,8,7,8)  # (1,14,4,14,4)
            mask = mask.reshape(1, -1)  # (1,3136)

        # record the corresponding coordinates of visible patches
        coords_h = torch.arange(H, device=x.device)  # (56)
        coords_w = torch.arange(W, device=x.device)  # (56)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]), dim=-1)  # H W 2
        coords = coords.reshape(1, H * W, 2)

        # mask out patches
        vis_mask = ~mask  # ~mask means visible, (1, N_vis)
        x_vis = x[vis_mask.expand(B, -1)].reshape(B, -1, C)  # (2,1536,96) 768=8*8*12 (12=49//4)  # 784=4*4*49 (49=196//4)
        coords = coords[vis_mask].reshape(1, -1, 2)  # (1 N_vis 2)

        return x_vis, coords, vis_mask  # (2,1536,96) (1,1536,2) (1,3136)

    def forward(self, x, mask=False):
        if mask:
            x_org = copy.deepcopy(x)
            x = self.patch_embed(x)  # (2,5,224,224) -> (2,3136,96)
            x = self.pos_drop(x)

            ids_keep, mask, ids_restore = self.random_masking(x, x_org, self.mask_ratio)  # ids_keep: (2,24) mask:(2,49(num_patches)) ids_restore:(2,49)

            # mask out some patches according to the random mask
            x_down, coords_down, vis_mask_down = self.apply_mask(x, mask.bool(), self.patches_resolution)  # x:(2,3136,128) mask:(2,49) self.patches_resolution:(56,56)
            # (2,1536,96) (1,1536,2) (1,3136)  # (2,768,128) (1,768,2) (1,3136)

            for i, blk in enumerate(self.swin_block):
                if i < len(self.swin_block) - 1:
                    x, coords, vis_mask, x_down, coords_down, vis_mask_down, attn = blk(x_down, coords_down, vis_mask_down)  # (2,192,256) (1,192,2) (1,784) (2,48,512) (1,48,2) (1,196)
                else:
                    x, coords, vis_mask, attn = blk(x_down, coords_down, vis_mask_down)
                # (2,384,192) (1,384,2) (1,784)
                # (2,96,384) (1,96,2) (1,196)
                # (2,24,768) (1,24,2) (1,49)
                if i in self.out_indices:
                    if i == 0:
                        emb_l1 = x
                        coords_l1 = coords
                        # mask_tokens = torch.zeros((emb_l1.shape[0], 56 * 56 - emb_l1.shape[1], emb_l1.shape[2]), device=emb_l1.device)
                        # _emb_l1 = torch.cat([emb_l1, mask_tokens], dim=1)  # no cls token
                        # _emb_l1 = torch.gather(_emb_l1, dim=1,
                        #                        index=ids_restore_list[-1].unsqueeze(-1).repeat(1, 1, emb_l1.shape[2]))  # unshuffle
                        # _emb_l1 = _emb_l1.reshape(_emb_l1.shape[0], int(_emb_l1.shape[1] ** .5),
                        #                           int(_emb_l1.shape[1] ** .5), _emb_l1.shape[2]).permute(0, 3, 1, 2)  # (2,96,56,56)

                        _emb_l1 = torch.zeros((emb_l1.shape[0], 56 * 56, emb_l1.shape[-1]), device=emb_l1.device)
                        _emb_l1[:, coords_l1[0, :, 0] * 56 + coords_l1[0, :, 1], :] = x.to(torch.float32)
                        _emb_l1 = _emb_l1.reshape(_emb_l1.shape[0], int(_emb_l1.shape[1] ** .5),
                                                  int(_emb_l1.shape[1] ** .5), _emb_l1.shape[2]).permute(0, 3, 1, 2)
                        emb_stage1 = self.stage1_output_decode(_emb_l1).flatten(2).permute(0, 2, 1)  # (2,49,768)
                        emb_stage1 = torch.gather(emb_stage1, dim=1,
                                                  index=ids_keep.unsqueeze(-1).repeat(1, 1, emb_stage1.shape[-1]))  # (2,24,768)
                    elif i == 1:
                        emb_l2 = x
                        coords_l2 = coords
                        _emb_l2 = torch.zeros((emb_l2.shape[0], 28 * 28, emb_l2.shape[-1]), device=emb_l2.device)
                        _emb_l2[:, coords_l2[0, :, 0] * 28 + coords_l2[0, :, 1], :] = x.to(torch.float32)
                        _emb_l2 = _emb_l2.reshape(_emb_l2.shape[0], int(_emb_l2.shape[1] ** .5),
                                                  int(_emb_l2.shape[1] ** .5), _emb_l2.shape[2]).permute(0, 3, 1, 2)
                        emb_stage2 = self.stage2_output_decode(_emb_l2).flatten(2).permute(0, 2, 1)  # (2,49,768)
                        emb_stage2 = torch.gather(emb_stage2, dim=1,
                                                  index=ids_keep.unsqueeze(-1).repeat(1, 1, emb_stage2.shape[-1]))  # (2,24,768)
                    elif i == 2:
                        emb_l3 = x
                        coords_l3 = coords
                        _emb_l3 = torch.zeros((emb_l3.shape[0], 14 * 14, emb_l3.shape[-1]), device=emb_l3.device)
                        _emb_l3[:, coords_l3[0, :, 0] * 14 + coords_l3[0, :, 1], :] = x.to(torch.float32)
                        _emb_l3 = _emb_l3.reshape(_emb_l3.shape[0], int(_emb_l3.shape[1] ** .5),
                                                  int(_emb_l3.shape[1] ** .5), _emb_l3.shape[2]).permute(0, 3, 1, 2)
                        emb_stage3 = self.stage3_output_decode(_emb_l3).flatten(2).permute(0, 2, 1)  # (2,49,768)
                        emb_stage3 = torch.gather(emb_stage3, dim=1,
                                                  index=ids_keep.unsqueeze(-1).repeat(1, 1, emb_stage3.shape[-1]))  # (2,24,768)
                    else:
                        emb_l4 = x
                        coords_l4 = coords
                        emb_stage4 = emb_l4

            # (2,3136,96) 56 56 (2,784,192) 28 28 (128,3,49,49)
            # (2,784,192) 28 28 (2,196,384) 14 14 (32,6,49,49)
            # (2,196,384) 14 14 (2,49,768) 7 7 (8,12,49,49)
            # (2,49,768) 7 7 (2,49,768) 7 7 (2,24,49,49)

            if self.args.use_feature_fusion:
                emb_lh = self.norm_layer(emb_stage1 + emb_stage2 + emb_stage3 + emb_stage4)
            else:
                emb_lh = self.norm_layer(emb_stage4)

            return emb_l1, emb_l2, emb_l3, emb_l4, emb_lh, coords_l1, coords_l2, coords_l3, coords_l4, mask, ids_restore, attn
            # (2,1536,96) (2,384,192) (2,96,384) (2,24,768) (2,24,768) (2,49) (2,49)
        else:
            x = self.patch_embed(x)
            x = self.pos_drop(x)

            mask = torch.zeros((2, 49), device=x.device)
            # mask out some patches according to the random mask
            x_down, coords_down, vis_mask_down = self.apply_mask(x, mask.bool(), self.patches_resolution)

            out_embs = []
            for i, blk in enumerate(self.swin_block):
                if i < len(self.swin_block) - 1:
                    x, coords, vis_mask, x_down, coords_down, vis_mask_down, attn = blk(x_down, coords_down, vis_mask_down)  # (2,192,256) (1,192,2) (1,784) (2,48,512) (1,48,2) (1,196)
                else:
                    x, coords, vis_mask, attn = blk(x_down, coords_down, vis_mask_down)
                # (2,384,192) (1,384,2) (1,784)
                # (2,96,384) (1,96,2) (1,196)
                # (2,24,768) (1,24,2) (1,49)
                if i in self.out_indices:
                    if i == 0:
                        emb_l1 = x
                        out_embs.append(emb_l1.view(-1, 56, 56, emb_l1.shape[-1]).permute(0, 3, 1, 2).contiguous())
                    elif i == 1:
                        emb_l2 = x
                        out_embs.append(emb_l2.view(-1, 28, 28, emb_l2.shape[-1]).permute(0, 3, 1, 2).contiguous())
                    elif i == 2:
                        emb_l3 = x
                        out_embs.append(emb_l3.view(-1, 14, 14, emb_l3.shape[-1]).permute(0, 3, 1, 2).contiguous())
                    else:
                        emb_l4 = x
                        out_embs.append(emb_l4.view(-1, 7, 7, emb_l4.shape[-1]).permute(0, 3, 1, 2).contiguous())

            emb_h = self.norm_layer(x)

            if self.args.phase == "finetune_semseg" or self.args.phase == "finetune_flow":
                return emb_l1, emb_l2, emb_l3, emb_l4, emb_h, out_embs, attn
            else:
                return emb_l1, emb_l2, emb_l3, emb_l4, emb_h, attn


def swin_tiny_window7(args, **kwargs):
    model = SwinTransformer(args=args,
        pretrain_img_size=224, patch_size=4, decoder_num_patches=49,
        embed_dim=[96, 192, 384, 768], depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7,
        mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )

    return model

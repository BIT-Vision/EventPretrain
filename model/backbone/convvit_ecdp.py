from functools import partial

import torch
import torch.nn as nn

from utils.reshape import emb2patch_frame
from utils.pos_embed import get_2d_sincos_pos_embed
from model.sub_module.vit_block import PatchEmbed, ViTBlock
from model.sub_module.conv_block import ConvBlock


class ConvViT_ECDP(nn.Module):
    def __init__(self, args, input_size=224, patch_size=16,
                 embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm,
                 num_bins=5, mask_ratio=0., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.args = args
        self.patch_size = patch_size

        self.patch_embed1 = PatchEmbed(
            img_size=input_size[0], patch_size=patch_size[0], in_chans=num_bins, embed_dim=embed_dim[0])
        self.patch_embed2 = PatchEmbed(
            img_size=input_size[1], patch_size=patch_size[1], in_chans=embed_dim[0], embed_dim=embed_dim[1])
        self.patch_embed3 = PatchEmbed(
            img_size=input_size[2], patch_size=patch_size[2], in_chans=embed_dim[1], embed_dim=embed_dim[2])

        self.patch_embed4 = nn.Linear(embed_dim[2], embed_dim[2])
        self.num_patches = self.patch_embed3.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim[2]), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        self.conv_block1 = nn.ModuleList([
            ConvBlock(input_size=embed_dim[0], kernel_size=5, mlp_ratio=4., drop=drop_rate, drop_path=dpr[i])
            for i in range(depth[0])])
        self.conv_block2 = nn.ModuleList([
            ConvBlock(input_size=embed_dim[1], kernel_size=5, mlp_ratio=4., drop=drop_rate, drop_path=dpr[depth[0] + i])
            for i in range(depth[0])])
        self.vit_block = nn.ModuleList([
            ViTBlock(dim=embed_dim[2], num_heads=num_heads, mlp_ratio=mlp_ratio[2],
                     qkv_bias=True, qk_scale=None, drop=drop_rate, attn_drop=attn_drop_rate,
                     drop_path=dpr[depth[0] + depth[1] + i], norm_layer=norm_layer)
            for i in range(depth[2])])
        self.tokens = nn.Parameter(torch.zeros(1, 2, embed_dim[2]))

        if args.phase == "pretrain":
            self.mask_ratio = mask_ratio

        self.norm_layer = norm_layer(embed_dim[-1])

        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** .5),
                                            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed3.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        B = x.shape[0]  # batch_size: 64
        L = self.num_patches  # 196
        # B, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self.mask_ratio))  # 49

        if self.args.masking_strategy == "random":
            noise = torch.rand(B, L, device=x.device)  # noise: (2,196)  # noise in [0, 1]
        else:
            with torch.no_grad():
                sum_events = abs(torch.sum(x, dim=1))
                density = nn.AvgPool2d(self.patch_size, self.patch_size)(sum_events)
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

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, L], device=x.device)  # [64,196]
        mask[:, :len_keep] = 0  # (2,196)
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep, mask, ids_restore

    def forward(self, x, mask=False):
        if mask:
            # stage1
            ids_keep, mask, ids_restore = self.random_masking(x)
            mask_for_patch1 = mask.reshape(-1, 14, 14).unsqueeze(-1).repeat(1, 1, 1, 16).reshape(-1, 14, 14, 4, 4). \
                permute(0, 1, 3, 2, 4).reshape(x.shape[0], 56, 56).unsqueeze(1)
            x = self.patch_embed1(x)  # (2,128,56,56)
            x = self.pos_drop(x)
            for blk in self.conv_block1:
                x = blk(x, 1 - mask_for_patch1)

            # stage2
            mask_for_patch2 = mask.reshape(-1, 14, 14).unsqueeze(-1).repeat(1, 1, 1, 4).reshape(-1, 14, 14, 2, 2). \
                permute(0, 1, 3, 2, 4).reshape(x.shape[0], 28, 28).unsqueeze(1)
            x = self.patch_embed2(x)  # (2,256,28,28)
            for blk in self.conv_block2:
                x = blk(x, 1 - mask_for_patch2)

            # stage3
            x = self.patch_embed3(x)  # (2,384,14,14)
            x = x.flatten(2).permute(0, 2, 1)
            x = self.patch_embed4(x)  # (2,196,384))
            # add pos embed w/o cls token
            x = x + self.pos_embed
            x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))
            x = torch.cat((self.tokens.expand(x.shape[0], -1, -1), x), dim=1)  # (2,100,384)  #

            for i, blk in enumerate(self.vit_block):
                if i < len(self.vit_block) - 1:
                    x = blk(x)
                else:
                    x, attn = blk(x, return_attn=True)

            emb_h = self.norm_layer(x)
            emb_event, emb_image = emb_h[:, 0], emb_h[:, 1]

            return emb_event, emb_image, mask, ids_restore, attn

        else:
            # stage1
            x = self.patch_embed1(x)  # (2,128,56,56)
            x = self.pos_drop(x)
            for blk in self.conv_block1:
                x = blk(x)

            # stage2
            x = self.patch_embed2(x)  # (2,256,28,28)
            for blk in self.conv_block2:
                x = blk(x)

            # stage3
            x = self.patch_embed3(x)  # (2,384,14,14)
            x = x.flatten(2).permute(0, 2, 1)
            x = self.patch_embed4(x)  # (2,196,384))
            # add pos embed w/o cls token
            x = x + self.pos_embed
            x = torch.cat((self.tokens.expand(x.shape[0], -1, -1), x), dim=1)  # (2,100,384)  #

            for i, blk in enumerate(self.vit_block):
                if i < len(self.vit_block) - 1:
                    x = blk(x)
                else:
                    x, attn = blk(x, return_attn=True)

            x = self.norm_layer(x)
            emb = torch.cat((x[:, 0], x[:, 1]), 1)

            return emb, attn


def convvit_ecdp_small_patch16(args, **kwargs):
    model = ConvViT_ECDP(args=args,
        input_size=[224, 56, 28], patch_size=[4, 2, 2],
        embed_dim=[128, 256, 384], depth=[2, 2, 11], num_heads=12,
        mlp_ratio=[4, 4, 4], norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )

    return model


def convvit_ecdp_base_patch16(args, **kwargs):
    model = ConvViT_ECDP(args=args,
        input_size=[224, 56, 28], patch_size=[4, 2, 2],
        embed_dim=[256, 384, 768], depth=[2, 2, 11], num_heads=12,
        mlp_ratio=[4, 4, 4], norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )

    return model

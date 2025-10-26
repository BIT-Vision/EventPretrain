from functools import partial

import torch
import torch.nn as nn

from utils.reshape import emb2patch_frame
from utils.pos_embed import get_2d_sincos_pos_embed
from model.sub_module.vit_block import PatchEmbed, ViTBlock


class ViT(nn.Module):
    def __init__(self, args, input_size=224, patch_size=16, embed_dim=1024,
                 depth=24, num_heads=16, mlp_ratio=4., out_indices=[3, 5, 7, 11], norm_layer=nn.LayerNorm,
                 num_bins=5, mask_ratio=0., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.args = args
        self.patch_size = patch_size
        self.out_indices = out_indices

        self.patch_embed = PatchEmbed(
            img_size=input_size, patch_size=patch_size, in_chans=num_bins, embed_dim=embed_dim)

        self.num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.vit_block = nn.ModuleList([
            ViTBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                     qkv_bias=True, qk_scale=None, drop=drop_rate, attn_drop=attn_drop_rate,
                     drop_path=dpr[i], norm_layer=norm_layer)
                for i in range(depth)])

        if args.phase == "pretrain":
            if args.pr_phase == "rec" or args.pr_phase == "rec+con" or args.pr_phase == "rec-n":
                self.mask_ratio = mask_ratio

        self.norm_layer = norm_layer(embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** .5),
                                            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
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
                sum_events = abs(torch.sum(x, dim=1))  # (2,224,224)
                density = nn.AvgPool2d(self.patch_size, self.patch_size)(sum_events)  # (2,14,14)
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

    def forward(self, x, mask=False):
        if mask:
            ids_keep, mask, ids_restore = self.random_masking(x)
            x = self.patch_embed(x)  # (2,5,224,224)->(2,384,14,14)
            x = x.flatten(2).permute(0, 2, 1)  # (2,196,384)
            # add pos embed w/o cls token
            x = x + self.pos_embed
            x = self.pos_drop(x)
            x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))

            for i, blk in enumerate(self.vit_block):
                x = blk(x)
                if i == 1:
                    emb_l1 = x
                elif i == 3:
                    emb_l2 = x
            emb_h = x

            if self.args.use_feature_fusion:
                emb_lh = self.norm_layer(emb_l1 + emb_l2 + emb_h)
            else:
                emb_lh = self.norm_layer(emb_h)

            return emb_l1, emb_l2, emb_lh, mask, ids_restore  # (1,98,384) (1,98,384) (1,98,384)

        else:
            x = self.patch_embed(x)  # (2,5,224,224)->(2,384,14,14)
            x = x.flatten(2).permute(0, 2, 1)  # (2,196,384)
            # add pos embed w/o cls token
            x = x + self.pos_embed
            x = self.pos_drop(x)

            out_embs = []
            for i, blk in enumerate(self.vit_block):
                if i < len(self.vit_block) - 1:
                    x = blk(x)
                else:
                    x, attn = blk(x, return_attn=True)
                if i == 0:
                    emb_l1 = x
                elif i == 1:
                    emb_l2 = x
                if i in self.out_indices:
                    out_embs.append(emb2patch_frame(x))

            emb_h = self.norm_layer(x)
            if self.args.phase == "finetune_semseg" or self.args.phase == "finetune_flow":
                return emb_l1, emb_l2, emb_h, out_embs, attn
            else:
                return emb_l1, emb_l2, emb_h, attn  # (1,196,384) (1,196,384) (1,196,384) (2,12,198,198)


def vit_small_patch16(args, **kwargs):
    model = ViT(args=args,
        input_size=224, patch_size=16, embed_dim=384,
        depth=12, out_indices=[3, 5, 7, 11], num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch16(args, **kwargs):
    model = ViT(args=args,
        input_size=224, patch_size=16, embed_dim=768,
        depth=12, out_indices=[3, 5, 7, 11], num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

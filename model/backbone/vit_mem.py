from functools import partial

import torch
import torch.nn as nn

from utils.reshape import emb2patch_frame
from utils.pos_embed import get_2d_sincos_pos_embed
from model.sub_module.vit_block import RelativePositionBias, PatchEmbed_MEM, ViTBlock_MEM


class ViT_MEM(nn.Module):
    def __init__(self, args, input_size=224, patch_size=16, embed_dim=1024,
                 depth=24, num_heads=16, mlp_ratio=4., out_indices=[3, 5, 7, 11], norm_layer=nn.LayerNorm,
                 num_bins=5, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., init_values=None):
        super().__init__()

        self.args = args
        self.out_indices = out_indices

        self.patch_embed = PatchEmbed_MEM(
            img_size=input_size, patch_size=patch_size, in_chans=num_bins, embed_dim=embed_dim)

        self.num_patches = self.patch_embed.num_patches
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)
        self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.vit_block = nn.ModuleList([
            ViTBlock_MEM(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                         qkv_bias=True, qk_scale=None, drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[i], norm_layer=norm_layer,
                         init_values=init_values, window_size=self.patch_embed.patch_shape)
                for i in range(depth)])

        self.norm_layer = norm_layer(embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** .5),
        #                                     cls_token=False)
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

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

    def forward(self, x):
        x = self.patch_embed(x)  # (2,5,224,224)->(2,384,14,14)
        x = x.flatten(2).permute(0, 2, 1)  # (2,196,384)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)  # (2,198,384)
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias()
        out_embs = []
        for i, blk in enumerate(self.vit_block):
            if i < len(self.vit_block) - 1:
                x = blk(x, rel_pos_bias=rel_pos_bias)
                # x = blk(x)
            else:
                x, attn = blk(x, rel_pos_bias=rel_pos_bias, return_attn=True)
                # x, attn = blk(x, return_attn=True)
            if i in self.out_indices:
                out_embs.append(emb2patch_frame(x[:, 1:, :]))

        x = x[:, 1:, :].mean(1)
        emb = self.norm_layer(x)

        if self.args.phase == "finetune_semseg" or self.args.phase == "finetune_flow":
            return emb, out_embs, attn
        else:
            return emb, attn  # (2,768) (2,12,198,198)


def vit_mem_small_patch16(args, **kwargs):
    model = ViT_MEM(args=args,
        input_size=224, patch_size=16, embed_dim=384,
        depth=12, out_indices=[3, 5, 7, 11], num_heads=12,
        mlp_ratio=4, init_values=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_mem_base_patch16(args, **kwargs):
    model = ViT_MEM(args=args,
        input_size=224, patch_size=16, embed_dim=768,
        depth=12, out_indices=[3, 5, 7, 11], num_heads=12,
        mlp_ratio=4, init_values=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

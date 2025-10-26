from functools import partial

import torch
import torch.nn as nn

from utils.pos_embed import get_2d_sincos_pos_embed
from model.sub_module.vit_block import ViTBlock


class PrRecDecoder(nn.Module):
    def __init__(self, patch_size=16, num_patches=196,
                 encoder_embed_dim=768, embed_dim=512, depth=8, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 frame_chans=1):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = num_patches

        self.patch_embed = nn.Linear(encoder_embed_dim[-1], embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.vit_block = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, mlp_ratio[0], qkv_bias=True, qk_scale=None,
                  norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        self.pred = nn.Linear(embed_dim, self.patch_size ** 2 * frame_chans, bias=True)  # decoder to patch

        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1],
                                                    int(self.num_patches ** .5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

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

    def forward(self, x, ids_restore=None):
        x = self.patch_embed(x)

        if ids_restore is not None:
            # append mask tokens to sequence
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
            x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
            x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        x = x + self.pos_embed

        for blk in self.vit_block:
            x = blk(x)
        x = self.norm(x)

        x = self.pred(x)

        return x


def pretrain_rec_decoder_small_patch16(**kwargs):
    model = PrRecDecoder(
        patch_size=16, num_patches=196,
        encoder_embed_dim=[128, 256, 384], embed_dim=256, depth=8, num_heads=8,  # 维度(256)需要能整除num_heads(8)
        mlp_ratio=[4, 4, 4], norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model

def pretrain_rec_decoder_swin_tiny_patch32(**kwargs):
    model = PrRecDecoder(
        patch_size=32, num_patches=49,
        encoder_embed_dim=[96, 192, 384, 768], embed_dim=256, depth=8, num_heads=8,  # 维度(256)需要能整除num_heads(8)
        mlp_ratio=[4, 4, 4], norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model

def pretrain_rec_decoder_base_patch16(**kwargs):
    model = PrRecDecoder(
        patch_size=16, num_patches=196,
        encoder_embed_dim=[256, 384, 768], embed_dim=512, depth=8, num_heads=16,
        mlp_ratio=[4, 4, 4], norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model

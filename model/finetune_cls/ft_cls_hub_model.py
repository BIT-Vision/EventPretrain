import torch
import torch.nn as nn

from model.backbone import vit, convvit, swin, vit_ecdp, convvit_ecdp, vit_mem, swin_ecddp


class FtClsHubModel(nn.Module):
    def __init__(self, args, embed_dim=1024):
        super().__init__()
        self.backbone_type = args.backbone_type

        if args.backbone_type == "vit":
            if args.model_size == "small":
                self.backbone = vit.__dict__["vit_small_patch16"](args=args,
                                                                  num_bins=args.num_bins,
                                                                  drop_rate=args.drop_rate,
                                                                  attn_drop_rate=args.attn_drop_rate,
                                                                  drop_path_rate=args.drop_path_rate)
            elif args.model_size == "base":
                self.backbone = vit.__dict__["vit_base_patch16"](args=args,
                                                                 num_bins=args.num_bins,
                                                                 drop_rate=args.drop_rate,
                                                                 attn_drop_rate=args.attn_drop_rate,
                                                                 drop_path_rate=args.drop_path_rate)
            else:
                raise ValueError

        elif args.backbone_type == "convvit":
            if args.model_size == "small":
                self.backbone = convvit.__dict__["convvit_small_patch16"](args,
                                                                          num_bins=args.num_bins,
                                                                          drop_rate=args.drop_rate,
                                                                          attn_drop_rate=args.attn_drop_rate,
                                                                          drop_path_rate=args.drop_path_rate)
            elif args.model_size == "base":
                self.backbone = convvit.__dict__["convvit_base_patch16"](args,
                                                                         num_bins=args.num_bins,
                                                                         drop_rate=args.drop_rate,
                                                                         attn_drop_rate=args.attn_drop_rate,
                                                                         drop_path_rate=args.drop_path_rate)
            else:
                raise ValueError
        elif args.backbone_type == "swin":
            self.backbone = swin.__dict__["swin_tiny_window7"](args,
                                                               num_bins=args.num_bins,
                                                               drop_rate=args.drop_rate,
                                                               attn_drop_rate=args.attn_drop_rate,
                                                               drop_path_rate=args.drop_path_rate)
        elif args.backbone_type == "vit_ecdp":
            if args.model_size == "small":
                self.backbone = vit_ecdp.__dict__["vit_ecdp_small_patch16"](args,
                                                                            num_bins=args.num_bins,
                                                                            drop_rate=args.drop_rate,
                                                                            attn_drop_rate=args.attn_drop_rate,
                                                                            drop_path_rate=args.drop_path_rate)
            elif args.model_size == "base":
                self.backbone = vit_ecdp.__dict__["vit_ecdp_base_patch16"](args,
                                                                            num_bins=args.num_bins,
                                                                            drop_rate=args.drop_rate,
                                                                            attn_drop_rate=args.attn_drop_rate,
                                                                            drop_path_rate=args.drop_path_rate)
        elif args.backbone_type == "convvit_ecdp":
            if args.model_size == "small":
                self.backbone = convvit_ecdp.__dict__["convvit_ecdp_small_patch16"](args,
                                                                            num_bins=args.num_bins,
                                                                            drop_rate=args.drop_rate,
                                                                            attn_drop_rate=args.attn_drop_rate,
                                                                            drop_path_rate=args.drop_path_rate)
            elif args.model_size == "base":
                self.backbone = convvit_ecdp.__dict__["convvit_ecdp_base_patch16"](args,
                                                                            num_bins=args.num_bins,
                                                                            drop_rate=args.drop_rate,
                                                                            attn_drop_rate=args.attn_drop_rate,
                                                                            drop_path_rate=args.drop_path_rate)
        elif args.backbone_type == "vit_mem":
            if args.model_size == "small":
                self.backbone = vit_mem.__dict__["vit_mem_small_patch16"](args,
                                                                           num_bins=args.num_bins,
                                                                           drop_rate=args.drop_rate,
                                                                           attn_drop_rate=args.attn_drop_rate,
                                                                           drop_path_rate=args.drop_path_rate)
            elif args.model_size == "base":
                self.backbone = vit_mem.__dict__["vit_mem_base_patch16"](args,
                                                                          num_bins=args.num_bins,
                                                                          drop_rate=args.drop_rate,
                                                                          attn_drop_rate=args.attn_drop_rate,
                                                                          drop_path_rate=args.drop_path_rate)
            else:
                raise ValueError

        elif args.backbone_type == "swin_ecddp":
            self.backbone = swin_ecddp.__dict__["swin_ecddp_tiny_window7"](args,
                                                                          num_bins=args.num_bins,
                                                                          drop_rate=args.drop_rate,
                                                                          attn_drop_rate=args.attn_drop_rate,
                                                                          drop_path_rate=args.drop_path_rate)
        else:
            raise ValueError

        # classify head
        if args.backbone_type == "vit_ecdp" or args.backbone_type == "convvit_ecdp":
            self.classify_head = nn.Linear(embed_dim[-1] * 2, args.num_classes)
        else:
            self.classify_head = nn.Linear(embed_dim[-1], args.num_classes)

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
        if self.backbone_type == "vit_ecdp" or self.backbone_type == "convvit_ecdp" or self.backbone_type == "vit_mem":
            emb, attn = self.backbone(x)

            # classify head
            pred = self.classify_head(emb)

            return emb, pred, attn
        else:
            if self.backbone_type == "swin" or self.backbone_type == "swin_ecddp":
                emb_l1, emb_l2, emb_l3, emb_l4, emb_h, attn = self.backbone(x)
            else:
                emb_l1, emb_l2, emb_h, attn = self.backbone(x)

            # classify head
            emb_h_pool = emb_h.mean(dim=1)  # global pool without cls token (2,196,384) -> (2,384)
            pred = self.classify_head(emb_h_pool)

            if self.backbone_type == "swin" or self.backbone_type == "swin_ecddp":
                return emb_l1, emb_l2, emb_l3, emb_l4, emb_h, pred, attn
            else:
                return emb_l1, emb_l2, emb_h, pred, attn


def finetune_cls_hub_model_small_patch16(args):
    model = FtClsHubModel(args=args, embed_dim=[128, 256, 384])
    return model

def finetune_cls_hub_model_swin_tiny_window7(args):
    model = FtClsHubModel(args=args, embed_dim=[96, 192, 384, 768])
    return model

def finetune_cls_hub_model_base_patch16(args):
    model = FtClsHubModel(args=args, embed_dim=[256, 384, 768])
    return model

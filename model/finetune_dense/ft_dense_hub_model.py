import torch
import torch.nn as nn

from model.finetune_dense import ft_dense_decoder
from model.backbone import vit, convvit, swin, vit_ecdp, convvit_ecdp, vit_mem, swin_ecddp


class FtDenseHubModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.backbone_type = args.backbone_type
        if args.backbone_type == "vit":
            if args.model_size == "small":
                self.backbone = vit.__dict__["vit_small_patch16"](args=args,
                                                                  num_bins=args.num_bins,
                                                                  drop_rate=args.drop_rate,
                                                                  attn_drop_rate=args.attn_drop_rate,
                                                                  drop_path_rate=args.drop_path_rate)
                if args.phase == "finetune_semseg":
                    self.decode_head = ft_dense_decoder.__dict__["finetune_decode_head_small"](args,
                                                                                               out_channels=args.num_classes)
                    self.auxiliary_head = ft_dense_decoder.__dict__["finetune_auxiliary_head_small"](args,
                                                                                                     out_channels=args.num_classes)
                elif args.phase == "finetune_flow":
                    self.decode_head = ft_dense_decoder.__dict__["finetune_decode_head_small"](args, out_channels=2)
                    self.auxiliary_head = ft_dense_decoder.__dict__["finetune_auxiliary_head_small"](args,
                                                                                                     out_channels=2)
                else:
                    raise ValueError

            elif args.model_size == "base":
                self.backbone = vit.__dict__["vit_base_patch16"](args=args,
                                                                 num_bins=args.num_bins,
                                                                 drop_rate=args.drop_rate,
                                                                 attn_drop_rate=args.attn_drop_rate,
                                                                 drop_path_rate=args.drop_path_rate)
                if args.phase == "finetune_semseg":
                    self.decode_head = ft_dense_decoder.__dict__["finetune_decode_head_base"](args,
                                                                                              out_channels=args.num_classes)
                    self.auxiliary_head = ft_dense_decoder.__dict__["finetune_auxiliary_head_base"](args,
                                                                                                    out_channels=args.num_classes)
                elif args.phase == "finetune_flow":
                    self.decode_head = ft_dense_decoder.__dict__["finetune_flow_decode_head_base"](args, out_channels=2)
                    self.auxiliary_head = ft_dense_decoder.__dict__["finetune_flow_auxiliary_head_base"](args,
                                                                                                         out_channels=2)
                else:
                    raise ValueError
            else:
                raise ValueError

        elif args.backbone_type == "convvit":
            if args.model_size == "small":
                self.backbone = convvit.__dict__["convvit_small_patch16"](args=args,
                                                                          num_bins=args.num_bins,
                                                                          drop_rate=args.drop_rate,
                                                                          attn_drop_rate=args.attn_drop_rate,
                                                                          drop_path_rate=args.drop_path_rate)
                if args.phase == "finetune_semseg":
                    self.decode_head = ft_dense_decoder.__dict__["finetune_decode_head_extend_small"](args,
                                                                                               out_channels=args.num_classes)
                    self.auxiliary_head = ft_dense_decoder.__dict__["finetune_auxiliary_head_small"](args,
                                                                                                     out_channels=args.num_classes)
                elif args.phase == "finetune_flow":
                    self.decode_head = ft_dense_decoder.__dict__["finetune_decode_head_extend_small"](args, out_channels=2)
                    self.auxiliary_head = ft_dense_decoder.__dict__["finetune_auxiliary_head_small"](args,
                                                                                                     out_channels=2)
                else:
                    raise ValueError

            elif args.model_size == "base":
                self.backbone = convvit.__dict__["convvit_base_patch16"](args,
                                                                         num_bins=args.num_bins,
                                                                         drop_rate=args.drop_rate,
                                                                         attn_drop_rate=args.attn_drop_rate,
                                                                         drop_path_rate=args.drop_path_rate)
                if args.phase == "finetune_semseg":
                    self.decode_head = ft_dense_decoder.__dict__["finetune_decode_head_extend_base"](args,
                                                                                              out_channels=args.num_classes)
                    self.auxiliary_head = ft_dense_decoder.__dict__["finetune_auxiliary_head_base"](args,
                                                                                                    out_channels=args.num_classes)
                elif args.phase == "finetune_flow":
                    self.decode_head = ft_dense_decoder.__dict__["finetune_decode_head_extend_base"](args, out_channels=2)
                    self.auxiliary_head = ft_dense_decoder.__dict__["finetune_auxiliary_head_base"](args,
                                                                                                         out_channels=2)
                else:
                    raise ValueError
            else:
                raise ValueError
        elif args.backbone_type == "swin":
            self.backbone = swin.__dict__["swin_tiny_window7"](args=args,
                                                               num_bins=args.num_bins,
                                                               drop_rate=args.drop_rate,
                                                               attn_drop_rate=args.attn_drop_rate,
                                                               drop_path_rate=args.drop_path_rate)
            if args.phase == "finetune_semseg":
                self.decode_head = ft_dense_decoder.__dict__["finetune_decode_head_extend_small_swin"](args,
                                                                                                 out_channels=args.num_classes)
                self.auxiliary_head = ft_dense_decoder.__dict__["finetune_auxiliary_head_small_swin"](args,
                                                                                                out_channels=args.num_classes)
            elif args.phase == "finetune_flow":
                self.decode_head = ft_dense_decoder.__dict__["finetune_decode_head_extend_small_swin"](args, out_channels=2)
                self.auxiliary_head = ft_dense_decoder.__dict__["finetune_auxiliary_head_small_swin"](args,
                                                                                                out_channels=2)
            else:
                raise ValueError
        elif args.backbone_type == "vit_ecdp":
            if args.model_size == "small":
                self.backbone = vit_ecdp.__dict__["vit_ecdp_small_patch16"](args,
                                                                            num_bins=args.num_bins,
                                                                            drop_rate=args.drop_rate,
                                                                            attn_drop_rate=args.attn_drop_rate,
                                                                            drop_path_rate=args.drop_path_rate)
                if args.phase == "finetune_semseg":
                    self.decode_head = ft_dense_decoder.__dict__["finetune_decode_head_small"](args,
                                                                                               out_channels=args.num_classes)
                    self.auxiliary_head = ft_dense_decoder.__dict__["finetune_auxiliary_head_small"](args,
                                                                                                     out_channels=args.num_classes)
                elif args.phase == "finetune_flow":
                    self.decode_head = ft_dense_decoder.__dict__["finetune_decode_head_small"](args, out_channels=2)
                    self.auxiliary_head = ft_dense_decoder.__dict__["finetune_auxiliary_head_small"](args,
                                                                                                     out_channels=2)
                else:
                    raise ValueError
            elif args.model_size == "base":
                self.backbone = vit_ecdp.__dict__["vit_ecdp_base_patch16"](args,
                                                                            num_bins=args.num_bins,
                                                                            drop_rate=args.drop_rate,
                                                                            attn_drop_rate=args.attn_drop_rate,
                                                                            drop_path_rate=args.drop_path_rate)
                if args.phase == "finetune_semseg":
                    self.decode_head = ft_dense_decoder.__dict__["finetune_decode_head_base"](args,
                                                                                              out_channels=args.num_classes)
                    self.auxiliary_head = ft_dense_decoder.__dict__["finetune_auxiliary_head_base"](args,
                                                                                                    out_channels=args.num_classes)
                elif args.phase == "finetune_flow":
                    self.decode_head = ft_dense_decoder.__dict__["finetune_decode_head_base"](args, out_channels=2)
                    self.auxiliary_head = ft_dense_decoder.__dict__["finetune_auxiliary_head_base"](args,
                                                                                                    out_channels=2)
                else:
                    raise ValueError
        elif args.backbone_type == "convvit_ecdp":
            if args.model_size == "small":
                self.backbone = convvit_ecdp.__dict__["convvit_ecdp_small_patch16"](args,
                                                                            num_bins=args.num_bins,
                                                                            drop_rate=args.drop_rate,
                                                                            attn_drop_rate=args.attn_drop_rate,
                                                                            drop_path_rate=args.drop_path_rate)
                if args.phase == "finetune_semseg":
                    self.decode_head = ft_dense_decoder.__dict__["finetune_decode_head_small"](args,
                                                                                               out_channels=args.num_classes)
                    self.auxiliary_head = ft_dense_decoder.__dict__["finetune_auxiliary_head_small"](args,
                                                                                                     out_channels=args.num_classes)
                elif args.phase == "finetune_flow":
                    self.decode_head = ft_dense_decoder.__dict__["finetune_decode_head_small"](args, out_channels=2)
                    self.auxiliary_head = ft_dense_decoder.__dict__["finetune_auxiliary_head_small"](args,
                                                                                                     out_channels=2)
                else:
                    raise ValueError
            elif args.model_size == "base":
                self.backbone = convvit_ecdp.__dict__["convvit_ecdp_base_patch16"](args,
                                                                            num_bins=args.num_bins,
                                                                            drop_rate=args.drop_rate,
                                                                            attn_drop_rate=args.attn_drop_rate,
                                                                            drop_path_rate=args.drop_path_rate)
                if args.phase == "finetune_semseg":
                    self.decode_head = ft_dense_decoder.__dict__["finetune_decode_head_base"](args,
                                                                                              out_channels=args.num_classes)
                    self.auxiliary_head = ft_dense_decoder.__dict__["finetune_auxiliary_head_base"](args,
                                                                                                    out_channels=args.num_classes)
                elif args.phase == "finetune_flow":
                    self.decode_head = ft_dense_decoder.__dict__["finetune_decode_head_base"](args, out_channels=2)
                    self.auxiliary_head = ft_dense_decoder.__dict__["finetune_auxiliary_head_base"](args,
                                                                                                    out_channels=2)
                else:
                    raise ValueError
        elif args.backbone_type == "vit_mem":
            if args.model_size == "small":
                self.backbone = vit_mem.__dict__["vit_mem_small_patch16"](args,
                                                                           num_bins=args.num_bins,
                                                                           drop_rate=args.drop_rate,
                                                                           attn_drop_rate=args.attn_drop_rate,
                                                                           drop_path_rate=args.drop_path_rate)
                if args.phase == "finetune_semseg":
                    self.decode_head = ft_dense_decoder.__dict__["finetune_decode_head_small"](args,
                                                                                               out_channels=args.num_classes)
                    self.auxiliary_head = ft_dense_decoder.__dict__["finetune_auxiliary_head_small"](args,
                                                                                                     out_channels=args.num_classes)
                elif args.phase == "finetune_flow":
                    self.decode_head = ft_dense_decoder.__dict__["finetune_decode_head_small"](args, out_channels=2)
                    self.auxiliary_head = ft_dense_decoder.__dict__["finetune_auxiliary_head_small"](args,
                                                                                                     out_channels=2)
                else:
                    raise ValueError
            elif args.model_size == "base":
                self.backbone = vit_mem.__dict__["vit_mem_base_patch16"](args,
                                                                          num_bins=args.num_bins,
                                                                          drop_rate=args.drop_rate,
                                                                          attn_drop_rate=args.attn_drop_rate,
                                                                          drop_path_rate=args.drop_path_rate)
                if args.phase == "finetune_semseg":
                    self.decode_head = ft_dense_decoder.__dict__["finetune_decode_head_base"](args,
                                                                                              out_channels=args.num_classes)
                    self.auxiliary_head = ft_dense_decoder.__dict__["finetune_auxiliary_head_base"](args,
                                                                                                    out_channels=args.num_classes)
                elif args.phase == "finetune_flow":
                    self.decode_head = ft_dense_decoder.__dict__["finetune_decode_head_base"](args, out_channels=2)
                    self.auxiliary_head = ft_dense_decoder.__dict__["finetune_auxiliary_head_base"](args,
                                                                                                         out_channels=2)
                else:
                    raise ValueError
            else:
                raise ValueError

        elif args.backbone_type == "swin_ecddp":
            self.backbone = swin_ecddp.__dict__["swin_ecddp_tiny_window7"](args=args,
                                                                           num_bins=args.num_bins,
                                                                           drop_rate=args.drop_rate,
                                                                           attn_drop_rate=args.attn_drop_rate,
                                                                           drop_path_rate=args.drop_path_rate)
            if args.phase == "finetune_semseg":
                self.decode_head = ft_dense_decoder.__dict__["finetune_decode_head_extend_small_swin"](args,
                                                                                                 out_channels=args.num_classes)
                self.auxiliary_head = ft_dense_decoder.__dict__["finetune_auxiliary_head_small_swin"](args,
                                                                                                out_channels=args.num_classes)
            elif args.phase == "finetune_flow":
                self.decode_head = ft_dense_decoder.__dict__["finetune_decode_head_extend_small_swin"](args, out_channels=2)
                self.auxiliary_head = ft_dense_decoder.__dict__["finetune_auxiliary_head_small_swin"](args,
                                                                                                out_channels=2)
            else:
                raise ValueError
        else:
            raise ValueError

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
            emb, out_embs, attn = self.backbone(x)

            decode_predict = self.decode_head(out_embs)
            aux_predict = self.auxiliary_head(out_embs)

            return emb, out_embs, attn, decode_predict, aux_predict

        elif self.backbone_type == "swin" or self.backbone_type == "swin_ecddp":
            emb_l1, emb_l2, emb_l3, emb_l4, emb_h, out_embs, attn = self.backbone(x)

            decode_predict = self.decode_head(out_embs)
            aux_predict = self.auxiliary_head(out_embs)

            return emb_l1, emb_l2, emb_l3, emb_l4, emb_h, out_embs, attn, decode_predict, aux_predict
        else:
            emb_l1, emb_l2, emb_h, out_embs, attn = self.backbone(x)

            decode_predict = self.decode_head(out_embs)
            aux_predict = self.auxiliary_head(out_embs)

            return emb_l1, emb_l2, emb_h, out_embs, attn, decode_predict, aux_predict


def finetune_dense_hub_model_small_patch16(args):
    model = FtDenseHubModel(args=args)
    return model

def finetune_dense_hub_model_base_patch16(args):
    model = FtDenseHubModel(args=args)
    return model

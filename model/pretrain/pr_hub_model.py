import copy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone import vit, convvit, swin
from model.pretrain import pr_rec_decoder
from model.sub_module.mlp_head import _build_mlp_2d
from utils.reshape import frame2emb, emb2patch_frame, patch_frame2emb


class PrHubModel(nn.Module):
    def __init__(self, args, patch_size=16, num_patches=196, embed_dim=1024, mlp_dim=4096,
                 proj_mlp_layers=3, pred_mlp_layers=2, norm_layer=nn.LayerNorm,
                 emb_frames_dim=512, queue_length=65536, T=0.07):
        super().__init__()

        self.args = args
        self.patch_size = patch_size
        self.T = T
        self.backbone_type = args.backbone_type
        self.mask_ratio = args.mask_ratio
        self.norm_pix_loss = args.norm_pix_loss

        if args.backbone_type == "vit":
            if args.model_size == "small":
                self.backbone = vit.__dict__["vit_small_patch16"](args=args,
                                                                  num_bins=args.num_bins,
                                                                  mask_ratio=args.mask_ratio,
                                                                  drop_rate=args.drop_rate,
                                                                  attn_drop_rate=args.attn_drop_rate,
                                                                  drop_path_rate=args.drop_path_rate)
            elif args.model_size == "base":
                self.backbone = vit.__dict__["vit_base_patch16"](args=args,
                                                                 num_bins=args.num_bins,
                                                                 mask_ratio=args.mask_ratio,
                                                                 drop_rate=args.drop_rate,
                                                                 attn_drop_rate=args.attn_drop_rate,
                                                                 drop_path_rate=args.drop_path_rate)
            else:
                raise ValueError

        elif args.backbone_type == "convvit":
            if args.model_size == "small":
                self.backbone = convvit.__dict__["convvit_small_patch16"](args=args,
                                                                          num_bins=args.num_bins,
                                                                          mask_ratio=args.mask_ratio,
                                                                          drop_rate=args.drop_rate,
                                                                          attn_drop_rate=args.attn_drop_rate,
                                                                          drop_path_rate=args.drop_path_rate)
            elif args.model_size == "base":
                self.backbone = convvit.__dict__["convvit_base_patch16"](args=args,
                                                                         num_bins=args.num_bins,
                                                                         mask_ratio=args.mask_ratio,
                                                                         drop_rate=args.drop_rate,
                                                                         attn_drop_rate=args.attn_drop_rate,
                                                                         drop_path_rate=args.drop_path_rate)
            else:
                raise ValueError
        elif args.backbone_type == "swin":
            self.backbone = swin.__dict__["swin_tiny_window7"](args=args,
                                                               num_bins=args.num_bins,
                                                               mask_ratio=args.mask_ratio,
                                                               drop_rate=args.drop_rate,
                                                               attn_drop_rate=args.attn_drop_rate,
                                                               drop_path_rate=args.drop_path_rate)
        else:
            raise ValueError

        if args.pr_phase == "rec" or args.pr_phase == "rec+con" or args.pr_phase == "rec-n":
            if args.backbone_type == "swin":
                self.pretrain_rec_decoder = pr_rec_decoder.__dict__["pretrain_rec_decoder_swin_tiny_patch32"](
                    frame_chans=args.frame_chans)
            else:
                self.pretrain_rec_decoder = pr_rec_decoder.__dict__["pretrain_rec_decoder_small_patch16"](
                    frame_chans=args.frame_chans)

        if args.pr_phase == "adj" or args.pr_phase == "_adj" or args.pr_phase == "con" or args.pr_phase == "adj-n" or \
                args.pr_phase == "con-n" or args.pr_phase == "rec+con":
            # create the queue
            if args.use_queue:
                self.queue_length = queue_length
                self.register_buffer("queue", torch.randn(embed_dim[-1], num_patches, queue_length))
                self.queue = nn.functional.normalize(self.queue, dim=0)
                self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

            self.emb_h_proj = _build_mlp_2d(proj_mlp_layers, embed_dim[-1], mlp_dim, embed_dim[-1])
            self.emb_h_pred = _build_mlp_2d(pred_mlp_layers, embed_dim[-1], mlp_dim, embed_dim[-1])

            self.norm_clip_emb = norm_layer(emb_frames_dim)

            if args.backbone_type == "swin":
                self.clip_emb_proj = nn.Conv2d(emb_frames_dim, embed_dim[-1], 2, stride=2)
            else:
                self.clip_emb_proj = nn.Linear(emb_frames_dim, embed_dim[-1], bias=False)

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

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):  # (2,196,384)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_length % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, :, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_length  # move pointer

        self.queue_ptr[0] = ptr

    # L_rec
    def reconstruct_loss(self, reconstruct_pred, sub_frame, mask):  # reconstruct_pred:(b,l,d) sub_frame:(b,c,h,w) d=p*p*n
        sub_frame = frame2emb(self.patch_size, sub_frame)

        if self.norm_pix_loss:
            mean = sub_frame.float().mean(dim=-1, keepdim=True)
            var = sub_frame.float().var(dim=-1, keepdim=True)
            sub_frame = (sub_frame - mean) / (var + 1.e-6) ** .5

        reconstruct_loss = (reconstruct_pred - sub_frame) ** 2  # (2,196,256)
        reconstruct_loss = reconstruct_loss.mean(dim=-1)

        if self.mask_ratio == 0:
            reconstruct_loss = reconstruct_loss.mean()
        else:
            reconstruct_loss = (mask * reconstruct_loss).sum() / mask.sum()

        return reconstruct_loss

    # L_con
    def contrastive_loss_queue(self, emb_h, clip_emb):  # emb_high(q): (2,196,384) emb_frame(k): (2,196,384)
        q = F.normalize(emb_h, dim=-1)
        k = F.normalize(clip_emb, dim=-1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("blc,blc->bl", [q, k]).unsqueeze(-1)  # (2,196,1)
        # negative logits: NxK
        l_neg = torch.einsum("blc,clk->blk", [q, self.queue.clone().detach()])  # (2,196,65536)

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=-1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros((logits.shape[0], logits.shape[1]), dtype=torch.long).to(logits.device)  # (2,196)
        contrastive_loss = nn.CrossEntropyLoss()(logits.permute(0, 2, 1), labels)  # (2,65537,196)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return contrastive_loss

    def contrastive_loss(self, emb_h, clip_emb):
        # normalize
        q = F.normalize(emb_h, dim=-1)
        k = F.normalize(clip_emb, dim=-1)
        # gather all targets
        if self.args.distributed:
            k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nlc,mlc->nlm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        if self.args.distributed:
            labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).unsqueeze(-1).repeat(
                1, logits.shape[1]).to(logits.device)
        else:
            labels = (torch.arange(N, dtype=torch.long)).unsqueeze(-1).repeat(1, logits.shape[1]).to(logits.device)

        contrastive_loss = nn.CrossEntropyLoss()(logits.permute(0, 2, 1), labels)

        return contrastive_loss

    def forward(self, events_voxel_grid, supp_data, is_rec=False):
        if is_rec:
            sub_frame = supp_data

            if self.args.backbone_type == "swin":
                emb_l1, emb_l2, emb_l3, emb_l4, emb_lh, coords_l1, coords_l2, coords_l3, coords_l4, \
                    mask, ids_restore, attn = self.backbone(events_voxel_grid, mask=True)
            else:
                emb_l1, emb_l2, emb_lh, mask, ids_restore = self.backbone(events_voxel_grid, mask=True)
            reconstruct_pred = self.pretrain_rec_decoder(emb_lh, ids_restore)  # (b,l,d)
            reconstruct_loss = self.reconstruct_loss(reconstruct_pred, sub_frame, mask)

            if self.args.backbone_type == "swin":
                return reconstruct_loss, emb_l1, emb_l2, emb_l3, emb_l4, emb_lh, \
                    coords_l1, coords_l2, coords_l3, coords_l4, reconstruct_pred, mask, ids_restore, attn
            else:
                return reconstruct_loss, emb_l1, emb_l2, emb_lh, reconstruct_pred, mask, ids_restore

        else:
            clip_emb = supp_data
            if self.args.backbone_type == "swin":
                _, _, _, _, emb_h, attn = self.backbone(events_voxel_grid)
            else:
                _, _, emb_h, attn = self.backbone(events_voxel_grid)
            emb_h_org = copy.deepcopy(emb_h.detach())

            clip_emb = self.norm_clip_emb(clip_emb[:, 1:, :])
            clip_emb_org = copy.deepcopy(clip_emb.detach())
            if self.backbone_type == "swin":
                clip_emb_proj = patch_frame2emb(self.clip_emb_proj(emb2patch_frame(clip_emb)))  # (2,197,512) -> (2,196,384)
            else:
                clip_emb_proj = self.clip_emb_proj(clip_emb)

            for i, module in enumerate(self.emb_h_proj):
                if type(module) == nn.BatchNorm2d:
                    emb_h = emb2patch_frame(emb_h)
                    emb_h = module(emb_h)
                    emb_h = patch_frame2emb(emb_h)
                else:
                    emb_h = module(emb_h)

            for i, module in enumerate(self.emb_h_pred):
                if type(module) == nn.BatchNorm2d:
                    emb_h = emb2patch_frame(emb_h)
                    emb_h = module(emb_h)
                    emb_h = patch_frame2emb(emb_h)
                else:
                    emb_h = module(emb_h)
            emb_h_proj = emb_h

            if self.args.use_queue:
                contrastive_loss = self.contrastive_loss_queue(emb_h_proj, clip_emb_proj)
            else:
                contrastive_loss = self.contrastive_loss(emb_h_proj, clip_emb_proj)

            return contrastive_loss, emb_h_org, emb_h_proj, clip_emb_org, clip_emb_proj, attn


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def pretrain_hub_model_small_patch16(args, **kwargs):
    model = PrHubModel(args=args,
        patch_size=16, num_patches=196, embed_dim=[128, 256, 384], mlp_dim=4096,
        proj_mlp_layers=3, pred_mlp_layers=2, norm_layer=nn.LayerNorm, **kwargs
    )
    return model

def pretrain_hub_model_swin_tiny_patch16(args, **kwargs):
    model = PrHubModel(args=args,
        patch_size=32, num_patches=49, embed_dim=[96, 192, 384, 768], mlp_dim=4096,
        proj_mlp_layers=3, pred_mlp_layers=2, norm_layer=nn.LayerNorm, **kwargs
    )
    return model

def pretrain_hub_model_base_patch16(args, **kwargs):
    model = PrHubModel(args=args,
        patch_size=16, num_patches=196, embed_dim=[256, 384, 768], mlp_dim=4096,
        proj_mlp_layers=3, pred_mlp_layers=2, norm_layer=nn.LayerNorm, **kwargs  # norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    return model

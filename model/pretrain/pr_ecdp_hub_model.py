import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone import vit_ecdp, convvit_ecdp
from model.sub_module.mlp_head import _build_mlp_1d


class PrECDPModel(nn.Module):
    def __init__(self, args, patch_size=16, embed_dim=1024, mlp_dim=4096, proj_dim=256,
                 proj_mlp_layers=3, pred_mlp_layers=2, emb_frames_dim=512, queue_length=65536,
                 T_image=0.1, T_event=0.2):
        super().__init__()

        self.args = args
        self.patch_size = patch_size[0] * patch_size[1] * patch_size[2]
        self.queue_length = queue_length
        self.T_image, self.T_event = T_image, T_event

        self.backbone_type = args.backbone_type
        self.mask_ratio = args.mask_ratio

        if args.backbone_type == "vit_ecdp":
            if args.model_size == "small":
                self.backbone = vit_ecdp.__dict__["vit_ecdp_small_patch16"](args=args,
                                                                            num_bins=args.num_bins,
                                                                            mask_ratio=args.mask_ratio,
                                                                            drop_rate=args.drop_rate,
                                                                            attn_drop_rate=args.attn_drop_rate,
                                                                            drop_path_rate=args.drop_path_rate)
                self.backbone_ema = vit_ecdp.__dict__["vit_ecdp_small_patch16"](args=args,
                                                                                num_bins=args.num_bins,
                                                                                mask_ratio=args.mask_ratio,
                                                                                drop_rate=args.drop_rate,
                                                                                attn_drop_rate=args.attn_drop_rate,
                                                                                drop_path_rate=args.drop_path_rate)
            elif args.model_size == "base":
                self.backbone = vit_ecdp.__dict__["vit_ecdp_base_patch16"](args=args,
                                                                           num_bins=args.num_bins,
                                                                           mask_ratio=args.mask_ratio,
                                                                           drop_rate=args.drop_rate,
                                                                           attn_drop_rate=args.attn_drop_rate,
                                                                           drop_path_rate=args.drop_path_rate)
                self.backbone_ema = vit_ecdp.__dict__["vit_ecdp_base_patch16"](args=args,
                                                                               num_bins=args.num_bins,
                                                                               mask_ratio=args.mask_ratio,
                                                                               drop_rate=args.drop_rate,
                                                                               attn_drop_rate=args.attn_drop_rate,
                                                                               drop_path_rate=args.drop_path_rate)
            else:
                raise ValueError

        elif args.backbone_type == "convvit_ecdp":
            if args.model_size == "small":
                self.backbone = convvit_ecdp.__dict__["convvit_ecdp_small_patch16"](args=args,
                                                                                    num_bins=args.num_bins,
                                                                                    mask_ratio=args.mask_ratio,
                                                                                    drop_rate=args.drop_rate,
                                                                                    attn_drop_rate=args.attn_drop_rate,
                                                                                    drop_path_rate=args.drop_path_rate)
                self.backbone_ema = convvit_ecdp.__dict__["convvit_ecdp_small_patch16"](args=args,
                                                                                        num_bins=args.num_bins,
                                                                                        mask_ratio=args.mask_ratio,
                                                                                        drop_rate=args.drop_rate,
                                                                                        attn_drop_rate=args.attn_drop_rate,
                                                                                        drop_path_rate=args.drop_path_rate)
            elif args.model_size == "base":
                self.backbone = convvit_ecdp.__dict__["convvit_ecdp_base_patch16"](args=args,
                                                                                   num_bins=args.num_bins,
                                                                                   mask_ratio=args.mask_ratio,
                                                                                   drop_rate=args.drop_rate,
                                                                                   attn_drop_rate=args.attn_drop_rate,
                                                                                   drop_path_rate=args.drop_path_rate)
                self.backbone_ema = convvit_ecdp.__dict__["convvit_ecdp_base_patch16"](args=args,
                                                                                       num_bins=args.num_bins,
                                                                                       mask_ratio=args.mask_ratio,
                                                                                       drop_rate=args.drop_rate,
                                                                                       attn_drop_rate=args.attn_drop_rate,
                                                                                       drop_path_rate=args.drop_path_rate)
            else:
                raise ValueError
        else:
            raise ValueError

        for param_q, param_k in zip(self.backbone.parameters(), self.backbone_ema.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.backbone.event_head_proj = _build_mlp_1d(proj_mlp_layers, embed_dim[-1], mlp_dim, proj_dim)
        self.backbone_ema.event_head_proj = _build_mlp_1d(proj_mlp_layers, embed_dim[-1], mlp_dim, proj_dim)
        self.backbone.image_head_proj = _build_mlp_1d(proj_mlp_layers, embed_dim[-1], mlp_dim, proj_dim)
        self.backbone_ema.image_head_proj = _build_mlp_1d(proj_mlp_layers, embed_dim[-1], mlp_dim, proj_dim)

        self.event_head_pred = _build_mlp_1d(pred_mlp_layers, proj_dim, mlp_dim, proj_dim)
        self.image_head_pred = _build_mlp_1d(pred_mlp_layers, proj_dim, mlp_dim, proj_dim)

        self.clip_emb_proj = nn.Linear(emb_frames_dim, proj_dim, bias=False)

        # create the queue
        if args.use_queue:
            self.register_buffer("queue_image", torch.randn(proj_dim, queue_length))
            self.queue_image = F.normalize(self.queue_image, dim=0)
            self.register_buffer("queue_image_ptr", torch.zeros(1, dtype=torch.long))
            self.register_buffer("queue_event", torch.randn(proj_dim, queue_length))
            self.queue_event = F.normalize(self.queue_event, dim=0)
            self.register_buffer("queue_event_ptr", torch.zeros(1, dtype=torch.long))

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
    def _momentum_update(self, m):
        for param_q, param_k in zip(self.backbone.parameters(), self.backbone_ema.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)

    def vector_project(self, v1, v2):
        #v2 = v2.detach()
        #v1 = F.normalize(v1, dim=1)
        #v2 = F.normalize(v2, dim=1)
        #return (v1 * v2).sum(1, True) * v2
        return (v1 * v2) * (v2 / sum(v ** 2 for v in v2))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, queue, queue_ptr, keys):  # (2,196,384)
        batch_size = keys.shape[0]
        ptr = int(queue_ptr)
        assert self.queue_length % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_length  # move pointer

        queue_ptr[0] = ptr

    # L_con
    def contrastive_loss_queue(self, q, k, T, queue, queue_ptr, l2_norm=True):  # emb_high(q): (2,196,384) emb_frame(k): (2,196,384)
        if l2_norm:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("bc,bc->b", [q, k]).unsqueeze(-1)  # (2,1)
        # negative logits: NxK
        l_neg = torch.einsum("bc,ck->bk", [q, queue.clone().detach()])  # (2,196,65536)

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=-1)

        # apply temperature
        logits /= T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)  # (2,196)
        contrastive_loss = nn.CrossEntropyLoss()(logits, labels) # * (2 * T)

        # dequeue and enqueue
        self._dequeue_and_enqueue(queue, queue_ptr, k)

        return contrastive_loss

    def contrastive_loss(self, q, k, T, l2_norm=True):  # emb_high(q): (2,196,384) emb_frame(k): (2,196,384)
        if l2_norm:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
        # gather all targets
        if self.args.distributed:
            k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / T
        N = logits.shape[0]  # batch size per GPU
        if self.args.distributed:
            labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).to(logits.device)
        else:
            labels = (torch.arange(N, dtype=torch.long)).to(logits.device)

        contrastive_loss = nn.CrossEntropyLoss()(logits, labels) * (2 * T)

        return contrastive_loss

    def sinkhorn(self, out):
        Q = torch.exp(out).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1]  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if self.args.distributed:
            torch.distributed.all_reduce(sum_Q)
        Q = Q / sum_Q.detach()

        for it in range(3):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if self.args.distributed:
                torch.distributed.all_reduce(sum_of_rows)
            Q = Q / sum_of_rows.detach()
            Q = Q / K

            # normalize each column: total weight per sample must be 1/B
            Q = Q / torch.sum(Q, dim=0, keepdim=True)
            Q = Q / B

        Q = Q * B  # the colomns must sum to 1 so that Q is an assignment

        return Q.t()

    def kl_loss(self, q, k):
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        if self.args.distributed:
            q = concat_all_gather(q)
            k = concat_all_gather(k)

        f_log = nn.LogSoftmax(dim=-1)
        f = nn.Softmax(dim=-1)

        q = torch.einsum('nc,mc->nm', [q, q]) / self.T_image
        k = torch.einsum('nc,mc->nm', [k, k]) / self.T_image

        return nn.KLDivLoss(reduction="batchmean", log_target=False)(f_log(q), self.sinkhorn(k))  # f(k)

    def forward(self, events_image_q, events_image_k, supp_data, ema_m):
        clip_emb = supp_data[:, 0, :]
        emb_event_q, emb_image_q, mask_q, ids_restore_q, attn_q = self.backbone(events_image_q, mask=True)
        emb_event_q_org = copy.deepcopy(emb_event_q.detach())
        emb_image_q_org = copy.deepcopy(emb_image_q.detach())

        emb_event_q = self.backbone.event_head_proj(emb_event_q)
        emb_image_q = self.backbone.image_head_proj(emb_image_q)

        emb_event_q = self.event_head_pred(emb_event_q)
        emb_image_q = self.image_head_pred(emb_image_q)

        with torch.no_grad():
            self._momentum_update(ema_m)
            emb_event_k, emb_image_k, mask_k, ids_restore_k, attn_k = self.backbone_ema(events_image_k, mask=True)

            emb_event_k = self.backbone_ema.event_head_proj(emb_event_k)

        clip_emb_org = copy.deepcopy(clip_emb.detach())
        clip_emb_proj = self.clip_emb_proj(clip_emb)  # (2,512) -> (2,256)

        emb_event_q = self.vector_project(emb_event_q, clip_emb_proj)
        emb_event_k = self.vector_project(emb_event_k, clip_emb_proj)

        if self.args.use_queue:
            contrastive_loss_image = self.contrastive_loss_queue(emb_image_q, clip_emb_proj,
                                                                 self.T_image, self.queue_image, self.queue_image_ptr)
            contrastive_loss_event = self.contrastive_loss_queue(emb_event_q, emb_event_k,
                                                                 self.T_event, self.queue_event, self.queue_event_ptr,
                                                                 l2_norm=False)
        else:
            contrastive_loss_image = self.contrastive_loss(emb_image_q, clip_emb_proj, self.T_image)
            contrastive_loss_event = self.contrastive_loss(emb_event_q, emb_event_k, self.T_event, l2_norm=False)

        kl_loss = self.kl_loss(emb_image_q, clip_emb_proj)

        return contrastive_loss_image, contrastive_loss_event, kl_loss, \
            emb_event_q_org, emb_image_q_org, emb_event_q, emb_image_q, clip_emb_org, clip_emb_proj, \
            mask_q, ids_restore_q, attn_q, mask_k, ids_restore_k, attn_k


def pretrain_ecdp_model_small_patch16(args, **kwargs):
    model = PrECDPModel(args=args,
        patch_size=[4, 2, 2], embed_dim=[128, 256, 384], mlp_dim=4096, proj_dim=256,
        proj_mlp_layers=3, pred_mlp_layers=2, **kwargs
    )
    return model

def pretrain_ecdp_model_base_patch16(args, **kwargs):
    model = PrECDPModel(args=args,
        patch_size=[4, 2, 2], embed_dim=[256, 384, 768], mlp_dim=4096, proj_dim=256,
        proj_mlp_layers=3, pred_mlp_layers=2, **kwargs
    )
    return model

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

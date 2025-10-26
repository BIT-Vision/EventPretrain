import os
from pathlib import Path
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')
# from matplotlib import pyplot as plt
import copy

import torch

from visualize.visualize_utils.make_events_preview import make_events_preview, make_events_preview_norm
from utils.reshape import emb2patch_frame, emb2frame


def vis_pr_rec(args, events_voxel_grid, emb_l1, emb_l2, emb_lh,
               sub_frame, reconstruct_pred, mask, ids_restore,
               image_name, epoch):
    plt.figure()
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.02, hspace=0.3)

    events_voxel_grid, emb_l1, emb_l2, emb_lh, sub_frame, reconstruct_pred, mask, ids_restore = \
        events_voxel_grid.detach().cpu(), emb_l1.detach().cpu(), emb_l2.detach().cpu(), emb_lh.detach().cpu(), \
            sub_frame.detach().cpu(), reconstruct_pred.detach().cpu(), mask.detach().cpu(), ids_restore.detach().cpu()

    # events frame (row:1 col:1)
    plt.subplot(5, 4, 1)
    events_frame = make_events_preview(events_voxel_grid)
    plt.imshow(events_frame, cmap='gray')
    plt.axis('off')
    plt.title("events frame", fontsize=5)

    # events frame norm (row:1 col:2)
    plt.subplot(5, 4, 2)
    events_frame_norm = make_events_preview_norm(events_voxel_grid)
    plt.imshow(events_frame_norm, cmap='gray')
    plt.axis('off')
    plt.title("events frame norm", fontsize=5)

    # emb_l1 (row:2)
    if args.backbone_type == "convvit":
        for i in range(4):
            plt.subplot(5, 4, i + 5)
            emb_l1_per_chans = emb_l1[i: i + 1]
            emb_l1_per_chans = torch.einsum('chw->hwc', emb_l1_per_chans).numpy()
            plt.imshow(emb_l1_per_chans, cmap='viridis')
            plt.axis('off')
            plt.title("emb_l1_" + str(i + 1), fontsize=5)
    else:
        emb_l1_norm = (emb_l1 - emb_l1.min()) / (emb_l1.max() - emb_l1.min())
        mask_emb = torch.zeros(ids_restore.shape[0] - emb_l1.shape[0], emb_l1.shape[1])
        emb_l1_ = torch.cat([emb_l1_norm, mask_emb], dim=0)
        emb_l1 = torch.gather(emb_l1_, dim=0,
                              index=ids_restore.unsqueeze(-1).repeat(1, 1, emb_l1.shape[1]).squeeze(0))  # unshuffle
        emb_l1 = emb2patch_frame(emb_l1.unsqueeze(0)).squeeze(0)
        for i in range(4):
            plt.subplot(5, 4, i + 5)
            emb_l1_per_chans = emb_l1[i: i + 1]
            emb_l1_per_chans = torch.einsum('chw->hwc', emb_l1_per_chans).numpy()
            plt.imshow(emb_l1_per_chans, cmap='viridis')
            plt.axis('off')
            plt.title("emb_l1_" + str(i + 1), fontsize=5)

    # emb_l2 (row:3)
    if args.backbone_type == "convvit":
        for i in range(4):
            plt.subplot(5, 4, i + 9)
            emb_l2_per_chans = emb_l2[i: i + 1]
            emb_l2_per_chans = torch.einsum('chw->hwc', emb_l2_per_chans).numpy()
            plt.imshow(emb_l2_per_chans, cmap='viridis')
            plt.axis('off')
            plt.title("emb_l2_" + str(i + 1), fontsize=5)
    else:
        emb_l2_norm = (emb_l2 - emb_l2.min()) / (emb_l2.max() - emb_l2.min())
        mask_emb = torch.zeros(ids_restore.shape[0] - emb_l2.shape[0], emb_l2.shape[1])
        emb_l2_ = torch.cat([emb_l2_norm, mask_emb], dim=0)
        emb_l2 = torch.gather(emb_l2_, dim=0,
                              index=ids_restore.unsqueeze(-1).repeat(1, 1, emb_l2.shape[1]).squeeze(0))  # unshuffle
        emb_l2 = emb2patch_frame(emb_l2.unsqueeze(0)).squeeze(0)
        for i in range(4):
            plt.subplot(5, 4, i + 9)
            emb_l2_per_chans = emb_l2[i: i + 1]
            emb_l2_per_chans = torch.einsum('chw->hwc', emb_l2_per_chans).numpy()
            plt.imshow(emb_l2_per_chans, cmap='viridis')
            plt.axis('off')
            plt.title("emb_l2_" + str(i + 1), fontsize=5)

    # emb_lh (row:4)
    emb_lh_norm = (emb_lh - emb_lh.min()) / (emb_lh.max() - emb_lh.min())
    mask_emb = torch.zeros(ids_restore.shape[0] - emb_lh.shape[0], emb_lh.shape[1])
    emb_lh_ = torch.cat([emb_lh_norm, mask_emb], dim=0)
    emb_lh = torch.gather(emb_lh_, dim=0,
                          index=ids_restore.unsqueeze(-1).repeat(1, 1, emb_lh.shape[1]).squeeze(0))  # unshuffle
    emb_lh = emb2patch_frame(emb_lh.unsqueeze(0)).squeeze(0)
    for i in range(4):
        plt.subplot(5, 4, i + 13)
        emb_lh_per_chans = emb_lh[i: i + 1]
        emb_lh_per_chans = torch.einsum('chw->hwc', emb_lh_per_chans).numpy()
        plt.imshow(emb_lh_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("emb_l_h_" + str(i + 1), fontsize=5)

    # sub frame (row:5 col:1)
    plt.subplot(5, 4, 17)
    sub_frame_norm = (sub_frame - sub_frame.min()) / (sub_frame.max() - sub_frame.min())
    sub_frame = torch.einsum('chw->hwc', sub_frame_norm)
    plt.imshow(sub_frame, cmap='gray')
    plt.axis('off')
    plt.title("sub frame", fontsize=5)

    # mask upsampling
    mask = mask
    mask = mask.unsqueeze(-1).repeat(1, args.patch_size ** 2 * args.frame_chans)  # (H*W, p*p*1)
    mask = emb2frame(args, mask.unsqueeze(0), args.frame_chans).squeeze(0)  # 1 is removing, 0 is keeping
    mask = torch.einsum('chw->hwc', mask)
    mask = mask.int()

    # masked sub frame (row:5 col:2)
    plt.subplot(5, 4, 18)
    masked_sub_frame = sub_frame * (1 - mask)
    plt.imshow(masked_sub_frame, cmap='gray')
    plt.axis('off')
    plt.title("masked sub frame", fontsize=5)

    # reconstruct sub frame (row:5 col:3)
    plt.subplot(5, 4, 19)
    reconstruct_pred_norm = (reconstruct_pred - reconstruct_pred.min()) / (
            reconstruct_pred.max() - reconstruct_pred.min())
    reconstruct_pred = emb2frame(args, reconstruct_pred_norm.unsqueeze(0), args.frame_chans)  # (1,1,224,224)
    reconstruct_frame = torch.einsum('bchw->bhwc', reconstruct_pred).squeeze(0)  # (224,224,1)
    plt.imshow(reconstruct_frame, cmap='gray')
    plt.axis('off')
    plt.title("reconstruct frame", fontsize=5)

    # reconstruct visible sub frame (row:5 col:4)
    plt.subplot(5, 4, 20)
    reconstruct_visible_sub_frame = sub_frame * (1 - mask) + reconstruct_frame * mask
    plt.imshow(reconstruct_visible_sub_frame, cmap='gray')
    plt.axis('off')
    plt.title("reconstruct visible sub frame", fontsize=5)

    plt.suptitle(image_name, fontsize=10)
    Path(os.path.join(args.output_dir, args.rec_dir, args.vis_train_dir)).mkdir(parents=True, exist_ok=True)
    figure_name = ("epoch_0{}.png").format(epoch + 1) if (epoch + 1) < 10 else ("epoch_{}.png").format(epoch + 1)
    plt.savefig(os.path.join(args.output_dir, args.rec_dir, args.vis_train_dir, figure_name),
                bbox_inches='tight', pad_inches=0.1, dpi=100)
    plt.show()
    plt.close()

def vis_pr_rec_swin(args, events_voxel_grid, emb_l1, emb_l2, emb_l3, emb_l4, emb_lh,
                    coords_l1, coords_l2, coords_l3, coords_l4,
                    sub_frame, reconstruct_pred, mask, ids_restore, attn,
                    image_name, epoch):
    plt.figure()
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.02, hspace=0.3)

    events_voxel_grid, emb_l1, emb_l2, emb_l3, emb_l4, emb_lh, \
    coords_l1, coords_l2, coords_l3, coords_l4, \
    sub_frame, reconstruct_pred, mask, ids_restore, attn = \
        events_voxel_grid.detach().cpu(), emb_l1.detach().cpu(), emb_l2.detach().cpu(), \
        emb_l3.detach().cpu(), emb_l4.detach().cpu(), emb_lh.detach().cpu(), \
        coords_l1.detach().cpu(), coords_l2.detach().cpu(), coords_l3.detach().cpu(), coords_l4.detach().cpu(), \
        sub_frame.detach().cpu(), reconstruct_pred.detach().cpu(), mask.detach().cpu(), \
        ids_restore.detach().cpu(), attn.detach().cpu()

    # events frame (row:1 col:1)
    plt.subplot(7, 4, 1)
    events_frame = make_events_preview(events_voxel_grid)
    plt.imshow(events_frame, cmap='gray')
    plt.axis('off')
    plt.title("events frame", fontsize=5)

    # events frame norm (row:1 col:2)
    plt.subplot(7, 4, 2)
    events_frame_norm = make_events_preview_norm(events_voxel_grid)
    plt.imshow(events_frame_norm, cmap='gray')
    plt.axis('off')
    plt.title("events frame norm", fontsize=5)

    # attention map (row:1 col:3)
    attn = attn.mean(dim=0).mean(dim=0).unsqueeze(-1)
    attn_norm = (attn - attn.min()) / (attn.max() - attn.min())
    mask_attn = torch.zeros(ids_restore.shape[0] - attn.shape[0], attn.shape[1])
    attn_norm = torch.cat([attn_norm, mask_attn], dim=0)
    attn_norm = torch.gather(attn_norm, dim=0,
                             index=ids_restore.unsqueeze(-1).repeat(1, 1, attn_norm.shape[1]).squeeze(0))  # unshuffle
    attn_norm = emb2patch_frame(attn_norm.unsqueeze(0)).squeeze(0).squeeze(0)
    plt.imshow(attn_norm, cmap='viridis')
    plt.axis('off')
    plt.title("attn", fontsize=5)

    # emb_l1 (row:2) (1536,96) -> (3136,96) -> (56,56,96)
    _emb_l1 = torch.zeros((56 * 56, emb_l1.shape[-1]))
    emb_l1_norm = (emb_l1 - emb_l1.min()) / (emb_l1.max() - emb_l1.min())
    _emb_l1[coords_l1[:, 0] * 56 + coords_l1[:, 1], :] = emb_l1_norm.to(torch.float32)
    _emb_l1 = _emb_l1.reshape(int(_emb_l1.shape[0] ** .5), int(_emb_l1.shape[0] ** .5), _emb_l1.shape[1]).permute(2, 0, 1)

    for i in range(4):
        plt.subplot(7, 4, i + 5)
        emb_l1_per_chans = _emb_l1[i: i + 1]
        emb_l1_per_chans = torch.einsum('chw->hwc', emb_l1_per_chans).numpy()
        plt.imshow(emb_l1_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("emb_l1_" + str(i + 1), fontsize=5)

    # emb_l2 (row:3) (384,192) -> (784,192) -> (28,28,192)
    _emb_l2 = torch.zeros((28 * 28, emb_l2.shape[-1]))
    emb_l2_norm = (emb_l2 - emb_l2.min()) / (emb_l2.max() - emb_l2.min())
    _emb_l2[coords_l2[:, 0] * 28 + coords_l2[:, 1], :] = emb_l2_norm.to(torch.float32)
    _emb_l2 = _emb_l2.reshape(int(_emb_l2.shape[0] ** .5), int(_emb_l2.shape[0] ** .5), _emb_l2.shape[1]).permute(2, 0, 1)

    for i in range(4):
        plt.subplot(7, 4, i + 9)
        emb_l2_per_chans = _emb_l2[i: i + 1]
        emb_l2_per_chans = torch.einsum('chw->hwc', emb_l2_per_chans).numpy()
        plt.imshow(emb_l2_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("emb_l2_" + str(i + 1), fontsize=5)

    # emb_l3 (row:4) (96,384) -> (196,384) -> (14,14,384)
    _emb_l3 = torch.zeros((14 * 14, emb_l3.shape[-1]))
    emb_l3_norm = (emb_l3 - emb_l3.min()) / (emb_l3.max() - emb_l3.min())
    _emb_l3[coords_l3[:, 0] * 14 + coords_l3[:, 1], :] = emb_l3_norm.to(torch.float32)
    _emb_l3 = _emb_l3.reshape(int(_emb_l3.shape[0] ** .5), int(_emb_l3.shape[0] ** .5), _emb_l3.shape[1]).permute(2, 0, 1)

    for i in range(4):
        plt.subplot(7, 4, i + 13)
        emb_l3_per_chans = _emb_l3[i: i + 1]
        emb_l3_per_chans = torch.einsum('chw->hwc', emb_l3_per_chans).numpy()
        plt.imshow(emb_l3_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("emb_l3_" + str(i + 1), fontsize=5)

    # emb_l4 (row:5) (24,768) -> (49,768) -> (7,7,768)
    _emb_l4 = torch.zeros((7 * 7, emb_l4.shape[-1]))
    emb_l4_norm = (emb_l4 - emb_l4.min()) / (emb_l4.max() - emb_l4.min())
    _emb_l4[coords_l4[:, 0] * 7 + coords_l4[:, 1], :] = emb_l4_norm.to(torch.float32)
    _emb_l4 = _emb_l4.reshape(int(_emb_l4.shape[0] ** .5), int(_emb_l4.shape[0] ** .5), _emb_l4.shape[1]).permute(2, 0, 1)

    # emb_l4_norm = (emb_l4 - emb_l4.min()) / (emb_l4.max() - emb_l4.min())
    # mask_tokens = torch.zeros((ids_restore_1.shape[0] - emb_l4.shape[0], emb_l4.shape[1]))
    # emb_l4_ = torch.cat([emb_l4_norm, mask_tokens], dim=0)  # no cls token
    # emb_l4 = torch.gather(emb_l4_, dim=0, index=ids_restore_1.unsqueeze(-1).repeat(1, 1, emb_l4.shape[1]).squeeze(0))
    # emb_l4 = emb2patch_frame(emb_l4.unsqueeze(0)).squeeze(0)

    for i in range(4):
        plt.subplot(7, 4, i + 17)
        emb_l4_per_chans = _emb_l4[i: i + 1]
        emb_l4_per_chans = torch.einsum('chw->hwc', emb_l4_per_chans).numpy()
        plt.imshow(emb_l4_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("emb_l4_" + str(i + 1), fontsize=5)

    # emb_lh (row:6)
    _emb_lh = torch.zeros((7 * 7, emb_lh.shape[-1]))
    emb_lh_norm = (emb_lh - emb_lh.min()) / (emb_lh.max() - emb_lh.min())
    _emb_lh[coords_l4[:, 0] * 7 + coords_l4[:, 1], :] = emb_lh_norm.to(torch.float32)
    _emb_lh = _emb_lh.reshape(int(_emb_lh.shape[0] ** .5), int(_emb_lh.shape[0] ** .5), _emb_lh.shape[1]).permute(2, 0, 1)

    # emb_lh_norm = (emb_lh - emb_lh.min()) / (emb_lh.max() - emb_lh.min())
    # mask_emb = torch.zeros(ids_restore_1.shape[0] - emb_lh.shape[0], emb_lh.shape[1])
    # emb_lh_ = torch.cat([emb_lh_norm, mask_emb], dim=0)
    # emb_lh = torch.gather(emb_lh_, dim=0,
    #                       index=ids_restore_1.unsqueeze(-1).repeat(1, 1, emb_lh.shape[1]).squeeze(0))  # unshuffle
    # emb_lh = emb2patch_frame(emb_lh.unsqueeze(0)).squeeze(0)

    for i in range(4):
        plt.subplot(7, 4, i + 21)
        emb_lh_per_chans = _emb_lh[i: i + 1]
        emb_lh_per_chans = torch.einsum('chw->hwc', emb_lh_per_chans).numpy()
        plt.imshow(emb_lh_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("emb_l_h_" + str(i + 1), fontsize=5)

    # sub frame (row:7 col:1)
    plt.subplot(7, 4, 25)
    sub_frame_norm = (sub_frame - sub_frame.min()) / (sub_frame.max() - sub_frame.min())
    sub_frame = torch.einsum('chw->hwc', sub_frame_norm)
    plt.imshow(sub_frame, cmap='gray')
    plt.axis('off')
    plt.title("sub frame", fontsize=5)

    # mask upsampling
    mask = mask.unsqueeze(-1).repeat(1, args.patch_size ** 2 * args.frame_chans)  # (H*W, p*p*1)
    mask = emb2frame(args, mask.unsqueeze(0), args.frame_chans).squeeze(0)  # 1 is removing, 0 is keeping
    mask = torch.einsum('chw->hwc', mask)
    mask = mask.int()

    # masked sub frame (row:7 col:2)
    plt.subplot(7, 4, 26)
    masked_sub_frame = sub_frame * (1 - mask)
    plt.imshow(masked_sub_frame, cmap='gray')
    plt.axis('off')
    plt.title("masked sub frame", fontsize=5)

    # reconstruct sub frame (row:7 col:3)
    plt.subplot(7, 4, 27)
    reconstruct_pred_norm = (reconstruct_pred - reconstruct_pred.min()) / (
            reconstruct_pred.max() - reconstruct_pred.min())
    reconstruct_pred = emb2frame(args, reconstruct_pred_norm.unsqueeze(0), args.frame_chans)  # (1,1,224,224)
    reconstruct_frame = torch.einsum('bchw->bhwc', reconstruct_pred).squeeze(0)  # (224,224,1)
    plt.imshow(reconstruct_frame, cmap='gray')
    plt.axis('off')
    plt.title("reconstruct frame", fontsize=5)

    # reconstruct visible sub frame (row:7 col:4)
    plt.subplot(7, 4, 28)
    reconstruct_visible_sub_frame = sub_frame * (1 - mask) + reconstruct_frame * mask
    plt.imshow(reconstruct_visible_sub_frame, cmap='gray')
    plt.axis('off')
    plt.title("reconstruct visible sub frame", fontsize=5)

    plt.suptitle(image_name, fontsize=10)
    Path(os.path.join(args.output_dir, args.rec_dir, args.vis_train_dir)).mkdir(parents=True, exist_ok=True)
    figure_name = ("epoch_0{}.png").format(epoch + 1) if (epoch + 1) < 10 else ("epoch_{}.png").format(epoch + 1)
    plt.savefig(os.path.join(args.output_dir, args.rec_dir, args.vis_train_dir, figure_name),
                bbox_inches='tight', pad_inches=0.1, dpi=100)
    plt.show()
    plt.close()

def vis_pr_con(args, events_voxel_grid, emb_h_org, emb_h_proj,
               clip_emb_org, clip_emb_proj, attn,
               image_name, epoch):
    plt.figure()
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.02, hspace=0.3)

    events_voxel_grid, emb_h_org, emb_h_proj, clip_emb_org, clip_emb_proj, attn = \
        events_voxel_grid.detach().cpu(), emb_h_org.detach().cpu(), emb_h_proj.detach().cpu(), \
            clip_emb_org.detach().cpu(), clip_emb_proj.detach().cpu(), attn.detach().cpu()

    # events frame (row:1 col:1)
    plt.subplot(5, 4, 1)
    events_frame = make_events_preview(events_voxel_grid)
    plt.imshow(events_frame, cmap='gray')
    plt.axis('off')
    plt.title("events frame", fontsize=5)

    # events frame norm (row:1 col:2)
    plt.subplot(5, 4, 2)
    events_frame_norm = make_events_preview_norm(events_voxel_grid)
    plt.imshow(events_frame_norm, cmap='gray')
    plt.axis('off')
    plt.title("events frame norm", fontsize=5)

    # attention map (row:1 col:2)
    plt.subplot(5, 4, 3)
    attn_ = attn.mean(dim=0).mean(dim=0).unsqueeze(0)
    attn_ = attn_.reshape(
        shape=(attn_.shape[0], int(attn_.shape[1] ** .5), int(attn_.shape[1] ** .5)))
    attn_ = torch.einsum('chw->hwc', attn_).numpy()
    plt.imshow(attn_, cmap='viridis')
    plt.axis('off')
    plt.title("attention map", fontsize=5)

    # emb_h_org (row:2)
    emb_h_org = emb2patch_frame(emb_h_org.unsqueeze(0)).squeeze(0)
    for i in range(4):
        plt.subplot(5, 4, i + 5)
        emb_h_org_per_chans = emb_h_org[i: i + 1]
        emb_h_org_per_chans = torch.einsum('chw->hwc', emb_h_org_per_chans).numpy()
        plt.imshow(emb_h_org_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("emb_h_" + str(i + 1), fontsize=5)

    # emb_h_proj (row:3)
    emb_h_proj = emb2patch_frame(emb_h_proj.unsqueeze(0)).squeeze(0)
    for i in range(4):
        plt.subplot(5, 4, i + 9)
        emb_h_proj_per_chans = emb_h_proj[i: i + 1]
        emb_h_proj_per_chans = torch.einsum('chw->hwc', emb_h_proj_per_chans).numpy()
        plt.imshow(emb_h_proj_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("emb_h_proj_" + str(i + 1), fontsize=5)

    # clip_emb (row:4)
    clip_emb_org = emb2patch_frame(clip_emb_org.unsqueeze(0)).squeeze(0)
    for i in range(4):
        plt.subplot(5, 4, i + 13)
        clip_emb_org_per_chans = clip_emb_org[i: i + 1]
        clip_emb_org_per_chans = torch.einsum('chw->hwc', clip_emb_org_per_chans).numpy()
        plt.imshow(clip_emb_org_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("clip_emb_" + str(i + 1), fontsize=5)

    # clip_emb_proj (row:5)
    clip_emb_proj = emb2patch_frame(clip_emb_proj.unsqueeze(0)).squeeze(0)
    for i in range(4):
        plt.subplot(5, 4, i + 17)
        clip_emb_proj_per_chans = clip_emb_proj[i: i + 1]
        clip_emb_proj_per_chans = torch.einsum('chw->hwc', clip_emb_proj_per_chans).numpy()
        plt.imshow(clip_emb_proj_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("clip_emb_proj_" + str(i + 1), fontsize=5)

    plt.suptitle(image_name, fontsize=10)
    figure_name = ("epoch_0{}.png").format(epoch + 1) if (epoch + 1) < 10 else ("epoch_{}.png").format(epoch + 1)
    if args.pr_phase == "adj":
        phase_dir = args.adj_dir
    elif args.pr_phase == "_adj":
        phase_dir = args._adj_dir
    elif args.pr_phase == 'con':
        phase_dir = args.con_dir
    elif args.pr_phase == 'adj-n':
        phase_dir = args.adj_n_dir
    else:
        phase_dir = args.con_n_dir
    Path(os.path.join(args.output_dir, phase_dir, args.vis_train_dir)).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(args.output_dir, phase_dir, args.vis_train_dir, figure_name),
                bbox_inches='tight', pad_inches=0.1, dpi=100)
    plt.show()
    plt.close()


def vis_pr_rec_and_con(args, events_voxel_grid, emb_l1, emb_l2, emb_lh,
                       sub_frame, reconstruct_pred, mask, ids_restore,
                       emb_h_org, emb_h_proj,
                       clip_emb_org, clip_emb_proj, attn,
                       image_name, epoch):
    plt.figure()
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.02, hspace=0.6)

    events_voxel_grid, emb_l1, emb_l2, emb_lh, sub_frame, reconstruct_pred, mask, ids_restore, \
        emb_h_org, emb_h_proj, clip_emb_org, clip_emb_proj, attn = \
        events_voxel_grid.detach().cpu(), emb_l1.detach().cpu(), emb_l2.detach().cpu(), emb_lh.detach().cpu(), \
            sub_frame.detach().cpu(), reconstruct_pred.detach().cpu(), mask.detach().cpu(), ids_restore.detach().cpu(), \
            emb_h_org.detach().cpu(), emb_h_proj.detach().cpu(), \
            clip_emb_org.detach().cpu(), clip_emb_proj.detach().cpu(), attn.detach().cpu()

    # events frame (row:1 col:1)
    plt.subplot(9, 4, 1)
    events_frame = make_events_preview(events_voxel_grid)
    plt.imshow(events_frame, cmap='gray')
    plt.axis('off')
    plt.title("events frame", fontsize=5)

    # events frame norm (row:1 col:2)
    plt.subplot(9, 4, 2)
    events_frame_norm = make_events_preview_norm(events_voxel_grid)
    plt.imshow(events_frame_norm, cmap='gray')
    plt.axis('off')
    plt.title("events frame norm", fontsize=5)

    # attention map (row:1 col:3)
    plt.subplot(9, 4, 3)
    attn_ = attn.mean(dim=0).mean(dim=0).unsqueeze(0)
    attn_ = attn_.reshape(
        shape=(attn_.shape[0], int(attn_.shape[1] ** .5), int(attn_.shape[1] ** .5)))
    attn_ = torch.einsum('chw->hwc', attn_).numpy()
    plt.imshow(attn_, cmap='viridis')
    plt.axis('off')
    plt.title("attention map", fontsize=5)

    # emb_l1 (row:2)
    if args.backbone_type == "convvit":
        for i in range(4):
            plt.subplot(9, 4, i + 5)
            emb_l1_per_chans = emb_l1[i: i + 1]
            emb_l1_per_chans = torch.einsum('chw->hwc', emb_l1_per_chans).numpy()
            plt.imshow(emb_l1_per_chans, cmap='viridis')
            plt.axis('off')
            plt.title("emb_l1_" + str(i + 1), fontsize=5)
    else:
        emb_l1_norm = (emb_l1 - emb_l1.min()) / (emb_l1.max() - emb_l1.min())
        mask_emb = torch.zeros(ids_restore.shape[0] - emb_l1.shape[0], emb_l1.shape[1])
        emb_l1_ = torch.cat([emb_l1_norm, mask_emb], dim=0)
        emb_l1 = torch.gather(emb_l1_, dim=0,
                              index=ids_restore.unsqueeze(-1).repeat(1, 1, emb_l1.shape[1]).squeeze(0))  # unshuffle
        emb_l1 = emb2patch_frame(emb_l1.unsqueeze(0)).squeeze(0)
        for i in range(4):
            plt.subplot(9, 4, i + 5)
            emb_l1_per_chans = emb_l1[i: i + 1]
            emb_l1_per_chans = torch.einsum('chw->hwc', emb_l1_per_chans).numpy()
            plt.imshow(emb_l1_per_chans, cmap='viridis')
            plt.axis('off')
            plt.title("emb_l1_" + str(i + 1), fontsize=5)

    # emb_l2 (row:3)
    if args.backbone_type == "convvit":
        for i in range(4):
            plt.subplot(9, 4, i + 9)
            emb_l2_per_chans = emb_l2[i: i + 1]
            emb_l2_per_chans = torch.einsum('chw->hwc', emb_l2_per_chans).numpy()
            plt.imshow(emb_l2_per_chans, cmap='viridis')
            plt.axis('off')
            plt.title("emb_l2_" + str(i + 1), fontsize=5)
    else:
        emb_l2_norm = (emb_l2 - emb_l2.min()) / (emb_l2.max() - emb_l2.min())
        mask_emb = torch.zeros(ids_restore.shape[0] - emb_l2.shape[0], emb_l2.shape[1])
        emb_l2_ = torch.cat([emb_l2_norm, mask_emb], dim=0)
        emb_l2 = torch.gather(emb_l2_, dim=0,
                              index=ids_restore.unsqueeze(-1).repeat(1, 1, emb_l2.shape[1]).squeeze(0))  # unshuffle
        emb_l2 = emb2patch_frame(emb_l2.unsqueeze(0)).squeeze(0)
        for i in range(4):
            plt.subplot(9, 4, i + 9)
            emb_l2_per_chans = emb_l2[i: i + 1]
            emb_l2_per_chans = torch.einsum('chw->hwc', emb_l2_per_chans).numpy()
            plt.imshow(emb_l2_per_chans, cmap='viridis')
            plt.axis('off')
            plt.title("emb_l2_" + str(i + 1), fontsize=5)

    # emb_lh (row:4)
    emb_lh_norm = (emb_lh - emb_lh.min()) / (emb_lh.max() - emb_lh.min())
    mask_emb = torch.zeros(ids_restore.shape[0] - emb_lh.shape[0], emb_lh.shape[1])
    emb_lh_ = torch.cat([emb_lh_norm, mask_emb], dim=0)
    emb_lh = torch.gather(emb_lh_, dim=0,
                          index=ids_restore.unsqueeze(-1).repeat(1, 1, emb_lh.shape[1]).squeeze(0))  # unshuffle
    emb_lh = emb2patch_frame(emb_lh.unsqueeze(0)).squeeze(0)
    for i in range(4):
        plt.subplot(9, 4, i + 13)
        emb_lh_per_chans = emb_lh[i: i + 1]
        emb_lh_per_chans = torch.einsum('chw->hwc', emb_lh_per_chans).numpy()
        plt.imshow(emb_lh_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("emb_l_h_" + str(i + 1), fontsize=5)

    # sub frame (row:5 col:1)
    plt.subplot(9, 4, 17)
    sub_frame_norm = (sub_frame - sub_frame.min()) / (sub_frame.max() - sub_frame.min())
    sub_frame = torch.einsum('chw->hwc', sub_frame_norm)
    plt.imshow(sub_frame, cmap='gray')
    plt.axis('off')
    plt.title("sub frame", fontsize=5)

    # mask upsampling
    mask = mask
    mask = mask.unsqueeze(-1).repeat(1, args.patch_size ** 2 * args.frame_chans)  # (H*W, p*p*3)
    mask = emb2frame(args, mask.unsqueeze(0), args.frame_chans).squeeze(0)  # 1 is removing, 0 is keeping
    mask = torch.einsum('chw->hwc', mask)
    mask = mask.int()

    # masked sub frame (row:5 col:2)
    plt.subplot(9, 4, 18)
    masked_sub_frame = sub_frame * (1 - mask)
    plt.imshow(masked_sub_frame, cmap='gray')
    plt.axis('off')
    plt.title("masked sub frame", fontsize=5)

    # reconstruct sub frame (row:5 col:3)
    plt.subplot(9, 4, 19)
    reconstruct_pred_norm = (reconstruct_pred - reconstruct_pred.min()) / (
            reconstruct_pred.max() - reconstruct_pred.min())
    reconstruct_pred = emb2frame(args, reconstruct_pred_norm.unsqueeze(0), args.frame_chans)  # (1,1,224,224)
    reconstruct_frame = torch.einsum('bchw->bhwc', reconstruct_pred).squeeze(0)  # (224,224,1)
    plt.imshow(reconstruct_frame, cmap='gray')
    plt.axis('off')
    plt.title("reconstruct frame", fontsize=5)

    # reconstruct visible sub frame (row:5 col:4)
    plt.subplot(9, 4, 20)
    reconstruct_visible_sub_frame = sub_frame * (1 - mask) + reconstruct_frame * mask
    plt.imshow(reconstruct_visible_sub_frame, cmap='gray')
    plt.axis('off')
    plt.title("reconstruct visible sub frame", fontsize=5)

    # emb_h_org (row:6)
    emb_h_org = emb2patch_frame(emb_h_org.unsqueeze(0)).squeeze(0)
    for i in range(4):
        plt.subplot(9, 4, i + 21)
        emb_h_org_per_chans = emb_h_org[i: i + 1]
        emb_h_org_per_chans = torch.einsum('chw->hwc', emb_h_org_per_chans).numpy()
        plt.imshow(emb_h_org_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("emb_h_" + str(i + 1), fontsize=5)

    # emb_h_proj (row:7)
    emb_h_proj = emb2patch_frame(emb_h_proj.unsqueeze(0)).squeeze(0)
    for i in range(4):
        plt.subplot(9, 4, i + 25)
        emb_h_proj_per_chans = emb_h_proj[i: i + 1]
        emb_h_proj_per_chans = torch.einsum('chw->hwc', emb_h_proj_per_chans).numpy()
        plt.imshow(emb_h_proj_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("emb_h_proj_" + str(i + 1), fontsize=5)

    # clip_emb (row:8)
    clip_emb_org = emb2patch_frame(clip_emb_org.unsqueeze(0)).squeeze(0)
    for i in range(4):
        plt.subplot(9, 4, i + 29)
        clip_emb_org_per_chans = clip_emb_org[i: i + 1]
        clip_emb_org_per_chans = torch.einsum('chw->hwc', clip_emb_org_per_chans).numpy()
        plt.imshow(clip_emb_org_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("clip_emb_" + str(i + 1), fontsize=5)

    # clip_emb_proj (row:9)
    clip_emb_proj = emb2patch_frame(clip_emb_proj.unsqueeze(0)).squeeze(0)
    for i in range(4):
        plt.subplot(9, 4, i + 33)
        clip_emb_proj_per_chans = clip_emb_proj[i: i + 1]
        clip_emb_proj_per_chans = torch.einsum('chw->hwc', clip_emb_proj_per_chans).numpy()
        plt.imshow(clip_emb_proj_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("clip_emb_proj_" + str(i + 1), fontsize=5)

    plt.suptitle(image_name, fontsize=10)
    figure_name = ("epoch_0{}.png").format(epoch + 1) if (epoch + 1) < 10 else ("epoch_{}.png").format(epoch + 1)

    Path(os.path.join(args.output_dir, args.rec_and_con_dir, args.vis_train_dir)).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(args.output_dir, args.rec_and_con_dir, args.vis_train_dir, figure_name),
                bbox_inches='tight', pad_inches=0.1, dpi=100)
    plt.show()
    plt.close()


def vis_pr_ecdp(args, events_image_q, events_image_k, emb_event_q_org, emb_image_q_org, emb_event_q, emb_image_q,
                clip_emb_org, clip_emb_proj, mask_q, ids_restore_q, attn_q,
                mask_k, ids_restore_k, attn_k, image_name, epoch):
    plt.figure()
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.02, hspace=0.6)

    events_image_q, events_image_k, emb_event_q_org, emb_image_q_org, emb_event_q, emb_image_q, \
    clip_emb_org, clip_emb_proj, \
    mask_q, ids_restore_q, attn_q, \
    mask_k, ids_restore_k, attn_k = \
        events_image_q.detach().cpu(), events_image_k.detach().cpu(), emb_event_q_org.detach().cpu(), emb_image_q_org.detach().cpu(), \
        emb_event_q.detach().cpu(), emb_image_q.detach().cpu(), \
        clip_emb_org.detach().cpu(), clip_emb_proj.detach().cpu(), \
        mask_q.detach().cpu(), ids_restore_q.detach().cpu(), attn_q.detach().cpu(), \
        mask_k.detach().cpu(), ids_restore_k.detach().cpu(), attn_k.detach().cpu()

    # events image q (row:1 col:1)
    _events_image_q = copy.deepcopy(events_image_q)  # avoid make_events_preview() modify it
    plt.subplot(3, 4, 1)
    _events_image_q = make_events_preview(_events_image_q)
    plt.imshow(_events_image_q)
    plt.axis('off')
    plt.title("events image q", fontsize=5)

    # masked events image q (row:1 col:2)
    # mask upsampling q
    mask_q = mask_q.unsqueeze(-1).repeat(1, args.patch_size ** 2 * args.frame_chans)  # (H*W, p*p*3)
    mask_q = emb2frame(args, mask_q.unsqueeze(0), args.frame_chans).squeeze(0)  # 1 is removing, 0 is keeping
    mask_q = torch.einsum('chw->hwc', mask_q)
    mask_q = mask_q.int()

    plt.subplot(3, 4, 2)
    _events_image_q = _events_image_q * (1 - mask_q)
    plt.imshow(_events_image_q)
    plt.axis('off')
    plt.title("masked q", fontsize=5)

    # events image k (row:1 col:3)
    _events_image_k = copy.deepcopy(events_image_k)  # avoid make_events_preview() modify it
    plt.subplot(3, 4, 3)
    _events_image_k = make_events_preview(_events_image_k)
    plt.imshow(_events_image_k)
    plt.axis('off')
    plt.title("events image k", fontsize=5)

    # masked events image k (row:1 col:4)
    # mask upsampling k
    mask_k = mask_k.unsqueeze(-1).repeat(1, args.patch_size ** 2 * args.frame_chans)  # (H*W, p*p*3)
    mask_k = emb2frame(args, mask_k.unsqueeze(0), args.frame_chans).squeeze(0)  # 1 is removing, 0 is keeping
    mask_k = torch.einsum('chw->hwc', mask_k)
    mask_k = mask_k.int()

    plt.subplot(3, 4, 4)
    _events_image_k = _events_image_k * (1 - mask_k)
    plt.imshow(_events_image_k)
    plt.axis('off')
    plt.title("masked k", fontsize=5)

    # attention map q (row:2 col:1)
    plt.subplot(3, 4, 5)
    attn_q = attn_q[:, 0, 2:].reshape(attn_q.shape[0], -1)
    attn_q = attn_q.mean(dim=0)  # (12,196) -> (196)
    attn_q = attn_q.unsqueeze(-1)
    attn_q_norm = (attn_q - attn_q.min()) / (attn_q.max() - attn_q.min())
    mask_attn_q = torch.zeros(ids_restore_q.shape[0] - attn_q.shape[0], attn_q.shape[1])
    attn_q_norm = torch.cat([attn_q_norm, mask_attn_q], dim=0)
    attn_q_norm = torch.gather(attn_q_norm, dim=0,
                          index=ids_restore_q.unsqueeze(-1).repeat(1, 1, attn_q_norm.shape[1]).squeeze(0))  # unshuffle
    attn_q_norm = emb2patch_frame(attn_q_norm.unsqueeze(0)).squeeze(0).squeeze(0)
    plt.imshow(attn_q_norm, cmap='viridis')
    plt.axis('off')
    plt.title("attn q", fontsize=5)

    # attention map k (row:2 col:2)
    plt.subplot(3, 4, 6)
    attn_k = attn_k[:, 0, 2:].reshape(attn_k.shape[0], -1)
    attn_k = attn_k.mean(dim=0)  # (12,196) -> (196)
    attn_k = attn_k.unsqueeze(-1)
    attn_k_norm = (attn_k - attn_k.min()) / (attn_k.max() - attn_k.min())
    mask_attn_k = torch.zeros(ids_restore_k.shape[0] - attn_k.shape[0], attn_k.shape[1])
    attn_k_norm = torch.cat([attn_k_norm, mask_attn_k], dim=0)
    attn_k_norm = torch.gather(attn_k_norm, dim=0, index=ids_restore_k.unsqueeze(-1).repeat(1, 1, attn_k_norm.shape[1]).squeeze(0))  # unshuffle
    attn_k_norm = emb2patch_frame(attn_k_norm.unsqueeze(0)).squeeze(0).squeeze(0)
    plt.imshow(attn_k_norm, cmap='viridis')
    plt.axis('off')
    plt.title("attn k", fontsize=5)

    # clip_emb_org (row:2 col:3)
    plt.subplot(3, 4, 7)
    clip_emb_org = clip_emb_org.reshape(16, 32)
    plt.imshow(clip_emb_org, cmap='viridis')
    plt.axis('off')
    plt.title("clip_emb_org", fontsize=5)

    # clip_emb_proj (row:2 col:4)
    plt.subplot(3, 4, 8)
    clip_emb_proj = clip_emb_proj.reshape(16, 16)
    plt.imshow(clip_emb_proj, cmap='viridis')
    plt.axis('off')
    plt.title("clip_emb_proj", fontsize=5)

    # emb_image_q_org (row:3 col:1)
    plt.subplot(3, 4, 9)
    emb_image_q_org = emb_image_q_org.reshape(16, 24)
    plt.imshow(emb_image_q_org)
    plt.axis('off')
    plt.title("emb_image_q_org", fontsize=5)

    # emb_image_q (row:3 col:2)
    plt.subplot(3, 4, 10)
    emb_image_q = emb_image_q.reshape(16, 16)
    plt.imshow(emb_image_q)
    plt.axis('off')
    plt.title("emb_image_q", fontsize=5)

    # emb_event_q_org (row:3 col:4)
    plt.subplot(3, 4, 11)
    emb_event_q_org = emb_event_q_org.reshape(16, 24)
    plt.imshow(emb_event_q_org)
    plt.axis('off')
    plt.title("emb_event_q_org", fontsize=5)

    # emb_event_q (row:3 col:3)
    plt.subplot(3, 4, 12)
    emb_event_q = emb_event_q.reshape(16, 16)
    plt.imshow(emb_event_q)
    plt.axis('off')
    plt.title("emb_event_q", fontsize=5)

    plt.suptitle(image_name, fontsize=10)
    figure_name = ("epoch_0{}.png").format(epoch + 1) if (epoch + 1) < 10 else ("epoch_{}.png").format(epoch + 1)

    Path(os.path.join(args.output_dir, args.ecdp_ef_dir, args.vis_train_dir)).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(args.output_dir, args.ecdp_ef_dir, args.vis_train_dir, figure_name),
                bbox_inches='tight', pad_inches=0.1, dpi=100)
    plt.show()
    plt.close()

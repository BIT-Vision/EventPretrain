import copy
import os
from pathlib import Path
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')
# from matplotlib import pyplot as plt
import torch

from visualize.visualize_utils.make_events_preview import make_events_preview, make_events_preview_norm
from utils.reshape import emb2patch_frame


def vis_ft_cls(args, events_voxel_grid, emb_l1, emb_l2, emb_h, attn, image_name, epoch,
               is_train=True, dataset_name="origin"):
    plt.figure()
    plt.subplots_adjust(left=0.2, bottom=0.1, right=0.8, top=0.9, wspace=0.1, hspace=0.5)

    emb_l1, emb_l2, emb_h, events_voxel_grid, attn = \
        emb_l1.detach().cpu(), emb_l2.detach().cpu(), emb_h.detach().cpu(), \
            events_voxel_grid.detach().cpu(), attn.detach().cpu()

    # events frame (row:1 col:1)
    plt.subplot(4, 4, 1)
    events_frame = make_events_preview(events_voxel_grid)
    plt.imshow(events_frame, cmap='gray')
    plt.axis('off')
    plt.title("events image", fontsize=5)

    # events frame norm (row:1 col:2)
    plt.subplot(4, 4, 2)
    events_frame_norm = make_events_preview_norm(events_voxel_grid)
    plt.imshow(events_frame_norm, cmap='gray')
    plt.axis('off')
    plt.title("events image norm", fontsize=5)

    # attention map (row:1 col:3)
    plt.subplot(4, 4, 3)
    attn_ = attn.mean(dim=0).mean(dim=0)  # (12,196,196) -> (196)
    attn_ = attn_.unsqueeze(0)
    attn_ = attn_.reshape(
        shape=(attn_.shape[0], int(attn_.shape[1] ** .5), int(attn_.shape[1] ** .5)))
    attn_ = torch.einsum('chw->hwc', attn_).numpy()
    plt.imshow(attn_, cmap='viridis')
    plt.axis('off')
    plt.title("attention map", fontsize=5)

    # emb_l1 (row:2)
    if args.backbone_type != "convvit":
        emb_l1 = emb2patch_frame(emb_l1.unsqueeze(0)).squeeze(0)
    for i in range(4):
        plt.subplot(4, 4, i + 5)
        emb_l1_per_chans = emb_l1[i: i + 1]
        emb_l1_per_chans = torch.einsum('chw->hwc', emb_l1_per_chans).numpy()
        plt.imshow(emb_l1_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("emb_l1_" + str(i + 1), fontsize=5)

    # emb_l2 (row:3)
    if args.backbone_type != "convvit":
        emb_l2 = emb2patch_frame(emb_l2.unsqueeze(0)).squeeze(0)
    for i in range(4):
        plt.subplot(4, 4, i + 9)
        emb_l2_per_chans = emb_l2[i: i + 1]
        emb_l2_per_chans = torch.einsum('chw->hwc', emb_l2_per_chans).numpy()
        plt.imshow(emb_l2_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("emb_l2_" + str(i + 1), fontsize=5)

    # emb_h (row:4)
    emb_h = emb2patch_frame(emb_h.unsqueeze(0)).squeeze(0)
    for i in range(4):
        plt.subplot(4, 4, i + 13)
        emb_h_per_chans = emb_h[i: i + 1]
        emb_h_per_chans = torch.einsum('chw->hwc', emb_h_per_chans).numpy()
        plt.imshow(emb_h_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("emb_h_" + str(i+1), fontsize=5)

    plt.suptitle(image_name, fontsize=10)
    figure_name = ("epoch_0{}.png").format(epoch + 1) if (epoch + 1) < 10 else ("epoch_{}.png").format(epoch + 1)
    if is_train:
        vis_dir = args.vis_train_dir
    else:
        if args.dataset_type != "n-imagenet":
            vis_dir = args.vis_val_dir
        else:
            vis_dir = args.vis_val_dir + "-" + dataset_name
    Path(os.path.join(args.output_dir, vis_dir)).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(args.output_dir, vis_dir, figure_name),
                bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()
    plt.close()

def vis_ft_cls_ecdp(args, events_voxel_grid, attn, image_name, epoch,
                    is_train=True, dataset_name="origin"):
    plt.figure()
    plt.subplots_adjust(left=0.2, bottom=0.1, right=0.8, top=0.9, wspace=0.1, hspace=0.5)

    events_voxel_grid, attn = events_voxel_grid.detach().cpu(), attn.detach().cpu()

    _events_voxel_grid = copy.deepcopy(events_voxel_grid)  # avoid make_events_preview() modify it
    # events frame (row:1 col:1)
    plt.subplot(1, 4, 1)
    events_frame = make_events_preview(_events_voxel_grid)
    plt.imshow(events_frame, cmap='gray')
    plt.axis('off')
    plt.title("events image", fontsize=5)

    # events frame norm (row:1 col:2)
    plt.subplot(1, 4, 2)
    events_frame_norm = make_events_preview_norm(_events_voxel_grid)
    plt.imshow(events_frame_norm, cmap='gray')
    plt.axis('off')
    plt.title("events image norm", fontsize=5)

    # attention map 1 (row:1 col:3)
    plt.subplot(1, 4, 3)
    attn_ = attn[:, 0, 2:].reshape(attn.shape[0], -1)
    attn_ = attn_.mean(dim=0)  # (12,196) -> (196)
    attn_ = attn_.unsqueeze(0)
    attn_ = attn_.reshape(
        shape=(attn_.shape[0], int(attn_.shape[1] ** .5), int(attn_.shape[1] ** .5)))
    attn_ = torch.einsum('chw->hwc', attn_).numpy()
    plt.imshow(attn_, cmap='viridis')
    plt.axis('off')
    plt.title("attention map", fontsize=5)

    # attention map 2 (row:1 col:4)
    plt.subplot(1, 4, 4)
    attn_ = attn[:, 1, 2:].reshape(attn.shape[0], -1)
    attn_ = attn_.mean(dim=0)  # (12,196) -> (196)
    attn_ = attn_.unsqueeze(0)
    attn_ = attn_.reshape(
        shape=(attn_.shape[0], int(attn_.shape[1] ** .5), int(attn_.shape[1] ** .5)))
    attn_ = torch.einsum('chw->hwc', attn_).numpy()
    plt.imshow(attn_, cmap='viridis')
    plt.axis('off')
    plt.title("attention map", fontsize=5)

    plt.suptitle(image_name, fontsize=10)
    figure_name = ("epoch_0{}.png").format(epoch + 1) if (epoch + 1) < 10 else ("epoch_{}.png").format(epoch + 1)
    if is_train:
        vis_dir = args.vis_train_dir
    else:
        if args.dataset_type != "n-imagenet":
            vis_dir = args.vis_val_dir
        else:
            vis_dir = args.vis_val_dir + "-" + dataset_name
    Path(os.path.join(args.output_dir, vis_dir)).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(args.output_dir, vis_dir, figure_name),
                bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()
    plt.close()

def vis_ft_cls_mem(args, events_voxel_grid, attn, image_name, epoch, is_train=True):
    plt.figure()
    plt.subplots_adjust(left=0.2, bottom=0.1, right=0.8, top=0.9, wspace=0.1, hspace=0.5)

    events_voxel_grid, attn = events_voxel_grid.detach().cpu(), attn.detach().cpu()

    _events_voxel_grid = copy.deepcopy(events_voxel_grid)  # avoid mask_events_preview() modify it
    # events frame (row:1 col:1)
    plt.subplot(1, 4, 1)
    events_frame = make_events_preview(_events_voxel_grid)
    plt.imshow(events_frame, cmap='gray')
    plt.axis('off')
    plt.title("events image", fontsize=5)

    # events frame norm (row:1 col:2)
    plt.subplot(1, 4, 2)
    events_frame_norm = make_events_preview_norm(_events_voxel_grid)
    plt.imshow(events_frame_norm, cmap='gray')
    plt.axis('off')
    plt.title("events image norm", fontsize=5)

    # attention map (row:1 col:3)
    plt.subplot(1, 4, 3)
    attn_ = attn[:, 1:, 1:].mean(dim=0).mean(dim=0)  # (12,196,196) -> (196)
    attn_ = attn_.unsqueeze(0)
    attn_ = attn_.reshape(
        shape=(attn_.shape[0], int(attn_.shape[1] ** .5), int(attn_.shape[1] ** .5)))
    attn_ = torch.einsum('chw->hwc', attn_).numpy()
    plt.imshow(attn_, cmap='viridis')
    plt.axis('off')
    plt.title("attention map", fontsize=5)

    plt.suptitle(image_name, fontsize=10)
    figure_name = ("epoch_0{}.png").format(epoch + 1) if (epoch + 1) < 10 else ("epoch_{}.png").format(epoch + 1)
    if is_train:
        vis_dir = args.vis_train_dir
    else:
        vis_dir = args.vis_val_dir

    Path(os.path.join(args.output_dir, vis_dir)).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(args.output_dir, vis_dir, figure_name),
                bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()
    plt.close()

def vis_ft_cls_swin(args, events_voxel_grid, emb_l1, emb_l2, emb_l3, emb_l4, emb_h, attn, image_name, epoch,
                    is_train=True, dataset_name="origin"):
    plt.figure()
    plt.subplots_adjust(left=0.2, bottom=0.1, right=0.8, top=0.9, wspace=0.1, hspace=0.5)

    emb_l1, emb_l2, emb_l3, emb_l4, emb_h, events_voxel_grid, attn = \
        emb_l1.detach().cpu(), emb_l2.detach().cpu(), emb_l3.detach().cpu(), emb_l4.detach().cpu(), \
            emb_h.detach().cpu(), events_voxel_grid.detach().cpu(), attn.detach().cpu()

    # events frame (row:1 col:1)
    plt.subplot(6, 4, 1)
    events_frame = make_events_preview(events_voxel_grid)
    plt.imshow(events_frame, cmap='gray')
    plt.axis('off')
    plt.title("events image", fontsize=5)

    # events frame norm (row:1 col:2)
    plt.subplot(6, 4, 2)
    events_frame_norm = make_events_preview_norm(events_voxel_grid)
    plt.imshow(events_frame_norm, cmap='gray')
    plt.axis('off')
    plt.title("events image norm", fontsize=5)

    # attention map (row:1 col:3)
    plt.subplot(6, 4, 3)
    attn_ = attn.mean(dim=0).mean(dim=0)  # (12,196,196) -> (196)
    attn_ = attn_.unsqueeze(0)
    attn_ = attn_.reshape(
        shape=(attn_.shape[0], int(attn_.shape[1] ** .5), int(attn_.shape[1] ** .5)))
    attn_ = torch.einsum('chw->hwc', attn_).numpy()
    plt.imshow(attn_, cmap='viridis')
    plt.axis('off')
    plt.title("attention map", fontsize=5)

    # emb_l1 (row:2)
    emb_l1 = emb2patch_frame(emb_l1.unsqueeze(0)).squeeze(0)
    for i in range(4):
        plt.subplot(6, 4, i + 5)
        emb_l1_per_chans = emb_l1[i: i + 1]
        emb_l1_per_chans = torch.einsum('chw->hwc', emb_l1_per_chans).numpy()
        plt.imshow(emb_l1_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("emb_l1_" + str(i + 1), fontsize=5)

    # emb_l2 (row:3)
    emb_l2 = emb2patch_frame(emb_l2.unsqueeze(0)).squeeze(0)
    for i in range(4):
        plt.subplot(6, 4, i + 9)
        emb_l2_per_chans = emb_l2[i: i + 1]
        emb_l2_per_chans = torch.einsum('chw->hwc', emb_l2_per_chans).numpy()
        plt.imshow(emb_l2_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("emb_l2_" + str(i + 1), fontsize=5)

    # emb_l3 (row:4)
    emb_l3 = emb2patch_frame(emb_l3.unsqueeze(0)).squeeze(0)
    for i in range(4):
        plt.subplot(6, 4, i + 13)
        emb_l3_per_chans = emb_l3[i: i + 1]
        emb_l3_per_chans = torch.einsum('chw->hwc', emb_l3_per_chans).numpy()
        plt.imshow(emb_l3_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("emb_l3_" + str(i + 1), fontsize=5)

    # emb_l4 (row:5)
    emb_l4 = emb2patch_frame(emb_l4.unsqueeze(0)).squeeze(0)
    for i in range(4):
        plt.subplot(6, 4, i + 17)
        emb_l4_per_chans = emb_l4[i: i + 1]
        emb_l4_per_chans = torch.einsum('chw->hwc', emb_l4_per_chans).numpy()
        plt.imshow(emb_l4_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("emb_l4_" + str(i + 1), fontsize=5)

    # emb_h (row:6)
    emb_h = emb2patch_frame(emb_h.unsqueeze(0)).squeeze(0)
    for i in range(4):
        plt.subplot(6, 4, i + 21)
        emb_h_per_chans = emb_h[i: i + 1]
        emb_h_per_chans = torch.einsum('chw->hwc', emb_h_per_chans).numpy()
        plt.imshow(emb_h_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("emb_h_" + str(i+1), fontsize=5)

    plt.suptitle(image_name, fontsize=10)
    figure_name = ("epoch_0{}.png").format(epoch + 1) if (epoch + 1) < 10 else ("epoch_{}.png").format(epoch + 1)
    if is_train:
        vis_dir = args.vis_train_dir
    else:
        if args.dataset_type != "n-imagenet":
            vis_dir = args.vis_val_dir
        else:
            vis_dir = args.vis_val_dir + "-" + dataset_name
    Path(os.path.join(args.output_dir, vis_dir)).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(args.output_dir, vis_dir, figure_name),
                bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()
    plt.close()

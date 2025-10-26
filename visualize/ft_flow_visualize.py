import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')
# from matplotlib import pyplot as plt
import copy

import torch

from visualize.visualize_utils.make_events_preview import make_events_preview, make_events_preview_norm
from utils.reshape import emb2patch_frame


def make_colorwheel():
    """
        Generates a color wheel for optical flow visualization as presented in:
            Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
            URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

        Code follows the original C++ source code of Daniel Scharstein.
        Code follows the the Matlab source code of Deqing Sun.

        Returns:
            np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR  # 55
    colorwheel = np.zeros((ncols, 3))  # (55,3)
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(RY) / RY)
    col = col + RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255 * np.arange(YG) / YG)
    colorwheel[col:col+YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255 * np.arange(GC) / GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255 * np.arange(BM) / BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
        Applies the flow color wheel to (possibly clipped) flow components u and v.

        According to the C++ source code of Daniel Scharstein
        According to the Matlab source code of Deqing Sun

        Args:
            u (np.ndarray): Input horizontal flow of shape [H,W]
            v (np.ndarray): Input vertical flow of shape [H,W]
            convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

        Returns:
            np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)  # (260,346,3)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        idx = (rad <= 1)
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)

    return flow_image  # (260,346,3)

def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
        Expects a two dimensional flow image of shape.

        Args:
            flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
            clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
            convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

        Returns:
            np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:, :, 0]  # x (260,346)
    v = flow_uv[:, :, 1]  # y (260,346)
    rad = np.sqrt(np.square(u) + np.square(v))  # 平方后开方
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)

    return flow_uv_to_colors(u, v, convert_to_bgr)

def vis_ft_flow(args, events_voxel_grid, sparse_mask,
                flow_label, flow_label_valid, decode_predict, aux_predict,
                emb_l1, emb_l2, emb_h, attn, seq_name, epoch, is_train=True, dataset_name="indoor_flying1"):
    plt.figure()
    plt.subplots_adjust(left=0.2, bottom=0.1, right=0.8, top=0.9, wspace=0.1, hspace=0.5)

    events_voxel_grid, sparse_mask, flow_label, flow_label_valid, decode_predict, aux_predict, emb_l1, emb_l2, emb_h, attn = \
        events_voxel_grid.detach().cpu(), sparse_mask.detach().cpu(), flow_label.detach().cpu(), flow_label_valid.detach().cpu(), \
        decode_predict.detach().cpu(), aux_predict.detach().cpu(), \
        emb_l1.detach().cpu(), emb_l2.detach().cpu(), emb_h.detach().cpu(), attn.detach().cpu()

    # events frame (row:1 col:1)
    plt.subplot(6, 4, 1)
    events_frame = make_events_preview(events_voxel_grid)
    plt.imshow(events_frame, cmap='gray')
    plt.axis('off')
    plt.title("events frame", fontsize=5)

    # events frame norm (row:1 clo:2)
    plt.subplot(6, 4, 2)
    events_frame_norm = make_events_preview_norm(events_voxel_grid)
    plt.imshow(events_frame_norm, cmap='gray')
    plt.axis('off')
    plt.title("events frame norm", fontsize=5)

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

    # flow label (row:2 col:1)
    plt.subplot(6, 4, 5)
    flow_label[flow_label_valid.repeat(2, 1, 1) == 0] = 0
    flow_label = torch.einsum('chw->hwc', flow_label).numpy()
    flow_label_image = flow_to_image(flow_label)
    plt.imshow(flow_label_image)
    plt.axis('off')
    plt.title("flow label", fontsize=5)

    # decode_predict (row:2 col:2)
    plt.subplot(6, 4, 6)
    decode_predict[flow_label_valid.repeat(2, 1, 1) == 0] = 0
    decode_predict = torch.einsum('chw->hwc', decode_predict).numpy()
    decode_predict_image = flow_to_image(decode_predict)
    plt.imshow(decode_predict_image)
    plt.axis('off')
    plt.title("decode predict", fontsize=5)

    # aux_predict (row:2 col:3)
    plt.subplot(6, 4, 7)
    aux_predict[flow_label_valid.repeat(2, 1, 1) == 0] = 0
    aux_predict = torch.einsum('chw->hwc', aux_predict).numpy()
    aux_predict_image = flow_to_image(aux_predict)
    plt.imshow(aux_predict_image)
    plt.axis('off')
    plt.title("aux predict", fontsize=5)

    # flow label_mask (row:4 col:1)
    plt.subplot(6, 4, 9)
    flow_label_image[sparse_mask == 0] = 255
    plt.imshow(flow_label_image)
    plt.axis('off')
    plt.title("flow label mask", fontsize=5)

    # decode_predict_mask (row:4 col:2)
    plt.subplot(6, 4, 10)
    decode_predict_image[sparse_mask == 0] = 255
    plt.imshow(decode_predict_image)
    plt.axis('off')
    plt.title("decode predict mask", fontsize=5)

    # aux_predict (row:4 col:3)
    plt.subplot(6, 4, 11)
    aux_predict_image[sparse_mask == 0] = 255
    plt.imshow(aux_predict_image)
    plt.axis('off')
    plt.title("aux predict mask", fontsize=5)

    # emb_l1 (row:3)
    if args.backbone_type != "convvit":
        emb_l1 = emb2patch_frame(emb_l1.unsqueeze(0)).squeeze(0)
    for i in range(4):
        plt.subplot(6, 4, i + 13)
        emb_l1_per_chans = emb_l1[i: i + 1]
        emb_l1_per_chans = torch.einsum('chw->hwc', emb_l1_per_chans).numpy()
        plt.imshow(emb_l1_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("emb_l1_" + str(i + 1), fontsize=5)

    # emb_l2 (row:4)
    if args.backbone_type != "convvit":
        emb_l2 = emb2patch_frame(emb_l2.unsqueeze(0)).squeeze(0)
    for i in range(4):
        plt.subplot(6, 4, i + 17)
        emb_l2_per_chans = emb_l2[i: i + 1]
        emb_l2_per_chans = torch.einsum('chw->hwc', emb_l2_per_chans).numpy()
        plt.imshow(emb_l2_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("emb_l2_" + str(i + 1), fontsize=5)

    # emb_h (row:5)
    emb_h = emb2patch_frame(emb_h.unsqueeze(0)).squeeze(0)
    for i in range(4):
        plt.subplot(6, 4, i + 21)
        emb_h_per_chans = emb_h[i: i + 1]
        emb_h_per_chans = torch.einsum('chw->hwc', emb_h_per_chans).numpy()
        plt.imshow(emb_h_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("emb_h_" + str(i+1), fontsize=5)

    plt.suptitle(seq_name, fontsize=10)
    figure_name = ("epoch_0{}.png").format(epoch + 1) if (epoch + 1) < 10 else ("epoch_{}.png").format(epoch + 1)
    if is_train:
        vis_dir = args.vis_train_dir
    else:
        vis_dir = args.vis_val_dir + "-" + dataset_name
    Path(os.path.join(args.output_dir, vis_dir)).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(args.output_dir, vis_dir, figure_name),
                bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()
    plt.close()

def vis_ft_flow_mem(args, events_voxel_grid, sparse_mask,
                flow_label, flow_label_valid, decode_predict, aux_predict,
                emb, attn, seq_name, epoch, is_train=True, dataset_name="indoor_flying1"):
    plt.figure()
    plt.subplots_adjust(left=0.2, bottom=0.1, right=0.8, top=0.9, wspace=0.1, hspace=0.5)

    events_voxel_grid, sparse_mask, flow_label, flow_label_valid, decode_predict, aux_predict, emb, attn = \
        events_voxel_grid.detach().cpu(), sparse_mask.detach().cpu(), flow_label.detach().cpu(), flow_label_valid.detach().cpu(), \
        decode_predict.detach().cpu(), aux_predict.detach().cpu(), \
        emb.detach().cpu(), attn.detach().cpu()

    _events_voxel_grid = copy.deepcopy(events_voxel_grid)  # avoid mask_events_preview() modify it
    # events frame (row:1 col:1)
    plt.subplot(4, 4, 1)
    events_frame = make_events_preview(_events_voxel_grid)
    plt.imshow(events_frame, cmap='gray')
    plt.axis('off')
    plt.title("events image", fontsize=5)

    # events frame norm (row:1 col:2)
    plt.subplot(4, 4, 2)
    events_frame_norm = make_events_preview_norm(_events_voxel_grid)
    plt.imshow(events_frame_norm, cmap='gray')
    plt.axis('off')
    plt.title("events image norm", fontsize=5)

    # attention map (row:1 col:3)
    plt.subplot(4, 4, 3)
    attn_ = attn[:, 1:, 1:].mean(dim=0).mean(dim=0)  # (12,196,196) -> (196)
    attn_ = attn_.unsqueeze(0)
    attn_ = attn_.reshape(
        shape=(attn_.shape[0], int(attn_.shape[1] ** .5), int(attn_.shape[1] ** .5)))
    attn_ = torch.einsum('chw->hwc', attn_).numpy()
    plt.imshow(attn_, cmap='viridis')
    plt.axis('off')
    plt.title("attention map", fontsize=5)

    # flow label (row:2 col:1)
    plt.subplot(4, 4, 5)
    flow_label[flow_label_valid.repeat(2, 1, 1) == 0] = 0
    flow_label = torch.einsum('chw->hwc', flow_label).numpy()
    flow_label_image = flow_to_image(flow_label)
    plt.imshow(flow_label_image)
    plt.axis('off')
    plt.title("flow label", fontsize=5)

    # decode_predict (row:2 col:2)
    plt.subplot(4, 4, 6)
    decode_predict[flow_label_valid.repeat(2, 1, 1) == 0] = 0
    decode_predict = torch.einsum('chw->hwc', decode_predict).numpy()
    decode_predict_image = flow_to_image(decode_predict)
    plt.imshow(decode_predict_image)
    plt.axis('off')
    plt.title("decode predict", fontsize=5)

    # aux_predict (row:2 col:3)
    plt.subplot(4, 4, 7)
    aux_predict[flow_label_valid.repeat(2, 1, 1) == 0] = 0
    aux_predict = torch.einsum('chw->hwc', aux_predict).numpy()
    aux_predict_image = flow_to_image(aux_predict)
    plt.imshow(aux_predict_image)
    plt.axis('off')
    plt.title("aux predict", fontsize=5)

    # flow label_mask (row:3 col:1)
    plt.subplot(4, 4, 9)
    flow_label_image[sparse_mask == 0] = 255
    plt.imshow(flow_label_image)
    plt.axis('off')
    plt.title("flow label mask", fontsize=5)

    # decode_predict_mask (row:3 col:2)
    plt.subplot(4, 4, 10)
    decode_predict_image[sparse_mask == 0] = 255
    plt.imshow(decode_predict_image)
    plt.axis('off')
    plt.title("decode predict mask", fontsize=5)

    # aux_predict (row:3 col:3)
    plt.subplot(4, 4, 11)
    aux_predict_image[sparse_mask == 0] = 255
    plt.imshow(aux_predict_image)
    plt.axis('off')
    plt.title("aux predict mask", fontsize=5)

    # emb (row:4)
    for i in range(4):
        plt.subplot(4, 4, i + 13)
        emb_h_per_chans = emb[i: i + 1]
        emb_h_per_chans = torch.einsum('chw->hwc', emb_h_per_chans).numpy()
        plt.imshow(emb_h_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("emb_h_" + str(i+1), fontsize=5)

    plt.suptitle(seq_name, fontsize=10)
    figure_name = ("epoch_0{}.png").format(epoch + 1) if (epoch + 1) < 10 else ("epoch_{}.png").format(epoch + 1)
    if is_train:
        vis_dir = args.vis_train_dir
    else:
        vis_dir = args.vis_val_dir + "-" + dataset_name
    Path(os.path.join(args.output_dir, vis_dir)).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(args.output_dir, vis_dir, figure_name),
                bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()
    plt.close()

def vis_ft_flow_ecdp(args, events_voxel_grid, sparse_mask,
                flow_label, flow_label_valid, decode_predict, aux_predict,
                emb, attn, seq_name, epoch, is_train=True, dataset_name="indoor_flying1"):
    plt.figure()
    plt.subplots_adjust(left=0.2, bottom=0.1, right=0.8, top=0.9, wspace=0.1, hspace=0.5)

    events_voxel_grid, sparse_mask, flow_label, flow_label_valid, decode_predict, aux_predict, emb, attn = \
        events_voxel_grid.detach().cpu(), sparse_mask.detach().cpu(), flow_label.detach().cpu(), flow_label_valid.detach().cpu(), \
        decode_predict.detach().cpu(), aux_predict.detach().cpu(), \
        emb.detach().cpu(), attn.detach().cpu()

    # events frame (row:1 col:1)
    _events_voxel_grid = copy.deepcopy(events_voxel_grid)  # avoid make_events_preview() modify it
    # events frame (row:1 col:1)
    plt.subplot(4, 4, 1)
    events_frame = make_events_preview(_events_voxel_grid)
    plt.imshow(events_frame, cmap='gray')
    plt.axis('off')
    plt.title("events image", fontsize=5)

    # events frame norm (row:1 clo:2)
    plt.subplot(4, 4, 2)
    events_frame_norm = make_events_preview_norm(_events_voxel_grid)
    plt.imshow(events_frame_norm, cmap='gray')
    plt.axis('off')
    plt.title("events frame norm", fontsize=5)

    # attention map 1 (row:1 col:3)
    plt.subplot(4, 4, 3)
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
    plt.subplot(4, 4, 4)
    attn_ = attn[:, 1, 2:].reshape(attn.shape[0], -1)
    attn_ = attn_.mean(dim=0)  # (12,196) -> (196)
    attn_ = attn_.unsqueeze(0)
    attn_ = attn_.reshape(
        shape=(attn_.shape[0], int(attn_.shape[1] ** .5), int(attn_.shape[1] ** .5)))
    attn_ = torch.einsum('chw->hwc', attn_).numpy()
    plt.imshow(attn_, cmap='viridis')
    plt.axis('off')
    plt.title("attention map", fontsize=5)

    # flow label (row:2 col:1)
    plt.subplot(4, 4, 5)
    flow_label[flow_label_valid.repeat(2, 1, 1) == 0] = 0
    flow_label = torch.einsum('chw->hwc', flow_label).numpy()
    flow_label_image = flow_to_image(flow_label)
    plt.imshow(flow_label_image)
    plt.axis('off')
    plt.title("flow label", fontsize=5)

    # decode_predict (row:2 col:2)
    plt.subplot(4, 4, 6)
    decode_predict[flow_label_valid.repeat(2, 1, 1) == 0] = 0
    decode_predict = torch.einsum('chw->hwc', decode_predict).numpy()
    decode_predict_image = flow_to_image(decode_predict)
    plt.imshow(decode_predict_image)
    plt.axis('off')
    plt.title("decode predict", fontsize=5)

    # aux_predict (row:2 col:3)
    plt.subplot(4, 4, 7)
    aux_predict[flow_label_valid.repeat(2, 1, 1) == 0] = 0
    aux_predict = torch.einsum('chw->hwc', aux_predict).numpy()
    aux_predict_image = flow_to_image(aux_predict)
    plt.imshow(aux_predict_image)
    plt.axis('off')
    plt.title("aux predict", fontsize=5)

    # flow label_mask (row:3 col:1)
    plt.subplot(4, 4, 9)
    flow_label_image[sparse_mask == 0] = 255
    plt.imshow(flow_label_image)
    plt.axis('off')
    plt.title("flow label mask", fontsize=5)

    # decode_predict_mask (row:3 col:2)
    plt.subplot(4, 4, 10)
    decode_predict_image[sparse_mask == 0] = 255
    plt.imshow(decode_predict_image)
    plt.axis('off')
    plt.title("decode predict mask", fontsize=5)

    # aux_predict (row:3 col:3)
    plt.subplot(4, 4, 11)
    aux_predict_image[sparse_mask == 0] = 255
    plt.imshow(aux_predict_image)
    plt.axis('off')
    plt.title("aux predict mask", fontsize=5)

    # emb (row:4)
    for i in range(4):
        plt.subplot(4, 4, i + 13)
        emb_h_per_chans = emb[i: i + 1]
        emb_h_per_chans = torch.einsum('chw->hwc', emb_h_per_chans).numpy()
        plt.imshow(emb_h_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("emb_h_" + str(i+1), fontsize=5)

    plt.suptitle(seq_name, fontsize=10)
    figure_name = ("epoch_0{}.png").format(epoch + 1) if (epoch + 1) < 10 else ("epoch_{}.png").format(epoch + 1)
    if is_train:
        vis_dir = args.vis_train_dir
    else:
        vis_dir = args.vis_val_dir + "-" + dataset_name
    Path(os.path.join(args.output_dir, vis_dir)).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(args.output_dir, vis_dir, figure_name),
                bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()
    plt.close()

def vis_ft_flow_swin(args, events_voxel_grid, sparse_mask,
                     flow_label, flow_label_valid, decode_predict, aux_predict,
                     emb_l1, emb_l2, emb_l3, emb_l4, emb_h, attn,
                     seq_name, epoch, is_train=True, dataset_name="indoor_flying1"):
    plt.figure()
    plt.subplots_adjust(left=0.2, bottom=0.1, right=0.8, top=0.9, wspace=0.1, hspace=0.5)

    events_voxel_grid, sparse_mask, flow_label, flow_label_valid, decode_predict, aux_predict, \
    emb_l1, emb_l2, emb_l3, emb_l4, emb_h, attn = \
        events_voxel_grid.detach().cpu(), sparse_mask.detach().cpu(), flow_label.detach().cpu(), flow_label_valid.detach().cpu(), \
        decode_predict.detach().cpu(), aux_predict.detach().cpu(), \
        emb_l1.detach().cpu(), emb_l2.detach().cpu(), emb_l3.detach().cpu(), emb_l4.detach().cpu(), \
        emb_h.detach().cpu(), attn.detach().cpu()

    # events frame (row:1 col:1)
    plt.subplot(8, 4, 1)
    events_frame = make_events_preview(events_voxel_grid)
    plt.imshow(events_frame, cmap='gray')
    plt.axis('off')
    plt.title("events frame", fontsize=5)

    # events frame norm (row:1 clo:2)
    plt.subplot(8, 4, 2)
    events_frame_norm = make_events_preview_norm(events_voxel_grid)
    plt.imshow(events_frame_norm, cmap='gray')
    plt.axis('off')
    plt.title("events frame norm", fontsize=5)

    # attention map (row:1 col:3)
    plt.subplot(8, 4, 3)
    attn_ = attn.mean(dim=0).mean(dim=0)  # (12,196,196) -> (196)
    attn_ = attn_.unsqueeze(0)
    attn_ = attn_.reshape(
        shape=(attn_.shape[0], int(attn_.shape[1] ** .5), int(attn_.shape[1] ** .5)))
    attn_ = torch.einsum('chw->hwc', attn_).numpy()
    plt.imshow(attn_, cmap='viridis')
    plt.axis('off')
    plt.title("attention map", fontsize=5)

    # flow label (row:2 col:1)
    plt.subplot(8, 4, 5)
    flow_label[flow_label_valid.repeat(2, 1, 1) == 0] = 0
    flow_label = torch.einsum('chw->hwc', flow_label).numpy()
    flow_label_image = flow_to_image(flow_label)
    plt.imshow(flow_label_image)
    plt.axis('off')
    plt.title("flow label", fontsize=5)

    # decode_predict (row:2 col:2)
    plt.subplot(8, 4, 6)
    decode_predict[flow_label_valid.repeat(2, 1, 1) == 0] = 0
    decode_predict = torch.einsum('chw->hwc', decode_predict).numpy()
    decode_predict_image = flow_to_image(decode_predict)
    plt.imshow(decode_predict_image)
    plt.axis('off')
    plt.title("decode predict", fontsize=5)

    # aux_predict (row:2 col:3)
    plt.subplot(8, 4, 7)
    aux_predict[flow_label_valid.repeat(2, 1, 1) == 0] = 0
    aux_predict = torch.einsum('chw->hwc', aux_predict).numpy()
    aux_predict_image = flow_to_image(aux_predict)
    plt.imshow(aux_predict_image)
    plt.axis('off')
    plt.title("aux predict", fontsize=5)

    # flow label_mask (row:3 col:1)
    plt.subplot(8, 4, 9)
    flow_label_image[sparse_mask == 0] = 255
    plt.imshow(flow_label_image)
    plt.axis('off')
    plt.title("flow label mask", fontsize=5)

    # decode_predict_mask (row:3 col:2)
    plt.subplot(8, 4, 10)
    decode_predict_image[sparse_mask == 0] = 255
    plt.imshow(decode_predict_image)
    plt.axis('off')
    plt.title("decode predict mask", fontsize=5)

    # aux_predict (row:3 col:3)
    plt.subplot(8, 4, 11)
    aux_predict_image[sparse_mask == 0] = 255
    plt.imshow(aux_predict_image)
    plt.axis('off')
    plt.title("aux predict mask", fontsize=5)

    # emb_l1 (row:4)
    emb_l1 = emb2patch_frame(emb_l1.unsqueeze(0)).squeeze(0)
    for i in range(4):
        plt.subplot(8, 4, i + 13)
        emb_l1_per_chans = emb_l1[i: i + 1]
        emb_l1_per_chans = torch.einsum('chw->hwc', emb_l1_per_chans).numpy()
        plt.imshow(emb_l1_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("emb_l1_" + str(i + 1), fontsize=5)

    # emb_l2 (row:5)
    emb_l2 = emb2patch_frame(emb_l2.unsqueeze(0)).squeeze(0)
    for i in range(4):
        plt.subplot(8, 4, i + 17)
        emb_l2_per_chans = emb_l2[i: i + 1]
        emb_l2_per_chans = torch.einsum('chw->hwc', emb_l2_per_chans).numpy()
        plt.imshow(emb_l2_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("emb_l2_" + str(i + 1), fontsize=5)

    # emb_l3 (row:6)
    emb_l3 = emb2patch_frame(emb_l3.unsqueeze(0)).squeeze(0)
    for i in range(4):
        plt.subplot(8, 4, i + 21)
        emb_l3_per_chans = emb_l3[i: i + 1]
        emb_l3_per_chans = torch.einsum('chw->hwc', emb_l3_per_chans).numpy()
        plt.imshow(emb_l3_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("emb_l3_" + str(i + 1), fontsize=5)

    # emb_l4 (row:7)
    emb_l4 = emb2patch_frame(emb_l4.unsqueeze(0)).squeeze(0)
    for i in range(4):
        plt.subplot(8, 4, i + 25)
        emb_l4_per_chans = emb_l4[i: i + 1]
        emb_l4_per_chans = torch.einsum('chw->hwc', emb_l4_per_chans).numpy()
        plt.imshow(emb_l4_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("emb_l4_" + str(i + 1), fontsize=5)

    # emb_h (row:8)
    emb_h = emb2patch_frame(emb_h.unsqueeze(0)).squeeze(0)
    for i in range(4):
        plt.subplot(8, 4, i + 29)
        emb_h_per_chans = emb_h[i: i + 1]
        emb_h_per_chans = torch.einsum('chw->hwc', emb_h_per_chans).numpy()
        plt.imshow(emb_h_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("emb_h_" + str(i+1), fontsize=5)

    plt.suptitle(seq_name, fontsize=10)
    figure_name = ("epoch_0{}.png").format(epoch + 1) if (epoch + 1) < 10 else ("epoch_{}.png").format(epoch + 1)
    if is_train:
        vis_dir = args.vis_train_dir
    else:
        vis_dir = args.vis_val_dir + "-" + dataset_name
    Path(os.path.join(args.output_dir, vis_dir)).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(args.output_dir, vis_dir, figure_name),
                bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()
    plt.close()

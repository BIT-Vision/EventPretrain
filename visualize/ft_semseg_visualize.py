import os
from pathlib import Path
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')
# from matplotlib import pyplot as plt
import copy

import torch

from visualize.visualize_utils.make_events_preview import make_events_preview, make_events_preview_norm
from utils.reshape import emb2patch_frame


dsec_color_class_0 = torch.tensor([0, 0, 0])
dsec_color_class_1 = torch.tensor([70, 70, 70])
dsec_color_class_2 = torch.tensor([190, 153, 153])
dsec_color_class_3 = torch.tensor([220, 20, 60])
dsec_color_class_4 = torch.tensor([153, 153, 153])
dsec_color_class_5 = torch.tensor([128, 64, 128])
dsec_color_class_6 = torch.tensor([244, 35, 232])
dsec_color_class_7 = torch.tensor([107, 142, 35])
dsec_color_class_8 = torch.tensor([0, 0, 142])
dsec_color_class_9 = torch.tensor([102, 102, 156])
dsec_color_class_10 = torch.tensor([220, 220, 0])

ddd17_color_class_0 = torch.tensor([128, 64, 128])
ddd17_color_class_1 = torch.tensor([70, 70, 70])
ddd17_color_class_2 = torch.tensor([220, 220, 0])
ddd17_color_class_3 = torch.tensor([107, 142, 35])
ddd17_color_class_4 = torch.tensor([220, 20, 60])
ddd17_color_class_5 = torch.tensor([0, 0, 142])

def draw_semseg_color_map(args, predict):
    seg_color_map = torch.zeros(predict.shape[1], predict.shape[2], 3, dtype=torch.int32)

    if args.dataset_type == "dsec":
        seg_color_map[predict.squeeze(0) == 0] = dsec_color_class_0.reshape(1, 3).repeat(
            (seg_color_map[predict.squeeze(0) == 0].shape[0], 1)).int()
        seg_color_map[predict.squeeze(0) == 1] = dsec_color_class_1.reshape(1, 3).repeat(
            (seg_color_map[predict.squeeze(0) == 1].shape[0], 1)).int()
        seg_color_map[predict.squeeze(0) == 2] = dsec_color_class_2.reshape(1, 3).repeat(
            (seg_color_map[predict.squeeze(0) == 2].shape[0], 1)).int()
        seg_color_map[predict.squeeze(0) == 3] = dsec_color_class_3.reshape(1, 3).repeat(
            (seg_color_map[predict.squeeze(0) == 3].shape[0], 1)).int()
        seg_color_map[predict.squeeze(0) == 4] = dsec_color_class_4.reshape(1, 3).repeat(
            (seg_color_map[predict.squeeze(0) == 4].shape[0], 1)).int()
        seg_color_map[predict.squeeze(0) == 5] = dsec_color_class_5.reshape(1, 3).repeat(
            (seg_color_map[predict.squeeze(0) == 5].shape[0], 1)).int()
        seg_color_map[predict.squeeze(0) == 6] = dsec_color_class_6.reshape(1, 3).repeat(
            (seg_color_map[predict.squeeze(0) == 6].shape[0], 1)).int()
        seg_color_map[predict.squeeze(0) == 7] = dsec_color_class_7.reshape(1, 3).repeat(
            (seg_color_map[predict.squeeze(0) == 7].shape[0], 1)).int()
        seg_color_map[predict.squeeze(0) == 8] = dsec_color_class_8.reshape(1, 3).repeat(
            (seg_color_map[predict.squeeze(0) == 8].shape[0], 1)).int()
        seg_color_map[predict.squeeze(0) == 9] = dsec_color_class_9.reshape(1, 3).repeat(
            (seg_color_map[predict.squeeze(0) == 9].shape[0], 1)).int()
        seg_color_map[predict.squeeze(0) == 10] = dsec_color_class_10.reshape(1, 3).repeat(
            (seg_color_map[predict.squeeze(0) == 10].shape[0], 1)).int()
    elif args.dataset_type == "ddd17":
        seg_color_map[predict.squeeze(0) == 0] = ddd17_color_class_0.reshape(1, 3).repeat(
            (seg_color_map[predict.squeeze(0) == 0].shape[0], 1)).int()
        seg_color_map[predict.squeeze(0) == 1] = ddd17_color_class_1.reshape(1, 3).repeat(
            (seg_color_map[predict.squeeze(0) == 1].shape[0], 1)).int()
        seg_color_map[predict.squeeze(0) == 2] = ddd17_color_class_2.reshape(1, 3).repeat(
            (seg_color_map[predict.squeeze(0) == 2].shape[0], 1)).int()
        seg_color_map[predict.squeeze(0) == 3] = ddd17_color_class_3.reshape(1, 3).repeat(
            (seg_color_map[predict.squeeze(0) == 3].shape[0], 1)).int()
        seg_color_map[predict.squeeze(0) == 4] = ddd17_color_class_4.reshape(1, 3).repeat(
            (seg_color_map[predict.squeeze(0) == 4].shape[0], 1)).int()
        seg_color_map[predict.squeeze(0) == 5] = ddd17_color_class_5.reshape(1, 3).repeat(
            (seg_color_map[predict.squeeze(0) == 5].shape[0], 1)).int()
    else:
        raise ValueError

    return seg_color_map

def vis_ft_semseg(args, events_voxel_grid, semseg_label, decode_predict, aux_predict,
               emb_l1, emb_l2, emb_h, attn, seq_name, epoch, is_train=True):
    plt.figure()
    plt.subplots_adjust(left=0.2, bottom=0.1, right=0.8, top=0.9, wspace=0.1, hspace=0.5)

    events_voxel_grid, semseg_label, decode_predict, aux_predict, emb_l1, emb_l2, emb_h, attn = \
        events_voxel_grid.detach().cpu(), semseg_label.detach().cpu(), \
        decode_predict.detach().cpu(), aux_predict.detach().cpu(), \
        emb_l1.detach().cpu(), emb_l2.detach().cpu(), emb_h.detach().cpu(), attn.detach().cpu()

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

    # attention map (row:1 col:3)
    plt.subplot(5, 4, 3)
    attn_ = attn.mean(dim=0).mean(dim=0)  # (12,196,196) -> (196)
    attn_ = attn_.unsqueeze(0)
    attn_ = attn_.reshape(
        shape=(attn_.shape[0], int(attn_.shape[1] ** .5), int(attn_.shape[1] ** .5)))
    attn_ = torch.einsum('chw->hwc', attn_).numpy()
    plt.imshow(attn_, cmap='viridis')
    plt.axis('off')
    plt.title("attention map", fontsize=5)

    # semseg label (row:2 col:1)
    plt.subplot(5, 4, 5)
    label_semseg_color_map = draw_semseg_color_map(args, semseg_label)
    plt.imshow(label_semseg_color_map)
    plt.axis('off')
    plt.title("semseg label", fontsize=5)

    # decode_predict (row:2 col:2)
    plt.subplot(5, 4, 6)
    decode_predict = torch.argmax(decode_predict.unsqueeze(0), dim=1)
    decode_semseg_color_map = draw_semseg_color_map(args, decode_predict)
    plt.imshow(decode_semseg_color_map)
    plt.axis('off')
    plt.title("decode predict", fontsize=5)

    # aux_predict (row:2 col:3)
    plt.subplot(5, 4, 7)
    aux_predict = torch.argmax(aux_predict.unsqueeze(0), dim=1)
    aux_semseg_color_map = draw_semseg_color_map(args, aux_predict)
    plt.imshow(aux_semseg_color_map)
    plt.axis('off')
    plt.title("aux predict", fontsize=5)

    # emb_l1 (row:3)
    if args.backbone_type != "convvit":
        emb_l1 = emb2patch_frame(emb_l1.unsqueeze(0)).squeeze(0)
    for i in range(4):
        plt.subplot(5, 4, i + 9)
        emb_l1_per_chans = emb_l1[i: i + 1]
        emb_l1_per_chans = torch.einsum('chw->hwc', emb_l1_per_chans).numpy()
        plt.imshow(emb_l1_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("emb_l1_" + str(i + 1), fontsize=5)

    # emb_l2 (row:4)
    if args.backbone_type != "convvit":
        emb_l2 = emb2patch_frame(emb_l2.unsqueeze(0)).squeeze(0)
    for i in range(4):
        plt.subplot(5, 4, i + 13)
        emb_l2_per_chans = emb_l2[i: i + 1]
        emb_l2_per_chans = torch.einsum('chw->hwc', emb_l2_per_chans).numpy()
        plt.imshow(emb_l2_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("emb_l2_" + str(i + 1), fontsize=5)

    # emb_h (row:5)
    emb_h = emb2patch_frame(emb_h.unsqueeze(0)).squeeze(0)
    for i in range(4):
        plt.subplot(5, 4, i + 17)
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
        vis_dir = args.vis_val_dir
    Path(os.path.join(args.output_dir, vis_dir)).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(args.output_dir, vis_dir, figure_name),
                bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()
    plt.close()

def vis_ft_semseg_ecdp(args, events_voxel_grid, semseg_label, decode_predict, aux_predict,
                      emb, attn, seq_name, epoch, is_train=True):
    plt.figure()
    plt.subplots_adjust(left=0.2, bottom=0.1, right=0.8, top=0.9, wspace=0.1, hspace=0.5)

    events_voxel_grid, semseg_label, decode_predict, aux_predict, emb, attn = \
        events_voxel_grid.detach().cpu(), semseg_label.detach().cpu(), \
            decode_predict.detach().cpu(), aux_predict.detach().cpu(), \
            emb.detach().cpu(), attn.detach().cpu()

    _events_voxel_grid = copy.deepcopy(events_voxel_grid)  # avoid make_events_preview() modify it
    # events frame (row:1 col:1)
    plt.subplot(3, 4, 1)
    events_frame = make_events_preview(_events_voxel_grid)
    plt.imshow(events_frame, cmap='gray')
    plt.axis('off')
    plt.title("events image", fontsize=5)

    # events frame norm (row:1 col:2)
    plt.subplot(3, 4, 2)
    events_frame_norm = make_events_preview_norm(_events_voxel_grid)
    plt.imshow(events_frame_norm, cmap='gray')
    plt.axis('off')
    plt.title("events image norm", fontsize=5)

    # attention map 1 (row:1 col:3)
    plt.subplot(3, 4, 3)
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
    plt.subplot(3, 4, 4)
    attn_ = attn[:, 1, 2:].reshape(attn.shape[0], -1)
    attn_ = attn_.mean(dim=0)  # (12,196) -> (196)
    attn_ = attn_.unsqueeze(0)
    attn_ = attn_.reshape(
        shape=(attn_.shape[0], int(attn_.shape[1] ** .5), int(attn_.shape[1] ** .5)))
    attn_ = torch.einsum('chw->hwc', attn_).numpy()
    plt.imshow(attn_, cmap='viridis')
    plt.axis('off')
    plt.title("attention map", fontsize=5)

    # semseg label (row:2 col:1)
    plt.subplot(3, 4, 5)
    label_semseg_color_map = draw_semseg_color_map(args, semseg_label)
    plt.imshow(label_semseg_color_map)
    plt.axis('off')
    plt.title("semseg label", fontsize=5)

    # decode_predict (row:2 col:2)
    plt.subplot(3, 4, 6)
    decode_predict = torch.argmax(decode_predict.unsqueeze(0), dim=1)
    decode_semseg_color_map = draw_semseg_color_map(args, decode_predict)
    plt.imshow(decode_semseg_color_map)
    plt.axis('off')
    plt.title("decode predict", fontsize=5)

    # aux_predict (row:2 col:3)
    plt.subplot(3, 4, 7)
    aux_predict = torch.argmax(aux_predict.unsqueeze(0), dim=1)
    aux_semseg_color_map = draw_semseg_color_map(args, aux_predict)
    plt.imshow(aux_semseg_color_map)
    plt.axis('off')
    plt.title("aux predict", fontsize=5)

    # emb (row:3)
    for i in range(4):
        plt.subplot(3, 4, i + 9)
        emb_h_per_chans = emb[i: i + 1]
        emb_h_per_chans = torch.einsum('chw->hwc', emb_h_per_chans).numpy()
        plt.imshow(emb_h_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("emb_h_" + str(i + 1), fontsize=5)

    plt.suptitle(seq_name, fontsize=10)
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

def vis_ft_semseg_mem(args, events_voxel_grid, semseg_label, decode_predict, aux_predict,
                      emb, attn, seq_name, epoch, is_train=True):
    plt.figure()
    plt.subplots_adjust(left=0.2, bottom=0.1, right=0.8, top=0.9, wspace=0.1, hspace=0.5)

    events_voxel_grid, semseg_label, decode_predict, aux_predict, emb, attn = \
        events_voxel_grid.detach().cpu(), semseg_label.detach().cpu(), \
            decode_predict.detach().cpu(), aux_predict.detach().cpu(), \
            emb.detach().cpu(), attn.detach().cpu()

    _events_voxel_grid = copy.deepcopy(events_voxel_grid)  # avoid mask_events_preview() modify it
    # events frame (row:1 col:1)
    plt.subplot(3, 4, 1)
    events_frame = make_events_preview(_events_voxel_grid)
    plt.imshow(events_frame, cmap='gray')
    plt.axis('off')
    plt.title("events image", fontsize=5)

    # events frame norm (row:1 col:2)
    plt.subplot(3, 4, 2)
    events_frame_norm = make_events_preview_norm(_events_voxel_grid)
    plt.imshow(events_frame_norm, cmap='gray')
    plt.axis('off')
    plt.title("events image norm", fontsize=5)

    # attention map (row:1 col:3)
    plt.subplot(3, 4, 3)
    attn_ = attn[:, 1:, 1:].mean(dim=0).mean(dim=0)  # (12,196,196) -> (196)
    attn_ = attn_.unsqueeze(0)
    attn_ = attn_.reshape(
        shape=(attn_.shape[0], int(attn_.shape[1] ** .5), int(attn_.shape[1] ** .5)))
    attn_ = torch.einsum('chw->hwc', attn_).numpy()
    plt.imshow(attn_, cmap='viridis')
    plt.axis('off')
    plt.title("attention map", fontsize=5)

    # semseg label (row:2 col:1)
    plt.subplot(3, 4, 5)
    label_semseg_color_map = draw_semseg_color_map(args, semseg_label)
    plt.imshow(label_semseg_color_map)
    plt.axis('off')
    plt.title("semseg label", fontsize=5)

    # decode_predict (row:2 col:2)
    plt.subplot(3, 4, 6)
    decode_predict = torch.argmax(decode_predict.unsqueeze(0), dim=1)
    decode_semseg_color_map = draw_semseg_color_map(args, decode_predict)
    plt.imshow(decode_semseg_color_map)
    plt.axis('off')
    plt.title("decode predict", fontsize=5)

    # aux_predict (row:2 col:3)
    plt.subplot(3, 4, 7)
    aux_predict = torch.argmax(aux_predict.unsqueeze(0), dim=1)
    aux_semseg_color_map = draw_semseg_color_map(args, aux_predict)
    plt.imshow(aux_semseg_color_map)
    plt.axis('off')
    plt.title("aux predict", fontsize=5)

    # emb (row:3)
    for i in range(4):
        plt.subplot(3, 4, i + 9)
        emb_h_per_chans = emb[i: i + 1]
        emb_h_per_chans = torch.einsum('chw->hwc', emb_h_per_chans).numpy()
        plt.imshow(emb_h_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("emb_h_" + str(i + 1), fontsize=5)

    plt.suptitle(seq_name, fontsize=10)
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

def vis_ft_semseg_swin(args, events_voxel_grid, semseg_label, decode_predict, aux_predict,
                       emb_l1, emb_l2, emb_l3, emb_l4, emb_h, attn, seq_name, epoch, is_train=True):
    plt.figure()
    plt.subplots_adjust(left=0.2, bottom=0.1, right=0.8, top=0.9, wspace=0.1, hspace=0.5)

    events_voxel_grid, semseg_label, decode_predict, aux_predict, emb_l1, emb_l2, emb_l3, emb_l4, emb_h, attn = \
        events_voxel_grid.detach().cpu(), semseg_label.detach().cpu(), \
        decode_predict.detach().cpu(), aux_predict.detach().cpu(), \
        emb_l1.detach().cpu(), emb_l2.detach().cpu(), emb_l3.detach().cpu(), emb_l4.detach().cpu(), \
        emb_h.detach().cpu(), attn.detach().cpu()

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
    plt.subplot(7, 4, 3)
    attn_ = attn.mean(dim=0).mean(dim=0)  # (12,196,196) -> (196)
    attn_ = attn_.unsqueeze(0)
    attn_ = attn_.reshape(
        shape=(attn_.shape[0], int(attn_.shape[1] ** .5), int(attn_.shape[1] ** .5)))
    attn_ = torch.einsum('chw->hwc', attn_).numpy()
    plt.imshow(attn_, cmap='viridis')
    plt.axis('off')
    plt.title("attention map", fontsize=5)

    # semseg label (row:2 col:1)
    plt.subplot(7, 4, 5)
    label_semseg_color_map = draw_semseg_color_map(args, semseg_label)
    plt.imshow(label_semseg_color_map)
    plt.axis('off')
    plt.title("semseg label", fontsize=5)

    # decode_predict (row:2 col:2)
    plt.subplot(7, 4, 6)
    decode_predict = torch.argmax(decode_predict.unsqueeze(0), dim=1)
    decode_semseg_color_map = draw_semseg_color_map(args, decode_predict)
    plt.imshow(decode_semseg_color_map)
    plt.axis('off')
    plt.title("decode predict", fontsize=5)

    # aux_predict (row:2 col:3)
    plt.subplot(7, 4, 7)
    aux_predict = torch.argmax(aux_predict.unsqueeze(0), dim=1)
    aux_semseg_color_map = draw_semseg_color_map(args, aux_predict)
    plt.imshow(aux_semseg_color_map)
    plt.axis('off')
    plt.title("aux predict", fontsize=5)

    # emb_l1 (row:3)
    emb_l1 = emb2patch_frame(emb_l1.unsqueeze(0)).squeeze(0)
    for i in range(4):
        plt.subplot(7, 4, i + 9)
        emb_l1_per_chans = emb_l1[i: i + 1]
        emb_l1_per_chans = torch.einsum('chw->hwc', emb_l1_per_chans).numpy()
        plt.imshow(emb_l1_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("emb_l1_" + str(i + 1), fontsize=5)

    # emb_l2 (row:4)
    emb_l2 = emb2patch_frame(emb_l2.unsqueeze(0)).squeeze(0)
    for i in range(4):
        plt.subplot(7, 4, i + 13)
        emb_l2_per_chans = emb_l2[i: i + 1]
        emb_l2_per_chans = torch.einsum('chw->hwc', emb_l2_per_chans).numpy()
        plt.imshow(emb_l2_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("emb_l2_" + str(i + 1), fontsize=5)

    # emb_l3 (row:5)
    emb_l3 = emb2patch_frame(emb_l3.unsqueeze(0)).squeeze(0)
    for i in range(4):
        plt.subplot(7, 4, i + 17)
        emb_l3_per_chans = emb_l3[i: i + 1]
        emb_l3_per_chans = torch.einsum('chw->hwc', emb_l3_per_chans).numpy()
        plt.imshow(emb_l3_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("emb_l3_" + str(i + 1), fontsize=5)

    # emb_l4 (row:6)
    emb_l4 = emb2patch_frame(emb_l4.unsqueeze(0)).squeeze(0)
    for i in range(4):
        plt.subplot(7, 4, i + 21)
        emb_l4_per_chans = emb_l4[i: i + 1]
        emb_l4_per_chans = torch.einsum('chw->hwc', emb_l4_per_chans).numpy()
        plt.imshow(emb_l4_per_chans, cmap='viridis')
        plt.axis('off')
        plt.title("emb_l4_" + str(i + 1), fontsize=5)

    # emb_h (row:7)
    emb_h = emb2patch_frame(emb_h.unsqueeze(0)).squeeze(0)
    for i in range(4):
        plt.subplot(7, 4, i + 25)
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
        vis_dir = args.vis_val_dir
    Path(os.path.join(args.output_dir, vis_dir)).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(args.output_dir, vis_dir, figure_name),
                bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()
    plt.close()

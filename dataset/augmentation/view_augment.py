import numpy as np
import math

import torch

from utils.reshape import resize


def view_crop(view, scale=(0.8, 1.0), ratio=(3/4, 4/3)):
    sensor_h, sensor_w = view.shape[1], view.shape[2]

    area = sensor_w * sensor_h

    for attempt in range(10):  # 10次都没有crop就中心crop（不crop）
        target_area = np.random.uniform(*scale) * area
        aspect_ratio = np.random.uniform(sensor_w / sensor_h * ratio[0], sensor_w / sensor_h * ratio[1])

        crop_width = int(round(math.sqrt(target_area * aspect_ratio)))
        crop_height = int(round(math.sqrt(target_area / aspect_ratio)))

        if np.random.randint(0, 10) < 5:
            crop_width, crop_height = crop_height, crop_width

        if crop_width < sensor_w and crop_height < sensor_h:
            crop_start_x = np.random.randint(0, sensor_w - crop_width)
            crop_start_y = np.random.randint(0, sensor_h - crop_height)
            view = view[:, crop_start_y: crop_start_y + crop_height,
                                crop_start_x: crop_start_x + crop_width]
            # print(crop_start_x, crop_start_y)

            break

    return view

def view_resize(view, size, mode):
    resize_h, resize_w = size[0], size[1]
    view = resize(input=view.unsqueeze(0), size=(resize_h, resize_w), mode=mode).squeeze(0)

    return view

def view_horizontal_flip(view):
    flip_flag = False
    if np.random.random() < 0.5:
        view = torch.flip(view, dims=[2])
        flip_flag = True

    return view, flip_flag

def evg_time_flip(args, events_voxel_grid):
    time_flip_flag = False
    if np.random.random() < 0.5:
        time_flip_flag = True
        events_voxel_grid = torch.flip(events_voxel_grid, dims=[0])
        if args.num_bins == 5 or args.num_bins == 6:
            events_voxel_grid = -events_voxel_grid


    return events_voxel_grid, time_flip_flag

def frame_time_flip(frame):
    frame = -frame

    return frame

def evg_augment(args, events_voxel_grid, size, mode='nearest', seed=None):
    if seed is not None:
        np.random.seed(seed)

    # if args.phase == "pretrain" and args.use_random_denoise:
    #     events_voxel_grid = view_random_denoise(args, events_voxel_grid)
    events_voxel_grid = view_crop(events_voxel_grid, scale=(args.crop_min, 1))
    events_voxel_grid = view_resize(events_voxel_grid, size, mode)
    events_voxel_grid, _ = view_horizontal_flip(events_voxel_grid)
    # events_voxel_grid = view_shift(events_voxel_grid)
    events_voxel_grid, time_flip_flag = evg_time_flip(args, events_voxel_grid)

    return events_voxel_grid, time_flip_flag

def frame_augment(args, frame, seed=None, time_flip_flag=False):
    if seed is not None:
        np.random.seed(seed)

    frame = view_crop(frame, scale=(args.crop_min, 1))
    frame = view_resize(frame, (args.input_size, args.input_size), 'bicubic')
    frame, _ = view_horizontal_flip(frame)
    if time_flip_flag:
        frame = frame_time_flip(frame)

    return frame

def semseg_label_augment(args, label, size, seed=None):
    if seed is not None:
        np.random.seed(seed)

    label = view_crop(label, scale=(args.crop_min, 1))
    label = view_resize(label, size, 'nearest')
    label, _ = view_horizontal_flip(label)

    return label.long()

def flow_label_augment(args, flow, size, time_flip_flag, seed=None):
    if seed is not None:
        np.random.seed(seed)

    flow = view_crop(flow, scale=(args.crop_min, 1))

    org_h, org_w = flow.shape[-2], flow.shape[-1]
    flow = view_resize(flow, size, 'nearest')
    new_h, new_w = size[0], size[1]
    flow = torch.einsum('chw->hwc', flow)
    flow = flow * torch.tensor([new_w / org_w, new_h / org_h])
    flow = torch.einsum('hwc->chw', flow)

    flow, flag = view_horizontal_flip(flow)
    if flag:
        flow = torch.einsum('chw->hwc', flow)
        flow = flow * torch.tensor([-1.0, 1.0])
        flow = torch.einsum('hwc->chw', flow)
    if time_flip_flag:
        flow = torch.einsum('chw->hwc', flow)
        flow = flow * torch.tensor([-1.0, -1.0])
        flow = torch.einsum('hwc->chw', flow)

    return flow

def flow_label_valid_augment(args, flow, size, seed=None):
    if seed is not None:
        np.random.seed(seed)

    flow = view_crop(flow, scale=(args.crop_min, 1))
    flow = view_resize(flow, size, 'nearest')
    flow, _ = view_horizontal_flip(flow)

    return flow

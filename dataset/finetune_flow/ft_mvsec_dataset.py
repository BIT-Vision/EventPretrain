import os
import numpy as np
import h5py
import cv2
import random
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

from dataset.dataset_utils.events_to_voxel_grid import events_to_voxel_grid
from dataset.dataset_utils.events_to_image import events_to_image_ecdp, events_to_image_mem, remove_hot_pixel_mem, events_to_EvRep
from dataset.augmentation.events_augment import events_augment
from dataset.augmentation.view_augment import evg_augment, flow_label_augment, flow_label_valid_augment, view_resize
from visualize.visualize_utils.make_events_preview import make_events_preview

Valid_Time_Index = {
    'indoor_flying1': [314, 2199],
    'indoor_flying2': [314, 2199],
    'indoor_flying3': [314, 2199],
    'outdoor_day1': [245, 3000],
    'outdoor_day2': [4375, 7002],
}

class FinetuneMVSECDataset(Dataset):
    def __init__(self, args, is_train=True):
        if is_train:
            train_seq = []
            train_seq_name_list = ['outdoor_day1', 'outdoor_day2', 'indoor_flying1']
            for train_seq_name in train_seq_name_list:
                train_seq.append(FinetuneMVSECSeqDataset(args, is_train, train_seq_name))
            self.train_dataset = torch.utils.data.ConcatDataset(train_seq)
        else:
            self.val_seq = []
            val_seq_name_list = ['indoor_flying1', 'indoor_flying2', 'indoor_flying3']
            for val_seq_name in val_seq_name_list:
                self.val_seq.append(FinetuneMVSECSeqDataset(args, is_train, val_seq_name))
            # self.val_dataset = torch.utils.data.ConcatDataset(val_seq)

    def get_train_dataset(self):
        return self.train_dataset

    def get_val_dataset(self, index):
        return self.val_seq[index]


class FinetuneMVSECSeqDataset(Dataset):
    def __init__(self, args, is_train, seq_name):
        self.args = args
        self.is_train = is_train
        self.seq_name = seq_name

        self.raw_index_shift = Valid_Time_Index[seq_name][0]
        self.raw_index_max = Valid_Time_Index[seq_name][1] - 1 - (args.skip_num - 1)
        self.raw_index = [i for i in range(self.raw_index_shift, self.raw_index_max)]

        if seq_name == "indoor_flying1":
            index_length = int(0.01 * (self.raw_index_max - self.raw_index_shift))
            random.seed(args.seed)
            train_raw_index = random.sample(self.raw_index, index_length)
            if is_train:
                self.raw_index = train_raw_index
                self.data_length = len(self.raw_index)
                assert self.data_length == index_length
            else:
                self.raw_index = [i for i in self.raw_index if i not in train_raw_index]
                self.data_length = len(self.raw_index)
                assert self.data_length == self.raw_index_max - self.raw_index_shift - index_length
        else:
            self.data_length = len(self.raw_index)
            assert self.data_length == self.raw_index_max - self.raw_index_shift  # 2999 - 245 = 2754

        # load file
        dataset_root = args.mvsec_root
        self.data_filepath = os.path.join(dataset_root, seq_name + "_data.hdf5")
        self.gt_filepath = os.path.join(dataset_root, seq_name + "_gt.hdf5")

        data_file = h5py.File(self.data_filepath, 'r')
        self.events_data = data_file.get('davis/left/events')  # (10587257,4)
        self.image_data = data_file.get('davis/left/image_raw')  # (11937,260,346)
        self.image_ts_data = data_file.get('davis/left/image_raw_ts')  # (11937)
        self.image_event_inds = data_file.get('davis/left/image_raw_event_inds')  # (11937)
        assert len(self.image_data) == len(self.image_ts_data)

        gt_file = h5py.File(self.gt_filepath, 'r')
        self.flow_dist_data = gt_file.get('davis/left/flow_dist')  # (5134,2,260,346)  # 2维：第一个为x，第二个为y
        self.flow_dist_ts = gt_file.get('davis/left/flow_dist_ts')  # (5134)
        self.flow_dist_ts_numpy = np.array(self.flow_dist_ts, dtype=np.float64)  # (5134) float64而不是float32

        self.image_length = len(self.image_data)  # (11937)
        self.event_length = len(self.events_data)  # (10587257)
        self.flow_length = len(self.flow_dist_data)  # (5134)

        assert self.data_length <= self.image_length

    """
    The ground truth flow maps are not time synchronized with the grayscale images. Therefore, we
    need to propagate the ground truth flow over the time between two images.
    This function assumes that the ground truth flow is in terms of pixel displacement, not velocity.

    Pseudo code for this process is as follows:

    x_orig = range(cols)
    y_orig = range(rows)
    x_prop = x_orig
    y_prop = y_orig
    Find all GT flows that fit in [image_timestamp, image_timestamp+image_dt].
    for all of these flows:
      x_prop = x_prop + gt_flow_x(x_prop, y_prop)
      y_prop = y_prop + gt_flow_y(x_prop, y_prop)

    The final flow, then, is x_prop - x-orig, y_prop - y_orig.
    Note that this is flow in terms of pixel displacement, with units of pixels, not pixel velocity.

    Inputs:
      x_flow_in, y_flow_in - list of numpy arrays, each array corresponds to per pixel flow at
        each timestamp.
      gt_timestamps - timestamp for each flow array.
      start_time, end_time - gt flow will be estimated between start_time and end time.
    """
    def gen_correspond_gt_flow(self, flows, flows_ts, start_time, end_time):
        flow_length = len(flows)  # len(flows) = 1
        assert flow_length == len(flows_ts) - 1  # len(flows_ts) = 2

        x_flow = flows[0][0]  # (260,346)
        y_flow = flows[0][1]  # (260,346)
        gt_dt = flows_ts[1] - flows_ts[0]  # 128.0
        pre_dt = end_time - start_time  # 0.021898984909057617

        if start_time > flows_ts[0] and end_time <= flows_ts[1]:
            x_flow *= pre_dt / gt_dt
            y_flow *= pre_dt / gt_dt

            return np.concatenate((x_flow[np.newaxis, :], y_flow[np.newaxis, :]), axis=0)  # (2,260,346)

        else:
            x_indices, y_indices = np.meshgrid(np.arange(x_flow.shape[1]), np.arange(x_flow.shape[0]))

            x_indices = x_indices.astype(np.float32)
            y_indices = y_indices.astype(np.float32)

            orig_x_indices = np.copy(x_indices)
            orig_y_indices = np.copy(y_indices)

            # Mask keeps track of the points that leave the image, and zeros out the flow afterwards.
            x_mask = np.ones(x_indices.shape, dtype=bool)
            y_mask = np.ones(y_indices.shape, dtype=bool)

            scale_factor = (flows_ts[1] - start_time) / gt_dt
            total_dt = flows_ts[1] - start_time

            self.prop_flow(x_flow, y_flow, x_indices, y_indices, x_mask, y_mask, scale_factor=scale_factor)

            for i in range(1, flow_length - 1):
                x_flow = flows[i][0]
                y_flow = flows[i][1]

                self.prop_flow(x_flow, y_flow, x_indices, y_indices, x_mask, y_mask)

                total_dt += flows_ts[i + 1] - flows_ts[i]

            gt_dt = flows_ts[flow_length] - flows_ts[flow_length - 1]
            pred_dt = end_time - flows_ts[flow_length - 1]
            total_dt += pred_dt

            x_flow = flows[flow_length - 1][0]
            y_flow = flows[flow_length - 1][1]

            scale_factor = pred_dt / gt_dt

            self.prop_flow(x_flow, y_flow, x_indices, y_indices, x_mask, y_mask, scale_factor)

            x_shift = x_indices - orig_x_indices
            y_shift = y_indices - orig_y_indices
            x_shift[~x_mask] = 0
            y_shift[~y_mask] = 0

            return np.concatenate((x_shift[np.newaxis, :], y_shift[np.newaxis, :]), axis=0)

    def prop_flow(self, x_flow, y_flow, x_indices, y_indices, x_mask, y_mask, scale_factor=1.0):
        flow_x_interp = cv2.remap(x_flow, x_indices, y_indices, cv2.INTER_NEAREST)
        flow_y_interp = cv2.remap(y_flow, x_indices, y_indices, cv2.INTER_NEAREST)

        x_mask[flow_x_interp == 0] = False
        y_mask[flow_y_interp == 0] = False

        x_indices += flow_x_interp * scale_factor
        y_indices += flow_y_interp * scale_factor

    def __getitem__(self, index):
        seed = np.random.randint(1000)

        # raw_index = index + self.raw_index_shift
        # assert raw_index < self.raw_index_max
        raw_index = self.raw_index[index]

        # image1 = self.image_data[raw_index]  # (260,340)
        image1_ts = self.image_ts_data[raw_index]  # 1506117923.8601716
        image1_event_index = self.image_event_inds[raw_index]  # 9186436
        # image2 = self.image_data[raw_index + self.args.skip_num]  # (260,340)
        image2_ts = self.image_ts_data[raw_index + self.args.skip_num]  # 1506117929.2473714
        image2_event_index = self.image_event_inds[raw_index + self.args.skip_num]  # 10418212
        assert image1_event_index < image2_event_index
        assert image2_event_index < self.event_length

        # events_voxel_grid
        events = self.events_data[image1_event_index: image2_event_index]  # (4753,4)

        # events augment
        if self.is_train:
            events = events_augment(self.args, events, size=(self.args.mvsec_sensor_h, self.args.mvsec_sensor_w))
        else:
            if self.args.val_event_noise:
                events = events_augment(self.args, events, size=(self.args.cal_sensor_h, self.args.cal_sensor_w))

        if self.args.num_bins == 2:
            events_voxel_grid_org = events_to_image_ecdp(self.args, events, size=(self.args.mvsec_sensor_h, self.args.mvsec_sensor_w))
        elif self.args.num_bins == 3:
            events_voxel_grid_org = events_to_image_mem(self.args, events, size=(self.args.mvsec_sensor_h, self.args.mvsec_sensor_w))
            events_voxel_grid_org = events_voxel_grid_org / 255  # Transforms.ToTensor()
            events_voxel_grid_org = remove_hot_pixel_mem(events_voxel_grid_org)
        else:
            if self.args.use_evrepsl:
                events_voxel_grid_org = events_to_EvRep(events[:, 0].astype(np.int16), events[:, 1].astype(np.int16),
                                               events[:, 2], events[:, 3], (self.args.mvsec_sensor_w, self.args.mvsec_sensor_h))

                events_voxel_grid_org = torch.from_numpy(events_voxel_grid_org).to(torch.float32)
            else:
                events_voxel_grid_org = events_to_voxel_grid(self.args, events, size=(self.args.mvsec_sensor_h, self.args.mvsec_sensor_w))

        if self.is_train:
            events_voxel_grid, time_flip_flag = evg_augment(self.args, events_voxel_grid_org,
                                                            size=(self.args.input_size, self.args.input_size),
                                                            mode='bilinear', seed=seed)
            events_voxel_grid_org, time_flip_flag = evg_augment(self.args, events_voxel_grid_org,
                                                                size=(self.args.mvsec_sensor_h, self.args.mvsec_sensor_w),
                                                                mode='bilinear', seed=seed)
        else:
            events_voxel_grid = view_resize(events_voxel_grid_org, (self.args.input_size, self.args.input_size), 'bilinear')

        if self.args.num_bins == 2:
            events_voxel_grid = events_voxel_grid / (events_voxel_grid.amax([1, 2], True) + 1)
            events_voxel_grid = (events_voxel_grid - 0.5) * 2
        elif self.args.num_bins == 3:
            if events_voxel_grid[0::2, :, :].max() != 0:
                factor = 1.0 / events_voxel_grid[0::2, :, :].max()
            else:
                factor = 1.0 / 0.001
            events_voxel_grid[0::2, :, :] = events_voxel_grid[0::2, :, :] * factor

        # events_frame = make_events_preview(events_voxel_grid_org)
        # plt.imshow(events_frame, cmap='gray')
        # plt.axis('off')
        # plt.show()
        # plt.close()
        #
        # events_frame = make_events_preview(events_voxel_grid)
        # plt.imshow(events_frame, cmap='gray')
        # plt.axis('off')
        # plt.show()
        # plt.close()

        # flow
        flow_left_index = np.searchsorted(self.flow_dist_ts_numpy, image1_ts, side='right') - 1  # 993
        flow_right_index = np.searchsorted(self.flow_dist_ts_numpy, image2_ts, side='right')  # 994
        assert flow_left_index <= flow_right_index
        assert flow_left_index < self.flow_length
        assert flow_right_index < self.flow_length

        flows = self.flow_dist_data[flow_left_index: flow_right_index]  # (1,2,260,346)
        flows_ts = self.flow_dist_ts_numpy[flow_left_index:flow_right_index + 1]  # (1,2,260,346)
        final_flow = self.gen_correspond_gt_flow(flows, flows_ts, image1_ts, image2_ts)  # (2,260,346)
        final_flow = torch.from_numpy(final_flow)

        final_flow_valid = (torch.norm(final_flow, p=2, dim=0, keepdim=False) > 0) & \
                           (final_flow[0].abs() < 1000) & \
                           (final_flow[1].abs() < 1000)
        final_flow_valid = final_flow_valid.float().unsqueeze(0)  # (1,260,346)

        if self.is_train:
            # if self.seq_name == "outdoor_day1" or self.seq_name == "outdoor_day2":
            #     final_flow = final_flow[:, :self.args.mvsec_sensor_h, start_w:start_w + sensor_w]
            #     final_flow_valid = final_flow_valid[:, :self.args.mvsec_sensor_h, start_w:start_w + sensor_w]

            final_flow = flow_label_augment(self.args, final_flow,
                                            (self.args.mvsec_sensor_h, self.args.mvsec_sensor_w),
                                            time_flip_flag, seed=seed)
            final_flow_valid = flow_label_valid_augment(self.args, final_flow_valid,
                                                        (self.args.mvsec_sensor_h, self.args.mvsec_sensor_w),
                                                        seed=seed)
        # plt.imshow(final_flow[0], cmap='gray')
        # plt.axis('off')
        # plt.show()
        # plt.close()
        # plt.imshow(final_flow[1], cmap='gray')
        # plt.axis('off')
        # plt.show()
        # plt.close()
        # plt.imshow(final_flow_valid.squeeze(0), cmap='gray')
        # plt.axis('off')
        # plt.show()
        # plt.close()

        data = {
            "events_voxel_grid": events_voxel_grid,  # (5,224,224)
            "events_voxel_grid_org": events_voxel_grid_org,  # (5,260,346)
            "flow_label": final_flow,  # (2,260,346)
            "flow_label_valid": final_flow_valid,  # (1,260,346)
            "seq_name": self.seq_name
        }

        return data

    def __len__(self):
        return self.data_length


class FinetuneMVSECTestDataset(Dataset):
    def __init__(self, args):
        self.val_seq = []
        val_seq_name_list = ['indoor_flying1', 'indoor_flying2', 'indoor_flying3']
        for val_seq_name in val_seq_name_list:
            self.val_seq.append(FinetuneMVSECSeqTestDataset(args, False, val_seq_name))
        # self.val_dataset = torch.utils.data.ConcatDataset(val_seq)

    def get_val_dataset(self, index):
        return self.val_seq[index]

class FinetuneMVSECSeqTestDataset(FinetuneMVSECSeqDataset):
    def __getitem__(self, index):
        # raw_index = index + self.raw_index_shift
        # assert raw_index < self.raw_index_max
        raw_index = self.raw_index[index]

        # image1 = self.image_data[raw_index]  # (260,340)
        image1_ts = self.image_ts_data[raw_index]  # 1506117923.8601716
        image1_event_index = self.image_event_inds[raw_index]  # 9186436
        # image2 = self.image_data[raw_index + self.args.skip_num]  # (260,340)
        image2_ts = self.image_ts_data[raw_index + self.args.skip_num]  # 1506117929.2473714
        image2_event_index = self.image_event_inds[raw_index + self.args.skip_num]  # 10418212
        assert image1_event_index < image2_event_index
        assert image2_event_index < self.event_length

        # events_voxel_grid
        events = self.events_data[image1_event_index: image2_event_index]  # (4753,4)

        # flow
        flow_left_index = np.searchsorted(self.flow_dist_ts_numpy, image1_ts, side='right') - 1  # 993
        flow_right_index = np.searchsorted(self.flow_dist_ts_numpy, image2_ts, side='right')  # 994
        assert flow_left_index <= flow_right_index
        assert flow_left_index < self.flow_length
        assert flow_right_index < self.flow_length

        flows = self.flow_dist_data[flow_left_index: flow_right_index]  # (1,2,260,346)
        flows_ts = self.flow_dist_ts_numpy[flow_left_index:flow_right_index + 1]  # (1,2,260,346)
        final_flow = self.gen_correspond_gt_flow(flows, flows_ts, image1_ts, image2_ts)  # (2,260,346)
        final_flow = torch.from_numpy(final_flow)

        final_flow_valid = (torch.norm(final_flow, p=2, dim=0, keepdim=False) > 0) & \
                           (final_flow[0].abs() < 1000) & \
                           (final_flow[1].abs() < 1000)
        final_flow_valid = final_flow_valid.float().unsqueeze(0)  # (1,260,346)

        data = {
            "events": events,
            "flow_label": final_flow,
            "flow_label_valid": final_flow_valid,
            "seq_name": self.seq_name
        }

        return data

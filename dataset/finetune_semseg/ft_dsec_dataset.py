import os
import h5py
import hdf5plugin
os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

from dataset.dataset_utils.events_to_voxel_grid import events_to_voxel_grid
from dataset.dataset_utils.events_to_image import events_to_image_ecdp, events_to_image_mem, remove_hot_pixel_mem, events_to_EvRep
from dataset.augmentation.events_augment import events_augment
from dataset.augmentation.view_augment import evg_augment, semseg_label_augment, view_resize
from visualize.visualize_utils.make_events_preview import make_events_preview


class FinetuneDSECDataset(Dataset):
    def __init__(self, args, is_train=True):
        if is_train:
            train_seq = []
            train_seq_name_list = ['zurich_city_00_a', 'zurich_city_01_a', 'zurich_city_02_a',
                                   'zurich_city_04_a', 'zurich_city_05_a', 'zurich_city_06_a',
                                   'zurich_city_07_a', 'zurich_city_08_a']
            for train_seq_name in train_seq_name_list:
                train_seq.append(FinetuneDSECSeqDataset(args, is_train, train_seq_name))
            self.train_dataset = torch.utils.data.ConcatDataset(train_seq)
        else:
            val_seq = []
            val_seq_name_list = ['zurich_city_13_a', 'zurich_city_14_c', 'zurich_city_15_a']
            for val_seq_name in val_seq_name_list:
                val_seq.append(FinetuneDSECSeqDataset(args, is_train, val_seq_name))
            self.val_dataset = torch.utils.data.ConcatDataset(val_seq)

    def get_train_dataset(self):
        return self.train_dataset

    def get_val_dataset(self):
        return self.val_dataset

class FinetuneDSECSeqDataset(Dataset):
    # seq_name (e.g. zurich_city_00_a)
    # ├── semantic
    # │   ├── left
    # │   │   ├── 11classes
    # │   │   │   ├── 000000.png
    # │   │   │   └── ...
    # │   │   └── 19classes
    # │   │       ├── 000000.png
    # │   │       └── ...
    # │   └── timestamps.txt
    # └── events
    #     └── left
    #         ├── events.h5
    #         └── rectify_map.h5
    def __init__(self, args, is_train, seq_name, remove_time_window=250):
        self.args = args
        self.is_train = is_train
        self.seq_name = seq_name

        if is_train:
            dataset_root = args.dsec_train_root
        else:
            dataset_root = args.dsec_val_root
        self.dataset_path = os.path.join(dataset_root, seq_name)

        self.remove_time_window = remove_time_window
        self.timestamps = np.loadtxt(os.path.join(self.dataset_path, 'semantic', 'left', seq_name + '_semantic_timestamps.txt'), dtype='int64')

        if args.num_classes == 11:
            self.label_dir_path = os.path.join(self.dataset_path, 'semantic', 'left', '11classes')
        elif args.num_classes == 19:
            self.label_dir_path = os.path.join(self.dataset_path, 'semantic', 'left', '19classes')
        else:
            raise ValueError
        self.label_pathstrings = sorted(os.listdir(self.label_dir_path))
        assert len(self.label_pathstrings) == self.timestamps.size

        # Remove several label paths and corresponding timestamps in the remove_time_window.
        # This is necessary because we do not have enough events before the first label.
        self.timestamps = self.timestamps[(self.remove_time_window // 100 + 1) * 2:]  # 6
        del self.label_pathstrings[:(self.remove_time_window // 100 + 1) * 2]
        assert len(self.label_pathstrings) == self.timestamps.size

        events_data_file_path = os.path.join(self.dataset_path, 'events', 'left', 'events.h5')
        events_data_h5py_file = h5py.File(events_data_file_path, 'r')
        self.events = dict()
        for dset_str in ['p', 'x', 'y', 't']:
            self.events[dset_str] = events_data_h5py_file['events/{}'.format(dset_str)]

        if "t_offset" in list(events_data_h5py_file.keys()):  # ['events', 'ms_to_idx', 't_offset']
            self.t_offset = int(events_data_h5py_file['t_offset'][()])
        else:
            self.t_offset = 0
        self.t_final = int(self.events['t'][-1]) + self.t_offset

        # This is the mapping from milliseconds to event index:
        # It is defined such that
        # (1) t[ms_to_idx[ms]] >= ms*1000
        # (2) t[ms_to_idx[ms] - 1] < ms*1000
        # ,where 'ms' is the time in milliseconds and 't' the event timestamps in microseconds.
        # As an example, given 't' and 'ms':
        # t:    0     500    2100    5000    5000    7100    7200    7200    8100    9000
        # ms:   0       1       2       3       4       5       6       7       8       9
        # we get
        # ms_to_idx:
        #       0       2       2       3       3       3       5       5       8       9
        self.ms_to_idx = np.asarray(events_data_h5py_file['ms_to_idx'], dtype='int64')

        events_rect_file_path = os.path.join(self.dataset_path, 'events', 'left', 'rectify_map.h5')
        self.rectify_ev_maps = dict()
        events_rect_h5py_file = h5py.File(events_rect_file_path, 'r')
        self.rectify_ev_maps = events_rect_h5py_file['rectify_map'][()]  # (480,640,2)

    def augment_parser(self, parser):
        def new_parser(x):
            return parser(x)

        return new_parser

    def get_time_indices_offsets(self, time_array, time_start_us, time_end_us):
        assert time_array.ndim == 1

        idx_start = -1
        if time_array[-1] < time_start_us:
            return time_array.size, time_array.size
        else:
            for idx_from_start in range(0, time_array.size, 1):
                if time_array[idx_from_start] >= time_start_us:
                    idx_start = idx_from_start
                    break
        assert idx_start >= 0
        assert time_array[idx_start] >= time_start_us
        if idx_start > 0:
            assert time_array[idx_start - 1] < time_start_us

        idx_end = time_array.size
        for idx_from_end in range(time_array.size - 1, -1, -1):
            if time_array[idx_from_end] >= time_end_us:
                idx_end = idx_from_end
            else:
                break
        if idx_end < time_array.size:
            assert time_array[idx_end] >= time_end_us
        if idx_end > 0:
            assert time_array[idx_end - 1] < time_end_us

        return idx_start, idx_end

    def get_events(self, t_end_us, events_num):
        """
            Get events (p, x, y, t) with fixed number of events
            Args:
                t_end_us: end time in microseconds
                events_num: number of events to load
            Returns:
                events: dictionary of (p, x, y, t) or None if the time window cannot be retrieved
        """
        t_end_us -= self.t_offset
        # μs -> ms
        t_end_lower_ms = math.floor(t_end_us / 1000)
        t_end_upper_ms = math.ceil(t_end_us / 1000)
        t_end_lower_ms_idx = self.ms_to_idx[t_end_lower_ms]
        t_end_upper_ms_idx = self.ms_to_idx[t_end_upper_ms]

        if t_end_lower_ms_idx != t_end_upper_ms_idx:
            time_array_conservative = np.asarray(self.events['t'][t_end_lower_ms_idx:t_end_upper_ms_idx])
            _, idx_end_offset = self.get_time_indices_offsets(time_array_conservative, t_end_us, t_end_us)
            t_end_us_idx = t_end_lower_ms_idx + idx_end_offset
        else:
            t_end_us_idx = t_end_lower_ms_idx

        t_start_us_idx = t_end_us_idx - events_num
        if t_start_us_idx < 0:
            t_start_us_idx = 0

        events = dict()
        for dset_str in self.events.keys():
            events[dset_str] = np.asarray(self.events[dset_str][t_start_us_idx:t_end_us_idx])

        return events

    def get_label(self, label_path):
        label = Image.open(label_path)
        label = np.array(label)
        label = torch.from_numpy(label).unsqueeze(0).long()

        return label

    def __getitem__(self, index):
        seed = np.random.randint(1000)

        ts_end = self.timestamps[index * 2]
        if self.is_train:
            fix_events_num = self.args.fix_events_num
        else:
            fix_events_num = self.args.val_fix_events_num
        events_data = self.get_events(ts_end, fix_events_num)

        if fix_events_num >= events_data['t'].size:
            start_index = 0
        else:
            start_index = -fix_events_num
        p = events_data['p'][start_index:]
        t = events_data['t'][start_index:]
        x = events_data['x'][start_index:]
        y = events_data['y'][start_index:]

        assert self.rectify_ev_maps.shape == (self.args.dsec_org_sensor_h, self.args.dsec_org_sensor_w, 2)  # (480,640,2)
        assert x.max() < self.args.dsec_org_sensor_w
        assert y.max() < self.args.dsec_org_sensor_h
        xy_rect = self.rectify_ev_maps[y, x]
        x_rect = xy_rect[:, 0]
        y_rect = xy_rect[:, 1]

        # (remove 40 bottom rows)
        mask = (x_rect >= 0) & (x_rect < self.args.dsec_sensor_w) & \
               (y_rect >= 0) & (y_rect < self.args.dsec_sensor_h)
        x_rect = x_rect[mask]
        y_rect = y_rect[mask]
        p = p[mask]
        t = t[mask]

        events = np.stack([x_rect, y_rect, t, p], axis=-1)

        # events augment
        if self.is_train:
            events = events_augment(self.args, events, size=(self.args.dsec_sensor_h, self.args.dsec_sensor_w))
        else:
            if self.args.val_event_noise:
                events = events_augment(self.args, events, size=(self.args.cal_sensor_h, self.args.cal_sensor_w))

        if self.args.num_bins == 2:
            # events = events_reshape(events, self.args.dsec_sensor_w, self.args.dsec_sensor_h,
            #                         self.args.input_size, self.args.input_size)
            # events_voxel_grid = events_to_image_ecdp(self.args, events, size=(self.args.input_size, self.args.input_size))
            events_voxel_grid = events_to_image_ecdp(self.args, events, size=(self.args.dsec_sensor_h, self.args.dsec_sensor_w))
        elif self.args.num_bins == 3:
            events_voxel_grid = events_to_image_mem(self.args, events, size=(self.args.dsec_sensor_h, self.args.dsec_sensor_w))
            events_voxel_grid = events_voxel_grid / 255  # Transforms.ToTensor()
            # events_voxel_grid = view_resize_mem(events_voxel_grid, (self.args.input_size, self.args.input_size))
            events_voxel_grid = remove_hot_pixel_mem(events_voxel_grid)
        else:
            if self.args.use_evrepsl:
                events_voxel_grid = events_to_EvRep(events[:, 0].astype(np.int16), events[:, 1].astype(np.int16),
                                               events[:, 2], events[:, 3], (self.args.dsec_sensor_w, self.args.dsec_sensor_h))

                events_voxel_grid = torch.from_numpy(events_voxel_grid).to(torch.float32)
            else:
                events_voxel_grid = events_to_voxel_grid(self.args, events, size=(self.args.dsec_sensor_h, self.args.dsec_sensor_w))

        if self.is_train:
            events_voxel_grid, _ = evg_augment(self.args, events_voxel_grid,
                                               size=(self.args.input_size, self.args.input_size), mode='bilinear', seed=seed)
        else:
            events_voxel_grid = view_resize(events_voxel_grid, (self.args.input_size, self.args.input_size), 'bilinear')

        if self.args.num_bins == 2:
            events_voxel_grid = events_voxel_grid / (events_voxel_grid.amax([1, 2], True) + 1)
            events_voxel_grid = (events_voxel_grid - 0.5) * 2
        elif self.args.num_bins == 3:
            factor = 1.0 / events_voxel_grid[0::2, :, :].max()
            events_voxel_grid[0::2, :, :] = events_voxel_grid[0::2, :, :] * factor

        # events_frame = make_events_preview(events_voxel_grid)
        # plt.imshow(events_frame, cmap='gray')
        # plt.axis('off')
        # plt.show()
        # plt.close()

        # label
        label_path = os.path.join(self.label_dir_path, self.label_pathstrings[index * 2])
        semseg_label = self.get_label(label_path)
        if self.is_train:
            semseg_label = semseg_label_augment(self.args, semseg_label.float(),
                                                (self.args.dsec_sensor_h, self.args.dsec_sensor_w), seed=seed)
        # plt.imshow(semseg_label.squeeze(0), cmap='viridis')
        # plt.axis('off')
        # plt.show()
        # plt.close()

        data = {
            "events_voxel_grid": events_voxel_grid,
            "semseg_label": semseg_label,  # (1,440,640)
            "seq_name": self.seq_name
        }

        return data

    def __len__(self):
        return (self.timestamps.size + 1) // 2


class FinetuneDSECTestDataset(Dataset):
    def __init__(self, args):
        val_seq = []
        val_seq_name_list = ['zurich_city_13_a', 'zurich_city_14_c', 'zurich_city_15_a']
        for val_seq_name in val_seq_name_list:
            val_seq.append(FinetuneDSECSeqTestDataset(args, False, val_seq_name))
        self.val_dataset = torch.utils.data.ConcatDataset(val_seq)

    def get_val_dataset(self):
        return self.val_dataset


class FinetuneDSECSeqTestDataset(FinetuneDSECSeqDataset):
    def __getitem__(self, index):
        ts_end = self.timestamps[index * 2]
        if self.is_train:
            fix_events_num = self.args.fix_events_num
        else:
            fix_events_num = self.args.val_fix_events_num
        events_data = self.get_events(ts_end, fix_events_num)

        if fix_events_num >= events_data['t'].size:
            start_index = 0
        else:
            start_index = -fix_events_num
        p = events_data['p'][start_index:]
        t = events_data['t'][start_index:]
        x = events_data['x'][start_index:]
        y = events_data['y'][start_index:]

        assert self.rectify_ev_maps.shape == (self.args.dsec_org_sensor_h, self.args.dsec_org_sensor_w, 2)  # (480,640,2)
        assert x.max() < self.args.dsec_org_sensor_w
        assert y.max() < self.args.dsec_org_sensor_h
        xy_rect = self.rectify_ev_maps[y, x]
        x_rect = xy_rect[:, 0]
        y_rect = xy_rect[:, 1]

        # (remove 40 bottom rows)
        mask = (x_rect >= 0) & (x_rect < self.args.dsec_sensor_w) & \
               (y_rect >= 0) & (y_rect < self.args.dsec_sensor_h)
        x_rect = x_rect[mask]
        y_rect = y_rect[mask]
        p = p[mask]
        t = t[mask]

        events = np.stack([x_rect, y_rect, t, p], axis=-1)

        # label
        label_path = os.path.join(self.label_dir_path, self.label_pathstrings[index * 2])
        semseg_label = self.get_label(label_path)

        data = {
            "events": events,
            "semseg_label": semseg_label,  # (1,440,640)
            "seq_name": self.seq_name
        }

        return data

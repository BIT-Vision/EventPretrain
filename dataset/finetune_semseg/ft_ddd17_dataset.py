import os
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


class FinetuneDDD17Dataset(Dataset):
    def __init__(self, args, is_train=True):
        if is_train:
            train_seq = []
            train_seq_name_list = ['dir0', 'dir3', 'dir4', 'dir6', 'dir7']
            for train_seq_name in train_seq_name_list:
                train_seq.append(FinetuneDDD17SeqDataset(args, is_train, train_seq_name))
            self.train_dataset = torch.utils.data.ConcatDataset(train_seq)
        else:
            val_seq = []
            val_seq_name_list = ['dir1']
            for val_seq_name in val_seq_name_list:
                val_seq.append(FinetuneDDD17SeqDataset(args, is_train, val_seq_name))
            self.val_dataset = torch.utils.data.ConcatDataset(val_seq)

    def get_train_dataset(self):
        return self.train_dataset

    def get_val_dataset(self):
        return self.val_dataset


class FinetuneDDD17SeqDataset(Dataset):
    def __init__(self, args, is_train, seq_name):
        self.args = args
        self.is_train = is_train
        self.seq_name = seq_name

        if is_train:
            self.dataset_root = args.ddd17_train_root
        else:
            self.dataset_root = args.ddd17_val_root

        # load label
        self.label_files = sorted(os.listdir(os.path.join(self.dataset_root, seq_name, "segmentation_masks")))

        # load events and image_idx -> event index mapping
        self.img_t_events_index = self.load_events_index(os.path.join(self.dataset_root, seq_name))

        # load events
        events_t_file = os.path.join(self.dataset_root, seq_name, "events.dat.t")
        events_xyp_file = os.path.join(self.dataset_root, seq_name, "events.dat.xyp")
        self.t_events, self.xyp_events = self.load_events(events_t_file, events_xyp_file)

    def load_events_index(self, dir, t_interval=50):
        # idx.npy : t0_ns idx0, t1_ns idx1, ..., tj_ns idxj, ..., tN_ns idxN
        # This file contains a mapping from j -> tj_ns idxj,
        # where j+1 is the idx of the img with timestamp tj_ns (in nanoseconds)
        # and idxj is the idx of the last event before the img (in events.dat.t and events.dat.xyp)
        if t_interval == 10:
            img_t_events_index = np.load(os.path.join(dir, "index/index_10ms.npy"))
        elif t_interval == 50:
            img_t_events_index = np.load(os.path.join(dir, "index/index_50ms.npy"))
        elif t_interval == 250:
            img_t_events_index = np.load(os.path.join(dir, "index/index_250ms.npy"))
        else:
            img_t_events_index = np.load(os.path.join(dir, "index/index_50ms.npy"))

        return img_t_events_index

    def load_events(self, t_file, xyp_file):
        num_events = int(os.path.getsize(t_file) / 8)  # timestamps take 8 bytes

        # events.dat.t : t0_ns, t1_ns, ..., tM_ns
        # events.dat.xyp : x0 y0 p0, ..., xM yM pM
        t_events = np.memmap(t_file, dtype="int64", mode="r", shape=(num_events, 1))
        xyp_events = np.memmap(xyp_file, dtype="int16", mode="r", shape=(num_events, 3))

        return t_events, xyp_events

    def extract_events_from_memmap(self, img_index, fixed_duration=False):
        events_num = self.args.fix_events_num + 10000

        if fixed_duration:
            _, events_index, events_index_before = self.img_t_events_index[img_index]
            events_index_before = max([events_index_before, 0])
        else:
            _, events_index, _ = self.img_t_events_index[img_index]
            events_index_before = max([events_index - events_num, 0])
        events_between_imgs = np.concatenate([
            np.array(self.t_events[events_index_before:events_index], dtype="float32"),
            np.array(self.xyp_events[events_index_before:events_index], dtype="float32")
        ], -1)

        events_between_imgs = events_between_imgs[:, [1, 2, 0, 3]]  # events have format xytp, and p is in [0,1]

        return events_between_imgs

    def get_label(self, label_path):
        label = Image.open(label_path)
        label = np.array(label)
        label = torch.from_numpy(label).unsqueeze(0).long()

        return label

    def __getitem__(self, index):
        seed = np.random.randint(1000)
        img_index = int(self.label_files[index][:-4].split("_")[-1]) - 1

        # events has form x, y, t_ns, p (in [0,1])
        events = self.extract_events_from_memmap(img_index)
        x = events[:, 0]
        y = events[:, 1]
        mask = (x >= 0) & (x < self.args.ddd17_sensor_w) & \
               (y >= 0) & (y < self.args.ddd17_sensor_h)
        events = events[mask]
        if self.is_train:
            fix_events_num = self.args.fix_events_num
        else:
            fix_events_num = self.args.val_fix_events_num
        start_index = max([events.shape[0] - fix_events_num, 0])
        events = events[start_index:]

        # events augment
        if self.is_train:
            events = events_augment(self.args, events, size=(self.args.ddd17_sensor_h, self.args.ddd17_sensor_w))
        else:
            if self.args.val_event_noise:
                events = events_augment(self.args, events, size=(self.args.ddd17_sensor_h, self.args.ddd17_sensor_w))

        if self.args.num_bins == 2:
            events_voxel_grid = events_to_image_ecdp(self.args, events,
                                                     size=(self.args.ddd17_sensor_h, self.args.ddd17_sensor_w))
        elif self.args.num_bins == 3:
            events_voxel_grid = events_to_image_mem(self.args, events, size=(self.args.ddd17_sensor_h, self.args.ddd17_sensor_w))
            events_voxel_grid = events_voxel_grid / 255  # Transforms.ToTensor()
            events_voxel_grid = remove_hot_pixel_mem(events_voxel_grid)
        else:
            if self.args.use_evrepsl:
                events_voxel_grid = events_to_EvRep(events[:, 0].astype(np.int16), events[:, 1].astype(np.int16),
                                               events[:, 2] / 1e6, events[:, 3], (self.args.ddd17_sensor_w, self.args.ddd17_sensor_h))

                events_voxel_grid = torch.from_numpy(events_voxel_grid).to(torch.float32)

            else:
                events_voxel_grid = events_to_voxel_grid(self.args, events, size=(self.args.ddd17_sensor_h, self.args.ddd17_sensor_w))

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


        # label
        label_path = os.path.join(self.dataset_root, self.seq_name, "segmentation_masks", self.label_files[index])
        semseg_label = self.get_label(label_path)
        if self.is_train:
            semseg_label = semseg_label_augment(self.args, semseg_label.float(),
                                                (self.args.ddd17_sensor_h, self.args.ddd17_sensor_w), seed=seed)
        # label = torch.einsum('chw->hwc', semseg_label)
        # plt.imshow(label, cmap='viridis')
        # plt.axis('off')
        # plt.show()

        data = {
            "events_voxel_grid": events_voxel_grid,  # (5,224,224)
            "semseg_label": semseg_label,
            "seq_name": self.seq_name
        }

        return data

    def __len__(self):
        return len(self.label_files)


class FinetuneDDD17TestDataset(Dataset):
    def __init__(self, args):
        val_seq = []
        val_seq_name_list = ['dir1']
        for val_seq_name in val_seq_name_list:
            val_seq.append(FinetuneDDD17SeqTestDataset(args, False, val_seq_name))
        self.val_dataset = torch.utils.data.ConcatDataset(val_seq)

    def get_val_dataset(self):
        return self.val_dataset


class FinetuneDDD17SeqTestDataset(FinetuneDDD17SeqDataset):
    def __getitem__(self, index):
        img_index = int(self.label_files[index][:-4].split("_")[-1]) - 1

        # events has form x, y, t_ns, p (in [0,1])
        events = self.extract_events_from_memmap(img_index)
        x = events[:, 0]
        y = events[:, 1]
        mask = (x >= 0) & (x < self.args.ddd17_sensor_w) & \
               (y >= 0) & (y < self.args.ddd17_sensor_h)
        events = events[mask]
        if self.is_train:
            fix_events_num = self.args.fix_events_num
        else:
            fix_events_num = self.args.val_fix_events_num
        start_index = max([events.shape[0] - fix_events_num, 0])
        events = events[start_index:]

        # label
        label_path = os.path.join(self.dataset_root, self.seq_name, "segmentation_masks", self.label_files[index])
        semseg_label = self.get_label(label_path)

        data = {
            "events": events,  # (5,224,224)
            "semseg_label": semseg_label,
            "seq_name": self.seq_name
        }

        return data

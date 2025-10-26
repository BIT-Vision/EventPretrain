import os
import re
import numpy as np

from torch.utils.data import Dataset

from dataset.dataset_utils.events_to_voxel_grid import events_to_voxel_grid
from dataset.dataset_utils.events_to_image import events_to_image_ecdp, events_to_image_mem, remove_hot_pixel_mem
from dataset.augmentation.events_augment import get_random_index, events_augment, events_reshape
from dataset.augmentation.view_augment import evg_augment, view_resize


class FinetuneCIFAR10DVSDataset(Dataset):
    def __init__(self, args, is_train=True):
        self.args = args

        self.is_train = is_train
        if is_train:
            self.cifar10_dvs_root = args.cifar10_dvs_train_root
        else:
            self.cifar10_dvs_root = args.cifar10_dvs_val_root

        self.class_dir_list = sorted(os.listdir(self.cifar10_dvs_root))
        assert len(self.class_dir_list) == args.num_classes

        self.events_file_list = []
        for class_dir in self.class_dir_list:
            events_file_list_per_class = sorted(os.listdir(os.path.join(self.cifar10_dvs_root, class_dir)))
            for events_file in events_file_list_per_class:
                self.events_file_list.append(events_file)

    def augment_parser(self, parser):
        def new_parser(x):
            return parser(x)

        return new_parser

    def load_events(self, events_file_name):
        image_class = re.split('_', events_file_name)[1]
        image_class_dir_path = os.path.join(self.cifar10_dvs_root, image_class)

        events = np.load(os.path.join(image_class_dir_path, events_file_name))

        return events

    def get_label(self, events_file_name):
        image_class = re.split('_', events_file_name)[1]
        label = self.class_dir_list.index(image_class)

        return label

    def __getitem__(self, index):
        events_file_name = self.events_file_list[index]
        image_name = events_file_name[:-4]

        # events
        events_parser = self.augment_parser(self.load_events)
        events = events_parser(events_file_name)
        start_index, end_index = get_random_index(self.args, events, self.is_train)
        events = events[start_index: end_index]

        # events augment
        if self.is_train:
            events = events_augment(self.args, events, size=(self.args.cifar_sensor_h, self.args.cifar_sensor_w))
        else:
            if self.args.val_event_noise:
                events = events_augment(self.args, events, size=(self.args.cifar_sensor_h, self.args.cifar_sensor_w))

        if self.args.num_bins == 2:
            events = events_reshape(events, self.args.cifar_sensor_w, self.args.cifar_sensor_h,
                                    self.args.input_size, self.args.input_size)
            events_voxel_grid = events_to_image_ecdp(self.args, events, size=(self.args.input_size, self.args.input_size))
        elif self.args.num_bins == 3:
            events_voxel_grid = events_to_image_mem(self.args, events, size=(self.args.cifar_sensor_h, self.args.cifar_sensor_w))
            events_voxel_grid = events_voxel_grid / 255  # Transforms.ToTensor()
            events_voxel_grid = remove_hot_pixel_mem(events_voxel_grid)
        else:
            events_voxel_grid = events_to_voxel_grid(self.args, events, size=(self.args.cifar_sensor_h, self.args.cifar_sensor_w))

        # view augment
        if self.is_train:
            events_voxel_grid, _ = evg_augment(self.args, events_voxel_grid,
                                               mode=self.args.resize_mode, size=(self.args.input_size, self.args.input_size))
        else:
            events_voxel_grid = view_resize(events_voxel_grid, (self.args.input_size, self.args.input_size), self.args.resize_mode)

        if self.args.num_bins == 2:
            events_voxel_grid = events_voxel_grid / (events_voxel_grid.amax([1, 2], True) + 1)
            events_voxel_grid = (events_voxel_grid - 0.5) * 2
        elif self.args.num_bins == 3:
            factor = 1.0 / events_voxel_grid[0::2, :, :].max()
            events_voxel_grid[0::2, :, :] = events_voxel_grid[0::2, :, :] * factor

        # label
        label_parser = self.augment_parser(self.get_label)
        label = label_parser(events_file_name)

        data = {
            "events_voxel_grid": events_voxel_grid,
            "label": label,
            "image_name": image_name,
        }

        return data

    def __len__(self):
        return len(self.events_file_list)

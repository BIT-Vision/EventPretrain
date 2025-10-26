import os
import re
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import Dataset

from dataset.dataset_utils.events_to_voxel_grid import events_to_voxel_grid
from dataset.dataset_utils.events_to_image import events_to_image_ecdp, events_to_image_mem, remove_hot_pixel_mem
from dataset.augmentation.events_augment import get_random_index, events_augment
from dataset.augmentation.view_augment import evg_augment, view_resize
from visualize.visualize_utils.make_events_preview import make_events_preview


class FinetuneNCarsDataset(Dataset):
    def __init__(self, args, is_train=True):
        self.args = args

        self.is_train = is_train
        if is_train:
            self.n_cars_root = args.n_cars_train_root
        else:
            self.n_cars_root = args.n_cars_val_root

        self.class_dir_list = sorted(os.listdir(self.n_cars_root))
        assert len(self.class_dir_list) == args.num_classes

        self.events_file_list = []
        for class_dir in self.class_dir_list:
            events_file_list_per_class = sorted(os.listdir(os.path.join(self.n_cars_root, class_dir)))
            for events_file_grid in events_file_list_per_class:
                self.events_file_list.append(events_file_grid)

    def augment_parser(self, parser):
        def new_parser(x):
            return parser(x)

        return new_parser

    def load_events(self, events_file_name):
        image_class = re.split('_', events_file_name)[0]
        image_class_dir_path = os.path.join(self.n_cars_root, image_class)

        events = np.load(os.path.join(image_class_dir_path, events_file_name))

        return events

    def get_label(self, events_file_name):
        image_class = re.split('_', events_file_name)[0]
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

        cars_sensor_h, cars_sensor_w = int(events[:, 1].max()) + 1, int(events[:, 0].max()) + 1
        # events augment
        if self.is_train:
            events = events_augment(self.args, events, size=(cars_sensor_h, cars_sensor_w))
        else:
            if self.args.val_event_noise:
                events = events_augment(self.args, events, size=(cars_sensor_h, cars_sensor_w))

        if self.args.num_bins == 2:
            events_voxel_grid = events_to_image_ecdp(self.args, events, size=(cars_sensor_h, cars_sensor_w))
        elif self.args.num_bins == 3:
            events_voxel_grid = events_to_image_mem(self.args, events, size=(cars_sensor_h, cars_sensor_w))
            events_voxel_grid = events_voxel_grid / 255  # Transforms.ToTensor()
            events_voxel_grid = remove_hot_pixel_mem(events_voxel_grid)

        else:
            events_voxel_grid = events_to_voxel_grid(self.args, events, size=(cars_sensor_h, cars_sensor_w))

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
            if events_voxel_grid[0::2, :, :].max() > 0:
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

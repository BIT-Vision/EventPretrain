import os
import re
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset

from dataset.dataset_utils.events_to_voxel_grid import events_to_voxel_grid
from dataset.dataset_utils.events_to_image import events_to_image_ecdp, events_to_image_mem, remove_hot_pixel_mem
from dataset.augmentation.events_augment import get_random_index, events_augment
from dataset.augmentation.view_augment import evg_augment, view_resize
from visualize.visualize_utils.make_events_preview import make_events_preview, make_events_preview_norm


class FinetuneESImageNetDataset(Dataset):
    def __init__(self, args, is_train=True):
        self.args = args

        self.is_train = is_train
        if is_train:
            self.es_imagenet_root = args.es_imagenet_train_root
        else:
            self.es_imagenet_root = args.es_imagenet_val_root

        self.class_dir_list = sorted(os.listdir(self.es_imagenet_root))[:args.num_classes]
        assert len(self.class_dir_list) == args.num_classes

        self.events_file_list = []
        for class_dir in self.class_dir_list:
            events_file_list_pre_class = sorted(os.listdir(os.path.join(self.es_imagenet_root, class_dir)))
            for events_file_grid in events_file_list_pre_class:
                self.events_file_list.append(events_file_grid)

        self.load_offset()

    def load_offset(self):
        if self.is_train:
            label_path = self.args.es_imagenet_train_label_path
        else:
            label_path = self.args.es_imagenet_val_label_path

        self.filename_ab_dic = {}
        with open(label_path, 'r') as file:
            for line in file:
                file_name = re.split(' ', line)[0]
                a = int(re.split(' ', line)[1])
                b = int(re.split(' ', line)[2])
                self.filename_ab_dic[file_name] = {}
                self.filename_ab_dic[file_name]['a'] = a
                self.filename_ab_dic[file_name]['b'] = b

    def augment_parser(self, parser):
        def new_parser(x):
            return parser(x)

        return new_parser

    def load_events(self, events_file_name):
        image_class = re.split('_', events_file_name)[0]
        image_class_dir_path = os.path.join(self.es_imagenet_root, image_class)

        events_file = np.load(os.path.join(image_class_dir_path, events_file_name))
        events_pos = events_file['pos']  # (15395,3) (x,y,t)
        events_pos = np.concatenate((events_pos, np.ones((events_pos.shape[0], 1))), axis=-1)
        events_neg = events_file['neg']  # (15698,3) (x,y,t)
        events_neg = np.concatenate((events_neg, np.zeros((events_neg.shape[0], 1))), axis=-1)
        events = np.concatenate((events_pos, events_neg), axis=0)
        events = events[events[:, 2].argsort()]

        a, b = self.filename_ab_dic[events_file_name]['a'], self.filename_ab_dic[events_file_name]['b']
        dx = (254 - a) // 2
        dy = (254 - b) // 2

        y = events[:, 0] + dx
        x = events[:, 1] + dy
        t = events[:, 2] - 1
        p = events[:, 3]

        mask = (x >= 16) & (x < 240) & (y >= 16) & (y < 240)
        x = x[mask] - 16
        y = y[mask] - 16
        t = t[mask]
        p = p[mask]

        events = np.stack([x, y, t, p], axis=-1)

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
        # events augment
        if self.is_train:
            events = events_augment(self.args, events, size=(self.args.esimg_sensor_h, self.args.esimg_sensor_w))


        if self.args.num_bins == 2:
            events_voxel_grid = events_to_image_ecdp(self.args, events, size=(self.args.input_size, self.args.input_size))
        elif self.args.num_bins == 3:
            events_voxel_grid = events_to_image_mem(self.args, events, size=(self.args.esimg_sensor_h, self.args.esimg_sensor_w))
            events_voxel_grid = events_voxel_grid / 255  # Transforms.ToTensor()
            events_voxel_grid = remove_hot_pixel_mem(events_voxel_grid)
        else:
            events_voxel_grid = events_to_voxel_grid(self.args, events, size=(self.args.esimg_sensor_h, self.args.esimg_sensor_w))

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

        # events_frame = make_events_preview(events_voxel_grid)
        # plt.imshow(events_frame, cmap='gray')
        # plt.axis('off')
        # plt.show()
        # events_frame = make_events_preview_norm(events_voxel_grid)
        # plt.imshow(events_frame, cmap='gray')
        # plt.axis('off')
        # plt.show()

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

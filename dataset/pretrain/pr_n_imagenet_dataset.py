import numpy as np
import os
import re
from PIL import Image
import clip
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

from dataset.dataset_utils.events_to_voxel_grid import events_to_voxel_grid
from dataset.dataset_utils.events_to_image import events_to_image_ecdp
from dataset.augmentation.events_augment import get_random_index, events_augment, events_reshape
from dataset.augmentation.view_augment import evg_augment
from visualize.visualize_utils.make_events_preview import make_events_preview


class NImageNetDataset(Dataset):
    def __init__(self, args):
        self.args = args

        self.n_imagenet_root = args.n_imagenet_train_root

        self.n_imagenet_class_dir_list = sorted(os.listdir(self.n_imagenet_root))[:args.num_classes]
        assert len(self.n_imagenet_class_dir_list) == args.num_classes

        self.n_imagenet_events_file_list = []
        for class_dir in self.n_imagenet_class_dir_list:
            events_file_list_pre_class = sorted(os.listdir(os.path.join(self.n_imagenet_root, class_dir)))
            for events_file_grid in events_file_list_pre_class:
                self.n_imagenet_events_file_list.append(events_file_grid)

    def augment_parser_1(self, parser):
        def new_parser(x):
            return parser(x)

        return new_parser

    def augment_parser_2(self, parser):
        def new_parser(x, y):
            return parser(x, y)

        return new_parser

    def load_events(self, image_name):
        image_class = re.split('_', image_name)[0]
        image_class_dir_path = os.path.join(self.n_imagenet_root, image_class)

        events = np.load(os.path.join(image_class_dir_path, image_name + ".npz"))

        events = events['event_data']
        events = np.vstack([events['x'], events['y'], events['t'], events['p']]).T
        events = events.astype(np.float64)
        events[:, 2] = events[:, 2] / 1e6

        return events

    def __len__(self):
        return len(self.n_imagenet_events_file_list)


class PretrainNImageNetDataset(NImageNetDataset):
    def __init__(self, args):
        super().__init__(args)
        # imagenet_clip_emb
        self.imagenet_root = args.imagenet_root
        _, self.preprocess = clip.load("ViT-B/16", device='cpu')

    def load_image(self, image_name):
        image_class = re.split(r'_', image_name)[0]
        image_path = os.path.join(self.imagenet_root, image_class, image_name + ".JPEG")
        image_preprocess = self.preprocess(Image.open(image_path))

        return image_preprocess

    def __getitem__(self, index):
        image_name = self.n_imagenet_events_file_list[index][:-4]

        # events voxel grid
        events_parser = self.augment_parser_1(self.load_events)
        events = events_parser(image_name)
        start_index, end_index = get_random_index(self.args, events, is_train=True)
        events = events[start_index: end_index]
        events = events_augment(self.args, events, size=(self.args.img_sensor_h, self.args.img_sensor_w))
        events = events_reshape(events, self.args.img_sensor_w, self.args.img_sensor_h,
                                self.args.input_size, self.args.input_size)
        events_voxel_grid = events_to_voxel_grid(self.args, events, size=(self.args.input_size, self.args.input_size))
        events_voxel_grid, _ = evg_augment(self.args, events_voxel_grid,
                                           size=(self.args.input_size, self.args.input_size))

        # _events_voxel_grid = copy.deepcopy(events_voxel_grid)
        # events_frame = make_events_preview(_events_voxel_grid)
        # plt.imshow(events_frame, cmap='gray')
        # plt.axis('off')
        # plt.show()

        # imagenet_image
        images_parser = self.augment_parser_1(self.load_image)
        image_preprocess = images_parser(image_name)

        data = {
            "events_voxel_grid": events_voxel_grid,  # (5,224,224)
            "image": image_preprocess,
            "image_name": image_name,
        }

        return data

class PretrainECDPNImageNetDataset(NImageNetDataset):
    def __init__(self, args):
        super().__init__(args)
        # imagenet_clip_emb
        self.imagenet_root = args.imagenet_root
        _, self.preprocess = clip.load("ViT-B/16", device='cpu')

    def load_clip_emb(self, image_name):
        image_class = re.split(r'_', image_name)[0]
        image_dir_path = os.path.join(self.imagenet_clip_emb_root, image_class, image_name)
        clip_emb_file_name = image_name + "_clip_emb.pt"

        clip_emb = torch.load(os.path.join(image_dir_path, clip_emb_file_name)).float().squeeze()

        return clip_emb

    def __getitem__(self, index):
        image_name = self.n_imagenet_events_file_list[index][:-4]

        # events
        events_parser = self.augment_parser_1(self.load_events)
        events = events_parser(image_name)

        # events image q
        seed_q = np.random.randint(1000)
        start_index, end_index = get_random_index(self.args, events, is_train=True, seed=seed_q)
        events_q = events[start_index: end_index]
        events_q = events_augment(self.args, events_q, seed=seed_q, size=(self.args.img_sensor_h, self.args.img_sensor_w))
        events_q = events_reshape(events_q, self.args.img_sensor_w, self.args.img_sensor_h,
                                  self.args.input_size, self.args.input_size)

        events_image_q = events_to_image_ecdp(self.args, events_q, size=(self.args.input_size, self.args.input_size))
        events_image_q, _ = evg_augment(self.args, events_image_q, size=(self.args.input_size, self.args.input_size), seed=seed_q)
        events_image_q = events_image_q / (events_image_q.amax([1, 2], True) + 1)
        events_image_q = (events_image_q - 0.5) * 2

        # _events_image_q = copy.deepcopy(events_image_q)
        # _events_image_q = make_events_preview(_events_image_q)
        # plt.imshow(_events_image_q, cmap='gray')
        # plt.axis('off')
        # plt.show()

        # events image k
        seed_k = np.random.randint(1000)
        start_index, end_index = get_random_index(self.args, events, is_train=True, seed=seed_k)
        events_k = events[start_index: end_index]
        events_k = events_augment(self.args, events_k, seed=seed_k, size=(self.args.img_sensor_h, self.args.img_sensor_w))
        events_k = events_reshape(events_k, self.args.img_sensor_w, self.args.img_sensor_h,
                                  self.args.input_size, self.args.input_size)

        events_image_k = events_to_image_ecdp(self.args, events_k, size=(self.args.input_size, self.args.input_size))
        events_image_k, _ = evg_augment(self.args, events_image_k, size=(self.args.input_size, self.args.input_size), seed=seed_k)
        events_image_k = events_image_k / (events_image_k.amax([1, 2], True) + 1)
        events_image_k = (events_image_k - 0.5) * 2

        # _events_image_k = copy.deepcopy(events_image_k)
        # _events_image_k = make_events_preview(_events_image_k)
        # plt.imshow(_events_image_k, cmap='gray')
        # plt.axis('off')
        # plt.show()

        # clip_emb
        embs_parser = self.augment_parser_1(self.load_clip_emb)
        clip_emb = embs_parser(image_name)

        data = {
            "events_image_q": events_image_q,  # (2,224,224)
            "events_image_k": events_image_k,  # (2,224,224)
            "clip_emb": clip_emb,
            "image_name": image_name,
        }

        return data

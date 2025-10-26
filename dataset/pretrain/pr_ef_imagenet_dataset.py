import numpy as np
import os
import re
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import copy

import torch
from torch.utils.data import Dataset

from dataset.augmentation.view_augment import evg_augment, frame_augment
from visualize.visualize_utils.make_events_preview import make_events_preview


class EForgNImageNetDataset(Dataset):
    def __init__(self, args, frame_index):
        self.args = args
        self.frame_index = frame_index

        self.ef_imagenet_org_root = args.ef_imagenet_org_train_root
        self.n_imagenet_root = args.n_imagenet_train_root

        self.image_name_list = []
        self.class_dir_list = sorted(os.listdir(self.ef_imagenet_org_root))

        for class_dir in self.class_dir_list:
            image_dir_list = sorted(os.listdir(os.path.join(self.ef_imagenet_org_root, class_dir)))
            for image_dir in image_dir_list:
                self.image_name_list.append(image_dir)

    def augment_parser_1(self, parser):
        def new_parser(x):
            return parser(x)

        return new_parser

    def augment_parser_2(self, parser):
        def new_parser(x, y):
            return parser(x, y)

        return new_parser

    def get_index(self, events):
        video_fps = 30
        t = 1 / video_fps
        frames_num = int(events[-1][0] // t)

        index_list = []
        index_list.append(0)
        for i in range(1, frames_num + 1):
            index = np.searchsorted(events[:, 0], t * i).item()
            index_list.append(index)

        return frames_num, index_list

    def load_ef_events(self, image_name, frame_index):
        image_class = re.split(r'_', image_name)[0]
        event_file_path = os.path.join(self.ef_imagenet_org_root, image_class, image_name,
                                       "events/noisy", image_name + "_noisy_events.txt")
        events = pd.read_csv(event_file_path, skiprows=6, sep=" ", header=None, names=['t', 'x', 'y', 'p'])
        events = np.array(events)
        frames_num, index_list = self.get_index(events)

        events = events[index_list[frame_index + 1]: index_list[frame_index + 2]]

        return events

    def load_ef_frame(self, image_name, frame_index):
        image_class = re.split(r'_', image_name)[0]
        frame_index_str = '0' + str(frame_index + 1) if frame_index + 1 < 10 else str(frame_index + 1)
        frame_file_path = os.path.join(self.ef_imagenet_org_root, image_class, image_name,
                                       "frames", image_name + "_" + frame_index_str + ".png")
        frame = Image.open(frame_file_path)
        frame = np.array(frame) / 255.

        return frame

    def load_n_events(self, image_name):
        image_class = re.split('_', image_name)[0]
        image_class_dir_path = os.path.join(self.n_imagenet_root, image_class)

        events = np.load(os.path.join(image_class_dir_path, image_name + ".npz"))
        events = events['event_data']
        events = np.vstack([events['x'], events['y'], events['t'], events['p']]).T
        events = events.astype(np.float64)
        events[:, 2] = events[:, 2] / 1e6

        return events[:30000]

    def __getitem__(self, index):
        image_name = self.image_name_list[index]
        # image_name = 'n02006656_10106'
        frame_index = self.frame_index

        # ef_events
        ef_events_parser = self.augment_parser_2(self.load_ef_events)
        ef_events = ef_events_parser(image_name, frame_index)

        # frame
        frame_parser = self.augment_parser_2(self.load_ef_frame)
        frame = frame_parser(image_name, frame_index)

        # n_events
        n_events_parser = self.augment_parser_1(self.load_n_events)
        n_events = n_events_parser(image_name)

        data = {
            "ef_events": ef_events,
            "n_events": n_events,
            "frame": frame,
            "image_name": image_name,
        }

        return data

    def __len__(self):
        return len(self.image_name_list)

class EFImageNetDataset(Dataset):
    def __init__(self, args):
        self.args = args

        self.imagenet_root = args.ef_imagenet_train_root

        self.image_name_list = []
        self.class_dir_list = sorted(os.listdir(self.imagenet_root))[:args.num_classes]
        assert len(self.class_dir_list) == args.num_classes

        for class_dir in self.class_dir_list:
            image_dir_list = sorted(os.listdir(os.path.join(self.imagenet_root, class_dir)))
            for image_dir in image_dir_list:
                self.image_name_list.append(image_dir)

    def augment_parser_1(self, parser):
        def new_parser(x):
            return parser(x)

        return new_parser

    def augment_parser_2(self, parser):
        def new_parser(x, y):
            return parser(x, y)

        return new_parser

    def load_events_voxel_grid(self, image_name, frame_index):
        image_class = re.split(r'_', image_name)[0]
        if frame_index < 10:
            noisy_events_dir_path = os.path.join(self.imagenet_root, image_class, image_name,
                                                 self.args.noisy_events_dir)
            noisy_events_file_name = image_name + "_0" + str(frame_index) + "_noisy_events_voxel_grid.pt"
            event_voxel_grid = torch.load(os.path.join(noisy_events_dir_path, noisy_events_file_name))
        else:
            clean_events_dir_path = os.path.join(self.imagenet_root, image_class, image_name,
                                                 self.args.clean_events_dir)
            clean_events_file_name = image_name + "_0" + str(frame_index - 6) + "_clean_events_voxel_grid.pt"
            event_voxel_grid = torch.load(os.path.join(clean_events_dir_path, clean_events_file_name))

        return event_voxel_grid

    def __len__(self):
        return len(self.image_name_list)


class PretrainEFImageNetDataset(EFImageNetDataset):
    def load_sub_frame(self, image_name, frame_index):
        image_class = re.split(r'_', image_name)[0]
        sub_frames_dir_path = os.path.join(self.imagenet_root, image_class, image_name, self.args.sub_frames_dir)
        sub_frame_file_name = image_name + "_0" + str(frame_index) + "_sub_frame.pt"
        sub_frame = torch.load(os.path.join(sub_frames_dir_path, sub_frame_file_name))

        return sub_frame

    def load_clip_emb(self, image_name):
        image_class = re.split(r'_', image_name)[0]
        image_dir_path = os.path.join(self.imagenet_root, image_class, image_name)
        clip_emb_file_name = image_name + "_clip_emb.pt"

        clip_emb = torch.load(os.path.join(image_dir_path, clip_emb_file_name)).float().squeeze()

        return clip_emb

    def __getitem__(self, index):
        image_name = self.image_name_list[index]
        frame_index = np.random.randint(0, 10)
        seed = np.random.randint(1000)

        # events voxel grid
        events_parser = self.augment_parser_2(self.load_events_voxel_grid)
        events_voxel_grid = events_parser(image_name, frame_index)
        if self.args.num_bins == 1:
            events_voxel_grid = events_voxel_grid.sum(dim=0)[None, :, :]
        events_voxel_grid, time_flip_flag = evg_augment(self.args, events_voxel_grid,
                                                        size=(self.args.input_size, self.args.input_size), seed=seed)

        # events_frame = make_events_preview(events_voxel_grid)
        # plt.imshow(events_frame, cmap='gray')
        # plt.axis('off')
        # plt.show()

        if self.args.pr_phase == "rec":
            # sub frame
            frames_parser = self.augment_parser_2(self.load_sub_frame)
            sub_frame = frames_parser(image_name, frame_index)
            sub_frame = frame_augment(self.args, sub_frame, seed=seed, time_flip_flag=time_flip_flag)

            data = {
                "events_voxel_grid": events_voxel_grid,  # (5,224,224)
                "sub_frame": sub_frame,  # (1,224,224)
                "image_name": image_name,
            }
        elif self.args.pr_phase == "adj" or self.args.pr_phase == "_adj" or self.args.pr_phase == "con":
            # clip_emb
            frames_parser = self.augment_parser_1(self.load_clip_emb)
            clip_emb = frames_parser(image_name)

            data = {
                "events_voxel_grid": events_voxel_grid,  # (5,224,224)
                "clip_emb": clip_emb,  # (197, 512)
                "image_name": image_name,
            }
        else:
            # sub frame
            frames_parser = self.augment_parser_2(self.load_sub_frame)
            sub_frame = frames_parser(image_name, frame_index)
            sub_frame = frame_augment(self.args, sub_frame, seed=seed, time_flip_flag=time_flip_flag)

            # clip_emb
            frames_parser = self.augment_parser_1(self.load_clip_emb)
            clip_emb = frames_parser(image_name)

            data = {
                "events_voxel_grid": events_voxel_grid,  # (5,224,224)
                "sub_frame": sub_frame,  # (1,224,224)
                "clip_emb": clip_emb,  # (197, 512)
                "image_name": image_name,
            }

        return data

    def __len__(self):
        return len(self.image_name_list)


class PretrainECDPEFImageNetDataset(EFImageNetDataset):
    def load_clip_emb(self, image_name):
        image_class = re.split(r'_', image_name)[0]
        image_dir_path = os.path.join(self.imagenet_root, image_class, image_name)
        clip_emb_file_name = image_name + "_clip_emb.pt"

        clip_emb = torch.load(os.path.join(image_dir_path, clip_emb_file_name)).float().squeeze()

        return clip_emb

    def __getitem__(self, index):
        image_name = self.image_name_list[index]

        # events voxel grid q
        seed_q = np.random.randint(1000)
        frame_index_q = np.random.randint(0, 10)
        events_parser = self.augment_parser_2(self.load_events_voxel_grid)
        events_voxel_grid_q = events_parser(image_name, frame_index_q)
        events_voxel_grid_q, _ = evg_augment(self.args, events_voxel_grid_q,
                                                        size=(self.args.input_size, self.args.input_size), seed=seed_q)

        # _events_voxel_grid_q = copy.deepcopy(events_voxel_grid_q)
        # _events_voxel_grid_q = make_events_preview(_events_voxel_grid_q)
        # plt.imshow(_events_voxel_grid_q, cmap='gray')
        # plt.axis('off')
        # plt.show()

        # events voxel grid k
        seed_k = np.random.randint(1000)
        frame_index_k = np.random.randint(0, 10)
        events_parser = self.augment_parser_2(self.load_events_voxel_grid)
        events_voxel_grid_k = events_parser(image_name, frame_index_k)
        events_voxel_grid_k, _ = evg_augment(self.args, events_voxel_grid_k,
                                             size=(self.args.input_size, self.args.input_size), seed=seed_k)

        # _events_voxel_grid_k = copy.deepcopy(events_voxel_grid_k)
        # _events_voxel_grid_k = make_events_preview(_events_voxel_grid_k)
        # plt.imshow(_events_voxel_grid_k, cmap='gray')
        # plt.axis('off')
        # plt.show()

        # clip_emb
        frames_parser = self.augment_parser_1(self.load_clip_emb)
        clip_emb = frames_parser(image_name)

        data = {
            "events_image_q": events_voxel_grid_q,  # (2,224,224)
            "events_image_k": events_voxel_grid_k,  # (2,224,224)
            "clip_emb": clip_emb,  # (197, 512)
            "image_name": image_name,
        }

        return data

class PretrainEFImageNetTestDataset(PretrainEFImageNetDataset):
    def __getitem__(self, index):
        image_name = self.image_name_list[index]
        frame_index = 7

        # events voxel grid
        events_parser = self.augment_parser_2(self.load_events_voxel_grid)
        events_voxel_grid = events_parser(image_name, frame_index)

        if self.args.pr_phase == "rec":
            # sub frame
            frames_parser = self.augment_parser_2(self.load_sub_frame)
            sub_frame = frames_parser(image_name, frame_index)

            data = {
                "events_voxel_grid": events_voxel_grid,  # (5,224,224)
                "sub_frame": sub_frame,  # (1,224,224)
                "image_name": image_name,
            }

        elif self.args.pr_phase == "adj" or self.args.pr_phase == "_adj" or self.args.pr_phase == "con":
            # clip_emb
            frames_parser = self.augment_parser_1(self.load_clip_emb)
            clip_emb = frames_parser(image_name)

            data = {
                "events_voxel_grid": events_voxel_grid,  # (5,224,224)
                "clip_emb": clip_emb,  # (197, 512)
                "image_name": image_name,
            }
        else:
            raise ValueError

        return data

    def __len__(self):
        return len(self.image_name_list)
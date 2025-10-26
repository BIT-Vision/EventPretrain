import os
os.environ['CUDA_VISIBLE_DEVICES'] = '8'
import argparse
import numpy as np
import datetime
import time
import json
from pathlib import Path
from ptflops import get_model_complexity_info

import torch
torch.set_num_threads(1)
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group

import timm
assert timm.__version__ == "0.3.2"  # version check
import clip

from trainer.pretrain.pr_trainer import pr_rec_one_epoch, pr_con_one_epoch, pr_con_n_one_epoch, pr_rec_and_con_one_epoch
from trainer.pretrain.pr_ecdp_trainer import pr_ecdp_one_epoch
from model.pretrain import pr_hub_model, pr_ecdp_hub_model
import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
import utils.lr_decay as lrd
from dataset.pretrain.pr_ef_imagenet_dataset import PretrainEFImageNetDataset, PretrainECDPEFImageNetDataset
from dataset.pretrain.pr_n_imagenet_dataset import PretrainNImageNetDataset, PretrainECDPNImageNetDataset


def get_args_parser():
    parser = argparse.ArgumentParser()

    # Experiment parameters
    parser.add_argument('--pr_phase', default='rec', type=str)  # rec, adj, con, adj-n, _adj, con-n, ecdp, ecdp-ef
    parser.add_argument('--backbone_type', default='vit', type=str)  # vit, convvit, swin
    parser.add_argument('--patch_size', default=32, type=int)  # swin: 32 others:16

    parser.add_argument('--checkpoint_path',
                        default="",
                        type=str)
    parser.add_argument('--use_checkpoint', action='store_true')
    # parser.set_defaults(use_checkpoint=True)
    parser.add_argument('--use_layer_decay', action='store_true')
    # parser.set_defaults(use_layer_decay=True)
    parser.add_argument('--use_layer_grafted', action='store_true')
    # parser.set_defaults(use_layer_grafted=True)

    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='sub_module learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--warmup_epochs', default=20, type=int)
    parser.add_argument('--epochs', default=5, type=int)

    parser.add_argument('--masking_strategy', default='random')  # random density anti-density
    parser.add_argument('--crop_min', type=float, default=0.8,
                        help='view augmentation')
    parser.add_argument('--use_queue', action='store_true')
    parser.set_defaults(use_queue=True)
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--fix_events_num', default=15000, type=int)
    parser.add_argument('--n_imagenet_train_root', default='/home/gstack/Datasets/N-ImageNet/train', type=str)
    parser.add_argument('--imagenet_clip_emb_root',
                        default='/path/to/clip_emb', type=str)
    parser.add_argument('--imagenet_root',
                        default='/path/to/EF-ImageNet', type=str)

    parser.add_argument('--exp_name', default='test')

    # Device Change Parameters
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--world_size', default=2, type=int, help='number of distributed processes')
    parser.add_argument('--master_port', default='12345', type=str)

    parser.add_argument('--ef_imagenet_train_root', default='F:/EventPretrain/Datasets/EF-ImageNet', type=str)
    parser.add_argument('--output_root_path', default='results')

    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--persistent_workers', default=False, type=bool)

    parser.add_argument('--print_freq', default=1, type=int)
    parser.add_argument('--log_freq', default=10, type=int)
    parser.add_argument('--save_model_freq', default=100, type=int)
    parser.add_argument('--vis_train_freq', default=100, type=int)

    parser.add_argument('--test_experiment', action='store_true')
    parser.set_defaults(test_experiment=True)
    parser.add_argument('--distributed', default=False, type=bool)

    # Train parameters
    parser.add_argument('--phase', default="pretrain", type=str)
    parser.add_argument('--mask_ratio', default=0.50, type=float)
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--drop_rate', type=float, default=0.)
    parser.add_argument('--attn_drop_rate', type=float, default=0.)
    parser.add_argument('--drop_path_rate', type=float, default=0.)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--visualize', action='store_true')
    parser.set_defaults(visualize=True)
    parser.add_argument('--backward', action='store_true')
    parser.set_defaults(backward=True)

    # Model parameters
    parser.add_argument('--use_feature_fusion', action='store_true')
    parser.set_defaults(use_feature_fusion=True)
    parser.add_argument('--model_size', default="small", type=str)
    parser.add_argument('--num_bins', default=5, type=int)
    parser.add_argument('--frame_chans', default=1, type=int)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--T', default=0.07, type=float)
    parser.add_argument('--queue_length', default=1024, type=int)
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=True)

    # ECDP
    parser.add_argument('--ema_m', default=0.99, type=float)
    parser.add_argument('--T_image', default=0.1, type=float)
    parser.add_argument('--T_event', default=0.2, type=float)
    parser.add_argument('--lambda_image', default=1, type=int)
    parser.add_argument('--lambda_event', default=1, type=int)
    parser.add_argument('--lambda_kl', default=2, type=int)

    # Pretrain Dataset parameters
    parser.add_argument('--dataset_type', default='pr_n-imagenet', type=str)
    parser.add_argument('--noisy_events_dir', default='events/noisy', type=str)
    parser.add_argument('--clean_events_dir', default='events/clean', type=str)
    parser.add_argument('--sub_frames_dir', default='sub_frames', type=str)
    parser.add_argument('--frames_dir', default='frames', type=str)
    parser.add_argument('--emb_frames_dim', default=512, type=int)
    parser.add_argument('--img_sensor_w', default=640, type=int)
    parser.add_argument('--img_sensor_h', default=480, type=int)

    # Dataloader parameters
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # output directory parameters
    parser.add_argument('--rec_dir', default='1-rec')
    parser.add_argument('--adj_dir', default='2-adj')
    parser.add_argument('--adj_n_dir', default='2-adj_n')
    parser.add_argument('--_adj_dir', default='2-_adj')
    parser.add_argument('--con_dir', default='3-con')
    parser.add_argument('--con_n_dir', default='3-con_n')
    parser.add_argument('--rec_and_con_dir', default='rec+con')
    parser.add_argument('--ecdp_dir', default='ecdp')
    parser.add_argument('--ecdp_ef_dir', default='ecdp-ef')
    parser.add_argument('--log_dir', default='log',
                        help='path where to tensorboard log')
    parser.add_argument('--checkpoints_dir', default='checkpoints',
                        help='path where to save checkpoints')
    parser.add_argument('--vis_train_dir', default='vis_train')

    return parser


def main(args):
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = True

    args.output_dir = os.path.join(args.output_root_path, "pretrain", args.exp_name)

    # generate dataset and dataloader
    if args.pr_phase == "rec-n" or args.pr_phase == "con-n" or args.pr_phase == "adj-n":
        dataset_train = PretrainNImageNetDataset(args)
    elif args.pr_phase == "ecdp":
        dataset_train = PretrainECDPNImageNetDataset(args)
    elif args.pr_phase == "ecdp-ef":
        dataset_train = PretrainECDPEFImageNetDataset(args)
    else:
        dataset_train = PretrainEFImageNetDataset(args)

    if args.pr_phase == "rec":
        phase_dir = args.rec_dir
    elif args.pr_phase == "adj":
        phase_dir = args.adj_dir
    elif args.pr_phase == "adj-n":
        phase_dir = args.adj_n_dir
    elif args.pr_phase == "_adj":
        phase_dir = args._adj_dir
    elif args.pr_phase == "con":
        phase_dir = args.con_dir
    elif args.pr_phase == "con-n":
        phase_dir = args.con_n_dir
    elif args.pr_phase == "rec+con":
        phase_dir = args.rec_and_con_dir
    elif args.pr_phase == "ecdp":
        phase_dir = args.ecdp_dir
    elif args.pr_phase == "ecdp-ef":
        phase_dir = args.ecdp_ef_dir
    else:
        raise ValueError

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )

        if global_rank == 0 and args.log_dir is not None:
            Path(os.path.join(args.output_dir, phase_dir, args.log_dir)).mkdir(parents=True, exist_ok=True)
            log_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, phase_dir, args.log_dir))
        else:
            log_writer = None
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

        if args.log_dir is not None:
            Path(os.path.join(args.output_dir, phase_dir, args.log_dir)).mkdir(parents=True, exist_ok=True)
            log_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, phase_dir, args.log_dir))
        else:
            log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_mem,
        persistent_workers=args.persistent_workers,
        sampler=sampler_train)

    # define the model
    if args.pr_phase == "ecdp" or args.pr_phase == "ecdp-ef":
        model = pr_ecdp_hub_model.__dict__["pretrain_ecdp_model_small_patch16"](args=args,
                                                                                emb_frames_dim=args.emb_frames_dim,
                                                                                queue_length=args.queue_length,
                                                                                T_image=args.T_image,
                                                                                T_event=args.T_event)
    else:
        if args.backbone_type == "swin":
            model = pr_hub_model.__dict__["pretrain_hub_model_swin_tiny_patch16"](args=args,
                                                                              emb_frames_dim=args.emb_frames_dim,
                                                                              queue_length=args.queue_length,
                                                                              T=args.T)
        else:
            model = pr_hub_model.__dict__["pretrain_hub_model_small_patch16"](args=args,
                                                                              emb_frames_dim=args.emb_frames_dim,
                                                                              queue_length=args.queue_length,
                                                                              T=args.T)

    # load checkpoint
    if args.use_checkpoint:
        pretrain_checkpoint = torch.load(args.checkpoint_path, map_location='cpu')['model']
        if args.pr_phase == "rec" or args.pr_phase == "adj" or args.pr_phase == "_adj" or args.pr_phase == "adj-n":
            for k in list(pretrain_checkpoint.keys()):
                if 'norm_l_h' in k:  # norm_l_h -> norm_layer
                    pretrain_checkpoint[k[:13] + "_layer" + k[17:]] = pretrain_checkpoint[k]
                    del pretrain_checkpoint[k]
        if args.pr_phase == "con":
            for k in list(pretrain_checkpoint.keys()):
                if 'norm_h' in k:  # norm_h -> norm_layer
                    pretrain_checkpoint[k[:13] + "_layer" + k[17:]] = pretrain_checkpoint[k]
                    del pretrain_checkpoint[k]

        pretrain_checkpoint_msg = model.load_state_dict(pretrain_checkpoint, strict=False)
        print("load pretrain checkpoint msg: ", pretrain_checkpoint_msg)

    if args.pr_phase == "adj" or args.pr_phase == "adj-n":
        for k, v in model.backbone.named_parameters():
            if 'norm_layer' not in k:
                v.requires_grad = False

    device = torch.device(args.device)  #  rank
    model.to(device)

    # model parameters
    print("Model = %s" % str(model))
    backbone_parameters = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    print('number of backbone params (M): %.5f' % (backbone_parameters / 1.e6))
    model_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of model params (M): %.5f' % (model_parameters / 1.e6))

    # def input_constructor(input_res):
    #     events_voxel_grid = torch.randn((1, *input_res), device=args.device)
    #     if args.pr_phase == "rec":
    #         supp_data = torch.randn((1, 1, input_res[1], input_res[2]), device=args.device)
    #         return dict(events_voxel_grid=events_voxel_grid, supp_data=supp_data, is_rec=True)
    #     elif args.pr_phase == "adj" or args.pr_phase == "adj-n" or args.pr_phase == "_adj" or \
    #         args.pr_phase == "con" or args.pr_phase == "con-n":
    #         supp_data = torch.randn((1, 197, args.emb_frames_dim), device=args.device)
    #         return dict(events_voxel_grid=events_voxel_grid, supp_data=supp_data, is_rec=False)
    #     else:  # ecdp, ecdp-ef
    #         raise ValueError

    # model parameters and flops
    # model_copy = copy.deepcopy(model)
    # flops, params = get_model_complexity_info(model_copy,
    #                                           input_res=(args.num_bins, args.input_size, args.input_size),
    #                                           input_constructor=input_constructor,
    #                                           as_strings=True)
    # print('finetune model Params:', params)
    # print('finetune model FLOPs:', flops)

    model_without_ddp = model
    if args.distributed:
        model = DDP(model, device_ids=[device], find_unused_parameters=True)
        model_without_ddp = model.module

    # modify lr
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("effective batch size: %d" % eff_batch_size)
    print("actual lr: %.2e" % args.lr)

    # following timm: set wd as 0 for bias and norm layers
    if args.use_layer_decay:
        if args.use_layer_grafted:
            param_groups = lrd.param_groups_lrd(args, model_without_ddp, args.weight_decay,
                                                layer_decay=args.layer_decay, layer_grafted=True)
        else:
            param_groups = lrd.param_groups_lrd(args, model_without_ddp, args.weight_decay,
                                                layer_decay=args.layer_decay)
    else:
        param_groups = lrd.param_groups_lrd(args, model_without_ddp, args.weight_decay, layer_decay=1)

    if args.use_layer_grafted:
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    else:
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # trainer
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        # train
        if args.pr_phase == "rec" or args.pr_phase == "rec-n":
            train_stats = pr_rec_one_epoch(args, model, data_loader_train, optimizer, epoch, loss_scaler,
                                             log_writer=log_writer)
        elif args.pr_phase == "adj" or args.pr_phase == "_adj" or args.pr_phase == "con":
            train_stats = pr_con_one_epoch(args, model, data_loader_train, optimizer, epoch, loss_scaler,
                                             log_writer=log_writer)
        elif args.pr_phase == "adj-n" or args.pr_phase == "con-n":
            clip_model, preprocess = clip.load("ViT-B/16", device=device)
            train_stats = pr_con_n_one_epoch(args, model, preprocess, clip_model, data_loader_train, optimizer, epoch, loss_scaler,
                                             log_writer=log_writer)
        elif args.pr_phase == "rec+con":
            train_stats = pr_rec_and_con_one_epoch(args, model, data_loader_train, optimizer, epoch, loss_scaler,
                                                    log_writer=log_writer)
        elif args.pr_phase == "ecdp" or args.pr_phase == "ecdp-ef":
            train_stats = pr_ecdp_one_epoch(args, model, data_loader_train, optimizer, epoch, loss_scaler,
                                            log_writer=log_writer)
        else:
            raise ValueError

        if (epoch + 1) % args.save_model_freq == 0 or (epoch + 1) == args.epochs:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, phase_dir, args.log_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    # destroy_process_group()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    # mp.spawn(main, args=(args,), nprocs=args.world_size)
    main(args)

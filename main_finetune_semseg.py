import os
os.environ['CUDA_VISIBLE_DEVICES'] = '8'
import argparse
import numpy as np
import datetime
import time
import json
from pathlib import Path
# from ptflops import get_model_complexity_info

import torch
torch.set_num_threads(1)
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

import timm
assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

from trainer.finetune_semseg.ft_semseg_trainer import ft_semseg_train_one_epoch, ft_semseg_val
from model.finetune_dense import ft_dense_hub_model
from model.sub_module import evrepsl_net
import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
import utils.lr_decay as lrd
from dataset.finetune_semseg.ft_dsec_dataset import FinetuneDSECDataset
from dataset.finetune_semseg.ft_ddd17_dataset import FinetuneDDD17Dataset


def get_args_parser():
    parser = argparse.ArgumentParser()

    # Experiment parameters
    parser.add_argument('--backbone_type', default='vit', type=str)
    parser.add_argument('--patch_size', default=16, type=int)  # swin: 32 others:16
    parser.add_argument('--num_patches', default=196, type=int)  # swin: 49 others: 196

    parser.add_argument('--dataset_type', default='dsec', type=str)  # dsec, ddd17
    parser.add_argument('--num_classes', default=11, type=int)  # dsec: 11, ddd17: 6
    parser.add_argument('--fix_events_num', default=200000, type=int)  # dsec:200000, ddd17:80000
    parser.add_argument('--num_bins', default=5, type=int,
                        help='number of bins per voxel grid')
    parser.add_argument('--model_size', default='small', type=str)

    parser.add_argument('--val_fix_events_num', default=200000, type=int)

    parser.add_argument('--pretrain_model_checkpoint',
                        default="",
                        type=str)
    parser.add_argument('--mem_checkpoint', action='store_true')
    # parser.set_defaults(mem_checkpoint=True)
    parser.add_argument('--ecdp_checkpoint', action='store_true')
    # parser.set_defaults(ecdp_checkpoint=True)
    parser.add_argument('--ecddp_checkpoint', action='store_true')
    #parser.set_defaults(ecddp_checkpoint=True)
    parser.add_argument('--use_checkpoint', action='store_true')
    # parser.set_defaults(use_checkpoint=True)
    parser.add_argument('--use_evrepsl', action='store_true')
    # parser.set_defaults(use_evrepsl=True)
    parser.add_argument('--evrepsl_model_checkpoint_path',
                        default="/path/to/RepGen.pth",
                        type=str)

    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='sub_module learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--warmup_epochs', default=20, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    parser.add_argument('--exp_name', default='test', type=str)

    # Device Change Parameters
    parser.add_argument('--device', default='cpu', help='device to use for training / testing')
    parser.add_argument('--batch_size', default=2, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--persistent_workers', default=False, type=bool)

    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--log_freq', default=10, type=int)
    parser.add_argument('--save_model_freq', default=100, type=int)
    parser.add_argument('--vis_train_freq', default=100, type=int)
    parser.add_argument('--vis_val_freq', default=100, type=int)

    parser.add_argument('--test_experiment', action='store_true')
    parser.set_defaults(test_experiment=True)
    parser.add_argument('--distributed', default=False, type=bool)

    parser.add_argument('--output_root_path', default='results', type=str)

    # Dataset Parameters
    parser.add_argument('--dsec_train_root', default='/path/to/DSEC/train', type=str)
    parser.add_argument('--dsec_val_root', default='/path/to/DSEC/test', type=str)
    parser.add_argument('--ddd17_train_root', default='/path/to/DDD17/train', type=str)
    parser.add_argument('--ddd17_val_root', default='/path/to/DDD17/test', type=str)

    # Train parameters
    parser.add_argument('--phase', default="finetune_semseg", type=str)
    parser.add_argument('--sample_mode', default='bilinear', type=str)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay')
    parser.add_argument('--clip_grad', type=float, default=3,
                        help='Clip gradient norm')
    parser.add_argument('--use_layer_decay', action='store_true')
    # parser.set_defaults(use_layer_decay=True)
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--drop_rate', type=float, default=0.)
    parser.add_argument('--attn_drop_rate', type=float, default=0.)
    parser.add_argument('--drop_path_rate', type=float, default=0.1)
    parser.add_argument('--backward', action='store_true')
    parser.set_defaults(backward=True)

    # Model parameters
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--crop_min', type=float, default=0.8)
    parser.add_argument('--visualize', action='store_true')
    parser.set_defaults(visualize=True)
    parser.add_argument('--decode_loss_weight', default=1.0, type=float)
    parser.add_argument('--aux_loss_weight', default=0.4, type=float)

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    # Dataset parameters
    parser.add_argument('--ignore_label', default=255, type=int)

    # DSEC Dataset parameters
    parser.add_argument('--dsec_org_sensor_w', default=640, type=int)
    parser.add_argument('--dsec_org_sensor_h', default=480, type=int)
    parser.add_argument('--dsec_sensor_w', default=640, type=int)
    parser.add_argument('--dsec_sensor_h', default=440, type=int)

    # DDD17 Dataset parameters
    parser.add_argument('--ddd17_org_sensor_w', default=346, type=int)
    parser.add_argument('--ddd17_org_sensor_h', default=260, type=int)
    parser.add_argument('--ddd17_sensor_w', default=346, type=int)
    parser.add_argument('--ddd17_sensor_h', default=200, type=int)

    # Dataloader parameters
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # output directory parameters
    parser.add_argument('--log_dir', default='log',
                        help='path where to tensorboard log')
    parser.add_argument('--checkpoints_dir', default='checkpoints',
                        help='path where to save visualize figure')
    parser.add_argument('--vis_train_dir', default='vis_train',
                        help='path where to save visualize figure')
    parser.add_argument('--vis_val_dir', default='vis_val',
                        help='path where to save visualize figure')

    return parser


def main(args):
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = True

    args.output_dir = os.path.join(args.output_root_path, "finetune_semseg", args.dataset_type, args.exp_name)

    # generate dataset and dataloader
    if args.dataset_type == "dsec":
        dataset_train = FinetuneDSECDataset(args, is_train=True).get_train_dataset()
        dataset_val = FinetuneDSECDataset(args, is_train=False).get_val_dataset()
    elif args.dataset_type == "ddd17":
        dataset_train = FinetuneDDD17Dataset(args, is_train=True).get_train_dataset()
        dataset_val = FinetuneDDD17Dataset(args, is_train=False).get_val_dataset()
    else:
        raise ValueError

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

        if global_rank == 0 and args.log_dir is not None:
            Path(os.path.join(args.output_dir, args.log_dir)).mkdir(parents=True, exist_ok=True)
            log_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, args.log_dir))
        else:
            log_writer = None
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.RandomSampler(dataset_val)

        if args.log_dir is not None:
            Path(os.path.join(args.output_dir, args.log_dir)).mkdir(parents=True, exist_ok=True)
            log_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, args.log_dir))
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

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_mem,
        persistent_workers=args.persistent_workers,
        sampler=sampler_val)

    # define the model
    if args.model_size == "small":
        model = ft_dense_hub_model.__dict__["finetune_dense_hub_model_small_patch16"](args)
    else:  # "base"
        model = ft_dense_hub_model.__dict__["finetune_dense_hub_model_base_patch16"](args)

    # load pretrain model
    if args.use_checkpoint:
        if args.mem_checkpoint:
            pretrain_model_checkpoint = torch.load(args.pretrain_model_checkpoint, map_location='cpu')['model']
            for k in list(pretrain_model_checkpoint.keys()):
                pretrain_model_checkpoint['backbone.' + k] = pretrain_model_checkpoint[k]
                del pretrain_model_checkpoint[k]
            for k in list(pretrain_model_checkpoint.keys()):
                if k.startswith('backbone.blocks.'):
                    pretrain_model_checkpoint['backbone.vit_block.' + k[16:]] = pretrain_model_checkpoint[k]
                    del pretrain_model_checkpoint[k]
            for k in list(pretrain_model_checkpoint.keys()):
                if k.startswith('backbone.norm.'):
                    pretrain_model_checkpoint['backbone.norm_layer.' + k[14:]] = pretrain_model_checkpoint[k]
                    del pretrain_model_checkpoint[k]

            rel_pos_bias = pretrain_model_checkpoint["backbone.rel_pos_bias.relative_position_bias_table"]
            for i in range(12):
                pretrain_model_checkpoint["backbone.vit_block.%d.attn.relative_position_bias_table" % i] = rel_pos_bias.clone()
        elif args.ecdp_checkpoint:
            pretrain_model_checkpoint = torch.load(args.pretrain_model_checkpoint, map_location='cpu')['checkpoint']
            for k in list(pretrain_model_checkpoint.keys()):
                if k.startswith('encoder_k.'):
                    del pretrain_model_checkpoint[k]
            for k in list(pretrain_model_checkpoint.keys()):
                if k.startswith('encoder_q.'):
                    pretrain_model_checkpoint['backbone.' + k[10:]] = pretrain_model_checkpoint[k]
                    del pretrain_model_checkpoint[k]
            for k in list(pretrain_model_checkpoint.keys()):
                if k.startswith('backbone.blocks.'):
                    pretrain_model_checkpoint['backbone.vit_block.' + k[16:]] = pretrain_model_checkpoint[k]
                    del pretrain_model_checkpoint[k]
            for k in list(pretrain_model_checkpoint.keys()):
                if k.startswith('backbone.norm.'):
                    pretrain_model_checkpoint['backbone.norm_layer.' + k[14:]] = pretrain_model_checkpoint[k]
                    del pretrain_model_checkpoint[k]
        elif args.ecddp_checkpoint:
            pretrain_model_checkpoint = torch.load(args.pretrain_model_checkpoint, map_location='cpu')['model']
            for k in list(pretrain_model_checkpoint.keys()):
                if k.startswith('student.'):
                    pretrain_model_checkpoint[k[8:]] = pretrain_model_checkpoint[k]
                    del pretrain_model_checkpoint[k]
        else:
            pretrain_model_checkpoint = torch.load(args.pretrain_model_checkpoint, map_location='cpu')['model']
            for k in list(pretrain_model_checkpoint.keys()):
                if k.startswith('pretrain_encoder.'):
                    pretrain_model_checkpoint['backbone.' + k[17:]] = pretrain_model_checkpoint[k]
                    del pretrain_model_checkpoint[k]

        pretrain_model_msg = model.load_state_dict(pretrain_model_checkpoint, strict=False)
        print("load pretrain model msg: ", pretrain_model_msg)

    device = torch.device(args.device)  # rank
    model.to(device)

    # model parameters
    print("Model = %s" % str(model))
    bb_model_parameters = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    print('number of backbone params (M): %.2f' % (bb_model_parameters / 1.e6))
    ft_model_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of finetune model params (M): %.2f' % (ft_model_parameters / 1.e6))

    # model parameters and flops
    # model_copy = copy.deepcopy(model)
    # flops, params = get_model_complexity_info(model_copy, (args.num_bins, args.input_size, args.input_size), as_strings=True)
    # print('finetune model Params:', params)
    # print('finetune model FLOPs:', flops)

    model_without_ddp = model
    if args.distributed:
        model = DDP(model, device_ids=[device], find_unused_parameters=True)
        model_without_ddp = model.module

    if args.use_evrepsl:
        evrepsl_model = evrepsl_net.__dict__["evrepsl_net"]()
        evrepsl_model_checkpoint = torch.load(args.evrepsl_model_checkpoint_path, map_location='cpu')
        evrepsl_model_msg = evrepsl_model.load_state_dict(evrepsl_model_checkpoint, strict=False)
        print("load evrepsl model msg: ", evrepsl_model_msg)
        evrepsl_model.to(device)
    else:
        evrepsl_model = None

    # modify lr
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("effective batch size: %d" % eff_batch_size)
    print("actual lr: %.2e" % args.lr)

    # following timm: set wd as 0 for bias and norm layers
    if args.use_layer_decay:
        param_groups = lrd.param_groups_lrd(args, model_without_ddp, args.weight_decay, layer_decay=args.layer_decay)
    else:
        param_groups = lrd.param_groups_lrd(args, model_without_ddp, args.weight_decay, layer_decay=1)

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # trainer
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_miou = 0.0
    max_macc = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        # train
        # '''
        train_stats = ft_semseg_train_one_epoch(args, model, data_loader_train, optimizer, epoch, loss_scaler,
                                         log_writer=log_writer, evrepsl_model=evrepsl_model)

        if (epoch + 1) % args.save_model_freq == 0 or (epoch + 1) == args.epochs:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, args.log_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        # '''

        # val
        test_stats = ft_semseg_val(args, model, data_loader_val, epoch, evrepsl_model=evrepsl_model)
        print(f"Accuracy of the network on the {len(dataset_val)} test miou: {test_stats['miou']:.2f}%")
        if max_miou < test_stats["miou"]:
            max_miou = test_stats["miou"]
            max_macc = test_stats["macc"]
            misc.save_best_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
        print(f'Max miou: {max_miou:.2f}%, macc: {max_macc:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar('perf/test_miou', test_stats['miou'], epoch)
            log_writer.add_scalar('perf/test_macc', test_stats['macc'], epoch)
            # log_writer.add_scalar('perf/test_decode_ce_loss', test_stats['decode_ce_loss'], epoch)
            # log_writer.add_scalar('perf/test_decode_dice_loss', test_stats['decode_dice_loss'], epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    main(args)

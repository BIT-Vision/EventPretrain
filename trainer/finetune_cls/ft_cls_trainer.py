import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn

from timm.utils import accuracy
from timm.loss import LabelSmoothingCrossEntropy

from utils.lr_sched import adjust_learning_rate
import utils.misc as misc
from visualize.ft_cls_visualize import vis_ft_cls, vis_ft_cls_ecdp, vis_ft_cls_mem, vis_ft_cls_swin


def ft_train_one_epoch(args, model, data_loader, optimizer, epoch, loss_scaler, log_writer=None, evrepsl_model=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch + 1)

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (events_voxel_grid, label, image_name) in enumerate(
            metric_logger.log_every(args, data_loader, args.print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % args.accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        with torch.cuda.amp.autocast():
            events_voxel_grid = events_voxel_grid.to(args.device, non_blocking=True)
            label = label.to(args.device, non_blocking=True)

            if args.use_evrepsl:
                events_voxel_grid = evrepsl_model(events_voxel_grid)

            if args.backbone_type == "vit_ecdp" or args.backbone_type == "convvit_ecdp" or args.backbone_type == "vit_mem":
                emb, pred, attn = model(events_voxel_grid)
            elif args.backbone_type == "swin" or args.backbone_type == "swin_ecddp":
                emb_l1, emb_l2, emb_l3, emb_l4, emb_h, pred, attn = model(events_voxel_grid)
            else:
                emb_l1, emb_l2, emb_h, pred, attn = model(events_voxel_grid)

        if args.test_experiment and args.visualize:
            if args.backbone_type == "vit_ecdp" or args.backbone_type == "convvit_ecdp":
                vis_ft_cls_ecdp(args, events_voxel_grid[0], attn[0], image_name[0], epoch, is_train=True)
            elif args.backbone_type == "vit_mem":
                vis_ft_cls_mem(args, events_voxel_grid[0], attn[0], image_name[0], epoch, is_train=True)
            elif args.backbone_type == "swin" or args.backbone_type == "swin_ecddp":
                vis_ft_cls_swin(args, events_voxel_grid[0], emb_l1[0], emb_l2[0], emb_l3[0], emb_l4[0], emb_h[0], attn[0],
                                image_name[0], epoch, is_train=True)
            elif args.backbone_type == "vit" or args.backbone_type == "convvit":
                vis_ft_cls(args, events_voxel_grid[0], emb_l1[0], emb_l2[0], emb_h[0], attn[0],
                           image_name[0], epoch, is_train=True)
            else:
                raise ValueError

        # cls loss
        if args.smoothing > 0:
            loss_cls = LabelSmoothingCrossEntropy(smoothing=args.smoothing)(pred, label)
        else:
            loss_cls = nn.CrossEntropyLoss()(pred, label)

        if args.backward:
            loss_cls /= args.accum_iter
            loss_scaler(loss_cls, optimizer, clip_grad=args.clip_grad, parameters=model.parameters(),  # backward
                        update_grad=(data_iter_step + 1) % args.accum_iter == 0)

            if (data_iter_step + 1) % args.accum_iter == 0:
                optimizer.zero_grad()

        if args.device == 'cuda':
            torch.cuda.synchronize()

        metric_logger.update(loss_cls=loss_cls.item())
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_cls_reduce = misc.all_reduce_mean(loss_cls.item())
        if log_writer is not None and (data_iter_step + 1) % args.log_freq == 0 and (data_iter_step + 1) % args.accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss_cls', loss_cls_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    if args.visualize and (epoch + 1) % args.vis_train_freq == 0:
        if args.backbone_type == "vit_ecdp" or args.backbone_type == "convvit_ecdp":
            vis_ft_cls_ecdp(args, events_voxel_grid[0], attn[0], image_name[0], epoch, is_train=True)
        elif args.backbone_type == "vit_mem":
            vis_ft_cls_mem(args, events_voxel_grid[0], attn[0], image_name[0], epoch, is_train=True)
        elif args.backbone_type == "swin" or args.backbone_type == "swin_ecddp":
            vis_ft_cls_swin(args, events_voxel_grid[0], emb_l1[0], emb_l2[0], emb_l3[0], emb_l4[0], emb_h[0], attn[0],
                            image_name[0], epoch, is_train=True)
        elif args.backbone_type == "vit" or args.backbone_type == "convvit":
            vis_ft_cls(args, events_voxel_grid[0], emb_l1[0], emb_l2[0], emb_h[0], attn[0],
                       image_name[0], epoch, is_train=True)
        else:
            raise ValueError

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def ft_val(args, model, data_loader, epoch, dataset_name="origin", evrepsl_model=None):
    model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    infer_time = 0
    for events_voxel_grid, label, image_name in metric_logger.log_every(args, data_loader, args.print_freq, header):
        with torch.cuda.amp.autocast():
            # events_voxel_grid = torch.load("/data/liuruonan/liuruonan/Datasets/Pretrain/EF-ImageNet/n01440764/n01440764_10026/events/noisy/n01440764_10026_00_noisy_events_voxel_grid.pt")
            events_voxel_grid = events_voxel_grid.to(args.device, non_blocking=True)
            label = label.to(args.device, non_blocking=True)

            if args.use_evrepsl:
                events_voxel_grid = evrepsl_model(events_voxel_grid)

            infer_start_time = time.time()
            if args.backbone_type == "vit_ecdp" or args.backbone_type == "convvit_ecdp" or args.backbone_type == "vit_mem":
                emb, pred, attn = model(events_voxel_grid)
            elif args.backbone_type == "swin" or args.backbone_type == "swin_ecddp":
                emb_l1, emb_l2, emb_l3, emb_l4, emb_h, pred, attn = model(events_voxel_grid)
            else:
                emb_l1, emb_l2, emb_h, pred, attn = model(events_voxel_grid)
            infer_end_time = time.time()
            infer_time += infer_end_time - infer_start_time

        if args.test_experiment and args.visualize:
            if args.backbone_type == "vit_ecdp" or args.backbone_type == "convvit_ecdp":
                vis_ft_cls_ecdp(args, events_voxel_grid[0], attn[0], image_name[0], epoch,
                                is_train=False, dataset_name=dataset_name)
            elif args.backbone_type == "vit_mem":
                vis_ft_cls_mem(args, events_voxel_grid[0], attn[0], image_name[0], epoch, is_train=False)
            elif args.backbone_type == "swin" or args.backbone_type == "swin_ecddp":
                vis_ft_cls_swin(args, events_voxel_grid[0], emb_l1[0], emb_l2[0], emb_l3[0], emb_l4[0], emb_h[0], attn[0],
                                image_name[0], epoch, is_train=True)
            elif args.backbone_type == "vit" or args.backbone_type == "convvit":
                vis_ft_cls(args, events_voxel_grid[0], emb_l1[0], emb_l2[0], emb_h[0], attn[0],
                           image_name[0], epoch, is_train=False, dataset_name=dataset_name)
            else:
                raise ValueError

        # cls loss
        loss_cls = nn.CrossEntropyLoss()(pred, label)

        # accuracy
        if args.dataset_type != "n-cars":
            acc1, acc5 = accuracy(pred, label, topk=(1, 5))
            metric_logger.update(loss_cls=loss_cls.item())
            metric_logger.update(acc1=acc1.item())
            metric_logger.update(acc5=acc5.item())
        else:
            acc1 = accuracy(pred, label, topk=(1,))
            metric_logger.update(loss_cls=loss_cls.item())
            metric_logger.update(acc1=acc1[0].item())

    if args.visualize and (epoch + 1) % args.vis_val_freq == 0:
        if args.backbone_type == "vit_mem":
            vis_ft_cls_mem(args, events_voxel_grid[0], attn[0], image_name[0], epoch, is_train=False)
        elif args.backbone_type == "vit_ecdp" or args.backbone_type == "convvit_ecdp":
            vis_ft_cls_ecdp(args, events_voxel_grid[0], attn[0], image_name[0], epoch,
                            is_train=False, dataset_name=dataset_name)
        elif args.backbone_type == "swin" or args.backbone_type == "swin_ecddp":
            vis_ft_cls_swin(args, events_voxel_grid[0], emb_l1[0], emb_l2[0], emb_l3[0], emb_l4[0], emb_h[0], attn[0],
                            image_name[0], epoch, is_train=True)
        elif args.backbone_type == "vit" or args.backbone_type == "convvit":
            vis_ft_cls(args, events_voxel_grid[0], emb_l1[0], emb_l2[0], emb_h[0], attn[0],
                       image_name[0], epoch, is_train=False, dataset_name=dataset_name)
        else:
            raise ValueError

    metric_logger.synchronize_between_processes()

    if args.dataset_type != "n-cars":
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss_cls {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss_cls))
    else:
        print('* Acc@1 {top1.global_avg:.3f} loss_cls {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, losses=metric_logger.loss_cls))

    print("average inference time (ms): %.2f" % (infer_time / len(data_loader) * 1.e3))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

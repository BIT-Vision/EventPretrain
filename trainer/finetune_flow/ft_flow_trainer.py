import matplotlib.pyplot as plt
import time

import torch

from utils.lr_sched import adjust_learning_rate
import utils.misc as misc
from utils.reshape import resize_flow
from visualize.ft_flow_visualize import vis_ft_flow, vis_ft_flow_mem, vis_ft_flow_ecdp, vis_ft_flow_swin
from trainer.finetune_flow.flow_loss import FlowLoss
from trainer.finetune_flow.flow_metric import flow_compute_aee_outlier


def ft_flow_train_one_epoch(args, model, data_loader, optimizer, epoch, loss_scaler, log_writer=None, evrepsl_model=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch + 1)

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    flow_loss = FlowLoss(args)
    for data_iter_step, (events_voxel_grid, events_voxel_grid_org, flow_label, flow_label_valid, seq_name) in enumerate(
            metric_logger.log_every(args, data_loader, args.print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % args.accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        with torch.cuda.amp.autocast():
            events_voxel_grid = events_voxel_grid.to(args.device, non_blocking=True)
            events_voxel_grid_org = events_voxel_grid_org.to(args.device, non_blocking=True)
            flow_label = flow_label.to(args.device, non_blocking=True)
            flow_label_valid = flow_label_valid.to(args.device, non_blocking=True)

            if args.use_evrepsl:
                events_voxel_grid = evrepsl_model(events_voxel_grid)

            if args.backbone_type == "vit_ecdp" or args.backbone_type == "convvit_ecdp" or args.backbone_type == "vit_mem":
                emb, out_embs, attn, decode_predict, aux_predict = model(events_voxel_grid)
            elif args.backbone_type == "swin" or args.backbone_type == "swin_ecddp":
                emb_l1, emb_l2, emb_l3, emb_l4, emb_h, out_embs, attn, decode_predict, aux_predict = model(events_voxel_grid)
            else:
                emb_l1, emb_l2, emb_h, out_embs, attn, decode_predict, aux_predict = model(events_voxel_grid)

        decode_predict = resize_flow(args, input=decode_predict, size=flow_label.shape[2:], mode=args.sample_mode)
        aux_predict = resize_flow(args, input=aux_predict, size=flow_label.shape[2:], mode=args.sample_mode)

        # l1_loss
        decode_l1_loss = flow_loss(decode_predict, flow_label, flow_label_valid)
        aux_l1_loss = flow_loss(aux_predict, flow_label, flow_label_valid)

        # aee, outlier
        events_mask = torch.linalg.norm(events_voxel_grid_org, ord=2, dim=1, keepdims=False) > 0
        sparse_mask = torch.logical_and(flow_label_valid.squeeze(1), events_mask)  # (2,224,224)
        # decode_aee_dense, decode_outlier_dense = flow_compute_aee_outlier(decode_predict, flow_label,
        #                                                       mask=flow_label_valid.squeeze(1))
        # aux_aee_dense, aux_outlier_dense = flow_compute_aee_outlier(aux_predict, flow_label,
        #                                                 mask=flow_label_valid.squeeze(1))
        # decode_aee_sparse, decode_outlier_sparse = flow_compute_aee_outlier(decode_predict, flow_label,
        #                                                       mask=sparse_mask)
        # aux_aee_sparse, aux_outlier_sparse = flow_compute_aee_outlier(aux_predict, flow_label,
        #                                                 mask=sparse_mask)

        if args.test_experiment and args.visualize:
            if args.backbone_type == "vit_mem":
                vis_ft_flow_mem(args, events_voxel_grid[0], sparse_mask[0],
                            flow_label[0], flow_label_valid[0], decode_predict[0], aux_predict[0],
                            out_embs[-1][0], attn[0], seq_name[0], epoch, is_train=True)
            elif args.backbone_type == "vit_ecdp":
                vis_ft_flow_ecdp(args, events_voxel_grid[0], sparse_mask[0],
                            flow_label[0], flow_label_valid[0], decode_predict[0], aux_predict[0],
                            out_embs[-1][0], attn[0], seq_name[0], epoch, is_train=True)
            elif args.backbone_type == "swin" or args.backbone_type == "swin_ecddp":
                vis_ft_flow_swin(args, events_voxel_grid[0], sparse_mask[0],
                                 flow_label[0], flow_label_valid[0], decode_predict[0], aux_predict[0],
                                 emb_l1[0], emb_l2[0], emb_l3[0], emb_l4[0], emb_h[0], attn[0],
                                 seq_name[0], epoch, is_train=True)
            elif args.backbone_type == "vit" or args.backbone_type == "convvit":
                vis_ft_flow(args, events_voxel_grid[0], sparse_mask[0],
                            flow_label[0], flow_label_valid[0], decode_predict[0], aux_predict[0],
                            emb_l1[0], emb_l2[0], emb_h[0], attn[0], seq_name[0], epoch, is_train=True)
            else:
                raise ValueError

        loss_total = args.decode_loss_weight * decode_l1_loss + args.aux_loss_weight * aux_l1_loss

        metric_logger.update(loss_total=loss_total.item())
        # metric_logger.update(decode_l1_loss=decode_l1_loss.item())
        # metric_logger.update(decode_aee_dense=decode_aee_dense.item())
        # metric_logger.update(decode_outlier_dense=decode_outlier_dense.item())
        # metric_logger.update(decode_aee_sparse=decode_aee_sparse.item())
        # metric_logger.update(decode_outlier_sparse=decode_outlier_sparse.item())
        # metric_logger.update(aux_l1_loss=aux_l1_loss.item())
        # metric_logger.update(aux_aee_dense=aux_aee_dense.item())
        # metric_logger.update(aux_outlier_dense=aux_outlier_dense.item())
        # metric_logger.update(aux_aee_sparse=aux_aee_sparse.item())
        # metric_logger.update(aux_outlier_sparse=aux_outlier_sparse.item())

        loss_total /= args.accum_iter
        if args.backward:
            loss_scaler(loss_total, optimizer, clip_grad=args.clip_grad, parameters=model.parameters(),  # backward
                        update_grad=(data_iter_step + 1) % args.accum_iter == 0)

            if (data_iter_step + 1) % args.accum_iter == 0:
                optimizer.zero_grad()

        if args.device == 'cuda':
            torch.cuda.synchronize()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_total_reduce = misc.all_reduce_mean(loss_total.item())
        if log_writer is not None and (data_iter_step + 1) % args.log_freq == 0 and (data_iter_step + 1) % args.accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss_total', loss_total_reduce, epoch_1000x)
            # log_writer.add_scalar('decode_l1_loss', decode_l1_loss.item(), epoch_1000x)
            # log_writer.add_scalar('decode_aee_dense', decode_aee_dense.item(), epoch_1000x)
            # log_writer.add_scalar('decode_outlier_dense', decode_outlier_dense.item(), epoch_1000x)
            # log_writer.add_scalar('decode_aee_sparse', decode_aee_sparse.item(), epoch_1000x)
            # log_writer.add_scalar('decode_outlier_sparse', decode_outlier_sparse.item(), epoch_1000x)
            # log_writer.add_scalar('aux_l1_loss', aux_l1_loss.item(), epoch_1000x)
            # log_writer.add_scalar('aux_aee_dense', aux_aee_dense.item(), epoch_1000x)
            # log_writer.add_scalar('aux_outlier_dense', aux_outlier_dense.item(), epoch_1000x)
            # log_writer.add_scalar('aux_aee_sparse', aux_aee_sparse.item(), epoch_1000x)
            # log_writer.add_scalar('aux_outlier_sparse', aux_outlier_sparse.item(), epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    if args.visualize and (epoch + 1) % args.vis_train_freq == 0:
        if args.backbone_type == "vit_mem":
            vis_ft_flow_mem(args, events_voxel_grid[0], sparse_mask[0],
                        flow_label[0], flow_label_valid[0], decode_predict[0], aux_predict[0],
                        out_embs[-1][0], attn[0], seq_name[0], epoch, is_train=True)
        elif args.backbone_type == "vit_ecdp":
            vis_ft_flow_ecdp(args, events_voxel_grid[0], sparse_mask[0],
                             flow_label[0], flow_label_valid[0], decode_predict[0], aux_predict[0],
                             out_embs[-1][0], attn[0], seq_name[0], epoch, is_train=True)
        elif args.backbone_type == "swin" or args.backbone_type == "swin_ecddp":
            vis_ft_flow_swin(args, events_voxel_grid[0], sparse_mask[0],
                             flow_label[0], flow_label_valid[0], decode_predict[0], aux_predict[0],
                             emb_l1[0], emb_l2[0], emb_l3[0], emb_l4[0], emb_h[0], attn[0],
                             seq_name[0], epoch, is_train=True)
        elif args.backbone_type == "vit" or args.backbone_type == "convvit":
            vis_ft_flow(args, events_voxel_grid[0], sparse_mask[0],
                        flow_label[0], flow_label_valid[0], decode_predict[0], aux_predict[0],
                        emb_l1[0], emb_l2[0], emb_h[0], attn[0], seq_name[0], epoch, is_train=True)  # 取当前batch_size的第0个
        else:
            raise ValueError

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def ft_flow_val(args, model, data_loader, epoch, dataset_name, evrepsl_model=None):
    model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    infer_time = 0
    flow_loss = FlowLoss(args)
    for events_voxel_grid, events_voxel_grid_org, flow_label, flow_label_valid, seq_name in metric_logger.log_every(args, data_loader, args.print_freq, header):
        with torch.cuda.amp.autocast():
            events_voxel_grid = events_voxel_grid.to(args.device, non_blocking=True)
            events_voxel_grid_org = events_voxel_grid_org.to(args.device, non_blocking=True)
            flow_label = flow_label.to(args.device, non_blocking=True)
            flow_label_valid = flow_label_valid.to(args.device, non_blocking=True)

            if args.use_evrepsl:
                events_voxel_grid = evrepsl_model(events_voxel_grid)

            infer_start_time = time.time()
            if args.backbone_type == "vit_ecdp" or args.backbone_type == "convvit_ecdp" or args.backbone_type == "vit_mem":
                emb, out_embs, attn, decode_predict, aux_predict = model(events_voxel_grid)
            elif args.backbone_type == "swin" or args.backbone_type == "swin_ecddp":
                emb_l1, emb_l2, emb_l3, emb_l4, emb_h, out_embs, attn, decode_predict, aux_predict = model(events_voxel_grid)
            else:
                emb_l1, emb_l2, emb_h, out_embs, attn, decode_predict, aux_predict = model(events_voxel_grid)
            infer_end_time = time.time()
            infer_time += infer_end_time - infer_start_time

        decode_predict = resize_flow(args, input=decode_predict, size=flow_label.shape[2:], mode=args.sample_mode)
        aux_predict = resize_flow(args, input=decode_predict, size=flow_label.shape[2:], mode=args.sample_mode)

        # l1_loss
        decode_l1_loss = flow_loss(decode_predict, flow_label, flow_label_valid)

        # aee, outlier
        events_mask = torch.linalg.norm(events_voxel_grid_org, ord=2, dim=1, keepdims=False) > 0
        sparse_mask = torch.logical_and(flow_label_valid.squeeze(1), events_mask)  # (2,224,224)
        # decode_aee_dense, decode_outlier_dense = flow_compute_aee_outlier(decode_predict, flow_label,
        #                                                           mask=flow_label_valid.squeeze(1))
        decode_aee_sparse, decode_outlier_sparse = flow_compute_aee_outlier(decode_predict, flow_label,
                                                              mask=sparse_mask)

        if args.test_experiment and args.visualize:
            if args.backbone_type == "vit_mem":
                vis_ft_flow_mem(args, events_voxel_grid[0], sparse_mask[0],
                            flow_label[0], flow_label_valid[0], decode_predict[0], aux_predict[0],
                            out_embs[-1][0], attn[0], seq_name[0], epoch, is_train=False)
            elif args.backbone_type == "vit_ecdp":
                vis_ft_flow_ecdp(args, events_voxel_grid[0], sparse_mask[0],
                            flow_label[0], flow_label_valid[0], decode_predict[0], aux_predict[0],
                            out_embs[-1][0], attn[0], seq_name[0], epoch, is_train=False)
            elif args.backbone_type == "swin" or args.backbone_type == "swin_ecddp":
                vis_ft_flow_swin(args, events_voxel_grid[0], sparse_mask[0],
                                 flow_label[0], flow_label_valid[0], decode_predict[0], aux_predict[0],
                                 emb_l1[0], emb_l2[0], emb_l3[0], emb_l4[0], emb_h[0], attn[0],
                                 seq_name[0], epoch, is_train=True)
            elif args.backbone_type == "vit" or args.backbone_type == "convvit":
                vis_ft_flow(args, events_voxel_grid[0], sparse_mask[0],
                            flow_label[0], flow_label_valid[0], decode_predict[0], aux_predict[0],
                            emb_l1[0], emb_l2[0], emb_h[0], attn[0], seq_name[0], epoch,
                            is_train=False, dataset_name=dataset_name)  # 取当前batch_size的第0个
            else:
                raise ValueError

        metric_logger.update(decode_l1_loss=decode_l1_loss.item())
        # metric_logger.update(decode_aee_dense=decode_aee_dense.item())
        # metric_logger.update(decode_outlier_dense=decode_outlier_dense.item())
        metric_logger.update(decode_aee_sparse=decode_aee_sparse.item())
        metric_logger.update(decode_outlier_sparse=decode_outlier_sparse.item())

    if args.visualize and (epoch + 1) % args.vis_val_freq == 0:
        if args.backbone_type == "vit_mem":
            vis_ft_flow_mem(args, events_voxel_grid[0], sparse_mask[0],
                            flow_label[0], flow_label_valid[0], decode_predict[0], aux_predict[0],
                            out_embs[-1][0], attn[0], seq_name[0], epoch, is_train=False, dataset_name=dataset_name)
        elif args.backbone_type == "vit_ecdp":
            vis_ft_flow_ecdp(args, events_voxel_grid[0], sparse_mask[0],
                             flow_label[0], flow_label_valid[0], decode_predict[0], aux_predict[0],
                             out_embs[-1][0], attn[0], seq_name[0], epoch, is_train=False, dataset_name=dataset_name)
        elif args.backbone_type == "swin" or args.backbone_type == "swin_ecddp":
            vis_ft_flow_swin(args, events_voxel_grid[0], sparse_mask[0],
                             flow_label[0], flow_label_valid[0], decode_predict[0], aux_predict[0],
                             emb_l1[0], emb_l2[0], emb_l3[0], emb_l4[0], emb_h[0], attn[0],
                             seq_name[0], epoch, is_train=True)
        elif args.backbone_type == "vit" or args.backbone_type == "convvit":
            vis_ft_flow(args, events_voxel_grid[0], sparse_mask[0],
                        flow_label[0], flow_label_valid[0], decode_predict[0], aux_predict[0],
                        emb_l1[0], emb_l2[0], emb_h[0], attn[0], seq_name[0], epoch,
                        is_train=False, dataset_name=dataset_name)  # 取当前batch_size的第0个
        else:
            raise ValueError

    metric_logger.synchronize_between_processes()

    # print('* decode_aee_dense {decode_aee_dense.global_avg:.3f} decode_outlier_dense {decode_outlier_dense.global_avg:.3f} '
    #       'decode_aee_sparse {decode_aee_sparse.global_avg:.3f} decode_outlier_sparse {decode_outlier_sparse.global_avg:.3f} '
    #       'decode_l1_loss {decode_l1_loss.global_avg:.3f}'
    #       .format(decode_aee_dense=metric_logger.decode_aee_dense, decode_outlier_dense=metric_logger.decode_outlier_dense,
    #               decode_aee_sparse=metric_logger.decode_aee_sparse, decode_outlier_sparse=metric_logger.decode_outlier_sparse,
    #               decode_l1_loss=metric_logger.decode_l1_loss))
    print('* decode_aee_sparse {decode_aee_sparse.global_avg:.3f} decode_outlier_sparse {decode_outlier_sparse.global_avg:.3f} '
        'decode_l1_loss {decode_l1_loss.global_avg:.3f}'
        .format(decode_aee_sparse=metric_logger.decode_aee_sparse,
                decode_outlier_sparse=metric_logger.decode_outlier_sparse,
                decode_l1_loss=metric_logger.decode_l1_loss))

    print("average inference time (ms): %.2f" % (infer_time / len(data_loader) * 1.e3))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

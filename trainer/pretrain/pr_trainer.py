import torch

from utils.lr_sched import adjust_learning_rate
from visualize.pr_visualize import vis_pr_rec, vis_pr_con, vis_pr_rec_and_con, vis_pr_rec_swin
# from visualize.pr_visualize_save import vis_pr_rec, vis_pr_con, vis_pr_rec_and_con
import utils.misc as misc


def pr_rec_one_epoch(args, model, data_loader, optimizer, epoch, loss_scaler, log_writer=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch + 1)

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (events_voxel_grid, sub_frame, image_name) in enumerate(
            metric_logger.log_every(args, data_loader, args.print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % args.accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        with torch.cuda.amp.autocast():
            events_voxel_grid = events_voxel_grid.to(args.device, non_blocking=True)  # (5,224,224)
            sub_frame = sub_frame.to(args.device, non_blocking=True)

            if args.backbone_type == "swin":
                reconstruct_loss, emb_l1, emb_l2, emb_l3, emb_l4, emb_lh, \
                    coords_l1, coords_l2, coords_l3, coords_l4, reconstruct_pred, mask, ids_restore, attn = \
                        model(events_voxel_grid, sub_frame, is_rec=True)
            else:
                reconstruct_loss, emb_l1, emb_l2, emb_lh, reconstruct_pred, mask, ids_restore = \
                        model(events_voxel_grid, sub_frame, is_rec=True)

        if args.test_experiment and args.visualize:
            if args.backbone_type == "swin":
                vis_pr_rec_swin(args, events_voxel_grid[0], emb_l1[0], emb_l2[0], emb_l3[0], emb_l4[0], emb_lh[0],
                                coords_l1[0], coords_l2[0], coords_l3[0], coords_l4[0],
                                sub_frame[0], reconstruct_pred[0], mask[0], ids_restore[0], attn[0],
                                image_name[0], epoch)
            else:
                vis_pr_rec(args, events_voxel_grid[0], emb_l1[0], emb_l2[0], emb_lh[0],
                           sub_frame[0], reconstruct_pred[0], mask[0], ids_restore[0],
                           image_name[0], epoch)
        metric_logger.update(reconstruct_loss=reconstruct_loss.item())
        reconstruct_loss /= args.accum_iter

        if args.backward:
            loss_scaler(reconstruct_loss, optimizer, parameters=model.parameters(),  # backward
                        update_grad=(data_iter_step + 1) % args.accum_iter == 0)

            if (data_iter_step + 1) % args.accum_iter == 0:
                optimizer.zero_grad()

        if args.device == 'cuda':
            torch.cuda.synchronize()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        reconstruct_loss_reduce = misc.all_reduce_mean(reconstruct_loss.item())
        if log_writer is not None and (data_iter_step + 1) % args.log_freq == 0 and (data_iter_step + 1) % args.accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('reconstruct_loss',
                                  reconstruct_loss_reduce,
                                  epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    if args.visualize and (epoch + 1) % args.vis_train_freq == 0:
        if args.backbone_type == "swin":
            vis_pr_rec_swin(args, events_voxel_grid[0], emb_l1[0], emb_l2[0], emb_l3[0], emb_l4[0], emb_lh[0],
                            coords_l1[0], coords_l2[0], coords_l3[0], coords_l4[0],
                            sub_frame[0], reconstruct_pred[0], mask[0], ids_restore[0], attn[0],
                            image_name[0], epoch)
        else:
            vis_pr_rec(args, events_voxel_grid[0], emb_l1[0], emb_l2[0], emb_lh[0],
                       sub_frame[0], reconstruct_pred[0], mask[0], ids_restore[0],
                       image_name[0], epoch)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def pr_con_one_epoch(args, model, data_loader, optimizer, epoch, loss_scaler, log_writer=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch + 1)

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (events_voxel_grid, clip_emb, image_name) in enumerate(
            metric_logger.log_every(args, data_loader, args.print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % args.accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        with torch.cuda.amp.autocast():
            events_voxel_grid = events_voxel_grid.to(args.device, non_blocking=True)  # (5,224,224)
            clip_emb = clip_emb.to(args.device, non_blocking=True)

            contrastive_loss, emb_h_org, emb_h_proj, clip_emb_org, clip_emb_proj, attn = \
                model(events_voxel_grid, clip_emb)

        if args.test_experiment and args.visualize:
            vis_pr_con(args, events_voxel_grid[0], emb_h_org[0], emb_h_proj[0],
                       clip_emb_org[0], clip_emb_proj[0], attn[0],
                       image_name[0], epoch)

        metric_logger.update(contrastive_loss=contrastive_loss.item())
        contrastive_loss /= args.accum_iter

        if args.backward:
            loss_scaler(contrastive_loss, optimizer, parameters=model.parameters(),  # backward
                        update_grad=(data_iter_step + 1) % args.accum_iter == 0)

            if (data_iter_step + 1) % args.accum_iter == 0:
                optimizer.zero_grad()

        if args.device == 'cuda':
            torch.cuda.synchronize()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        contrastive_loss_reduce = misc.all_reduce_mean(contrastive_loss.item())
        if log_writer is not None and (data_iter_step + 1) % args.log_freq == 0 and (data_iter_step + 1) % args.accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('contrastive_loss',
                                  contrastive_loss_reduce,
                                  epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    if args.visualize and (epoch + 1) % args.vis_train_freq == 0:
        vis_pr_con(args, events_voxel_grid[0], emb_h_org[0], emb_h_proj[0],
                   clip_emb_org[0], clip_emb_proj[0], attn[0],
                   image_name[0], epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def pr_con_n_one_epoch(args, model, preprocess, clip_model, data_loader, optimizer, epoch, loss_scaler, log_writer=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch + 1)

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (events_voxel_grid, image_preprocess, image_name) in enumerate(
            metric_logger.log_every(args, data_loader, args.print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % args.accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        with torch.cuda.amp.autocast():
            events_voxel_grid = events_voxel_grid.to(args.device, non_blocking=True)  # (5,224,224)
            image_preprocess = image_preprocess.to(args.device)
            clip_emb = clip_model.encode_image(image_preprocess).to(args.device, non_blocking=True)

            contrastive_loss, emb_h_org, emb_h_proj, clip_emb_org, clip_emb_proj, attn = \
                model(events_voxel_grid, clip_emb)

        if args.test_experiment and args.visualize:
            vis_pr_con(args, events_voxel_grid[0], emb_h_org[0], emb_h_proj[0],
                       clip_emb_org[0], clip_emb_proj[0], attn[0],
                       image_name[0], epoch)

        metric_logger.update(contrastive_loss=contrastive_loss.item())
        contrastive_loss /= args.accum_iter

        if args.backward:
            loss_scaler(contrastive_loss, optimizer, parameters=model.parameters(),  # backward
                        update_grad=(data_iter_step + 1) % args.accum_iter == 0)

            if (data_iter_step + 1) % args.accum_iter == 0:
                optimizer.zero_grad()

        if args.device == 'cuda':
            torch.cuda.synchronize()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        contrastive_loss_reduce = misc.all_reduce_mean(contrastive_loss.item())
        if log_writer is not None and (data_iter_step + 1) % args.log_freq == 0 and (data_iter_step + 1) % args.accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('contrastive_loss',
                                  contrastive_loss_reduce,
                                  epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    if args.visualize and (epoch + 1) % args.vis_train_freq == 0:
        vis_pr_con(args, events_voxel_grid[0], emb_h_org[0], emb_h_proj[0],
                   clip_emb_org[0], clip_emb_proj[0], attn[0],
                   image_name[0], epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def pr_rec_and_con_one_epoch(args, model, data_loader, optimizer, epoch, loss_scaler, log_writer=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch + 1)

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (events_voxel_grid, sub_frame, clip_emb, image_name) in enumerate(
            metric_logger.log_every(args, data_loader, args.print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % args.accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        with torch.cuda.amp.autocast():
            events_voxel_grid = events_voxel_grid.to(args.device, non_blocking=True)  # (5,224,224)
            sub_frame = sub_frame.to(args.device, non_blocking=True)
            clip_emb = clip_emb.to(args.device, non_blocking=True)

            # rec
            reconstruct_loss, emb_l1, emb_l2, emb_lh, reconstruct_pred, mask, ids_restore = \
                model(events_voxel_grid, sub_frame, is_rec=True)
            # con
            contrastive_loss, emb_h_org, emb_h_proj, clip_emb_org, clip_emb_proj, attn = \
                model(events_voxel_grid, clip_emb)

        if args.test_experiment and args.visualize:
            vis_pr_rec_and_con(args, events_voxel_grid[0], emb_l1[0], emb_l2[0], emb_lh[0],
                               sub_frame[0], reconstruct_pred[0], mask[0], ids_restore[0],
                               emb_h_org[0], emb_h_proj[0],
                               clip_emb_org[0], clip_emb_proj[0], attn[0],
                               image_name[0], epoch)

        metric_logger.update(reconstruct_loss=reconstruct_loss.item())
        metric_logger.update(contrastive_loss=contrastive_loss.item())
        loss_total = reconstruct_loss + contrastive_loss
        loss_total /= args.accum_iter

        if args.backward:
            loss_scaler(loss_total, optimizer, parameters=model.parameters(),  # backward
                        update_grad=(data_iter_step + 1) % args.accum_iter == 0)

            if (data_iter_step + 1) % args.accum_iter == 0:
                optimizer.zero_grad()

        if args.device == 'cuda':
            torch.cuda.synchronize()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        reconstruct_loss_reduce = misc.all_reduce_mean(reconstruct_loss.item())
        contrastive_loss_reduce = misc.all_reduce_mean(contrastive_loss.item())
        if log_writer is not None and (data_iter_step + 1) % args.log_freq == 0 and (data_iter_step + 1) % args.accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('reconstruct_loss',
                                  reconstruct_loss_reduce,
                                  epoch_1000x)
            log_writer.add_scalar('contrastive_loss',
                                  contrastive_loss_reduce,
                                  epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    if args.visualize and (epoch + 1) % args.vis_train_freq == 0:
        vis_pr_rec_and_con(args, events_voxel_grid[0], emb_l1[0], emb_l2[0], emb_lh[0],
                           sub_frame[0], reconstruct_pred[0], mask[0], ids_restore[0],
                           emb_h_org[0], emb_h_proj[0],
                           clip_emb_org[0], clip_emb_proj[0], attn[0],
                           image_name[0], epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

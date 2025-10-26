import matplotlib.pyplot as plt
import time

import torch

from utils.lr_sched import adjust_learning_rate
import utils.misc as misc
from utils.reshape import resize
from visualize.ft_semseg_visualize import vis_ft_semseg, vis_ft_semseg_ecdp, vis_ft_semseg_mem, vis_ft_semseg_swin
from trainer.finetune_semseg.semseg_loss import SemsegLoss
from trainer.finetune_semseg.semseg_metric import semseg_compute_confusion, semseg_confusion_to_miou, semseg_confusion_to_macc


def ft_semseg_train_one_epoch(args, model, data_loader, optimizer, epoch, loss_scaler, log_writer=None, evrepsl_model=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch + 1)

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    semseg_loss = SemsegLoss(args, num_classes=args.num_classes, ignore_index=args.ignore_label)
    for data_iter_step, (events_voxel_grid, semseg_label, seq_name) in enumerate(
            metric_logger.log_every(args, data_loader, args.print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % args.accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        with torch.cuda.amp.autocast():
            events_voxel_grid = events_voxel_grid.to(args.device, non_blocking=True)
            semseg_label = semseg_label.to(args.device, non_blocking=True)

            if args.use_evrepsl:
                events_voxel_grid = evrepsl_model(events_voxel_grid)
                # print(events_voxel_grid[0, 0, :, :].min().item(), events_voxel_grid[0, 0, :, :].max().item(),
                #       events_voxel_grid[0, 1, :, :].min().item(), events_voxel_grid[0, 1, :, :].max().item(),
                #       events_voxel_grid[0, 2, :, :].min().item(), events_voxel_grid[0, 2, :, :].max().item(),
                #       events_voxel_grid[0, 3, :, :].min().item(), events_voxel_grid[0, 3, :, :].max().item(),
                #       events_voxel_grid[0, 4, :, :].min().item(), events_voxel_grid[0, 4, :, :].max().item())
                # print(events_voxel_grid.max().item())
                # if torch.isnan(events_voxel_grid.max()):
                # plt.imshow(events_voxel_grid.detach().cpu().numpy()[0][0], cmap='gray')
                # plt.axis('off')
                # plt.show()
                # plt.close()
                #
                # plt.imshow(events_voxel_grid.detach().cpu().numpy()[0][1], cmap='gray')
                # plt.axis('off')
                # plt.show()
                # plt.close()
                #
                # plt.imshow(events_voxel_grid.detach().cpu().numpy()[0][2], cmap='gray')
                # plt.axis('off')
                # plt.show()
                # plt.close()
                #
                # plt.imshow(events_voxel_grid.detach().cpu().numpy()[0][3], cmap='gray')
                # plt.axis('off')
                # plt.show()
                # plt.close()
                #
                # plt.imshow(events_voxel_grid.detach().cpu().numpy()[0][4], cmap='gray')
                # plt.axis('off')
                # plt.show()
                # plt.close()

            if args.backbone_type == "vit_ecdp" or args.backbone_type == "convvit_ecdp" or args.backbone_type == "vit_mem":
                emb, out_embs, attn, decode_predict, aux_predict = model(events_voxel_grid)
            elif args.backbone_type == "swin" or args.backbone_type == "swin_ecddp":
                emb_l1, emb_l2, emb_l3, emb_l4, emb_h, out_embs, attn, decode_predict, aux_predict = model(events_voxel_grid)
            else:
                emb_l1, emb_l2, emb_h, out_embs, attn, decode_predict, aux_predict = model(events_voxel_grid)

        decode_predict = resize(input=decode_predict, size=semseg_label.shape[2:], mode=args.sample_mode)
        aux_predict = resize(input=aux_predict, size=semseg_label.shape[2:], mode=args.sample_mode)

        # ce_loss, dice_loss
        decode_ce_loss, decode_dice_loss = semseg_loss(decode_predict, semseg_label)
        aux_ce_loss, aux_dice_loss = semseg_loss(aux_predict, semseg_label)

        # moiu, macc
        # decode_confusion = semseg_compute_confusion(args, decode_predict, semseg_label)
        # decode_miou = semseg_confusion_to_miou(decode_confusion)
        # decode_macc = semseg_confusion_to_macc(decode_confusion)
        # aux_confusion = semseg_compute_confusion(args, aux_predict, semseg_label)
        # aux_miou = semseg_confusion_to_miou(aux_confusion)
        # aux_macc = semseg_confusion_to_macc(aux_confusion)

        if args.test_experiment and args.visualize:
            if args.backbone_type == "vit_ecdp" or args.backbone_type == "convvit_ecdp":
                vis_ft_semseg_ecdp(args, events_voxel_grid[0], semseg_label[0], decode_predict[0], aux_predict[0],
                              out_embs[-1][0], attn[0], seq_name[0], epoch, is_train=True)
            elif args.backbone_type == "vit_mem":
                vis_ft_semseg_mem(args, events_voxel_grid[0], semseg_label[0], decode_predict[0], aux_predict[0],
                              out_embs[-1][0], attn[0], seq_name[0], epoch, is_train=True)
            elif args.backbone_type == "swin" or args.backbone_type == "swin_ecddp":
                vis_ft_semseg_swin(args, events_voxel_grid[0], semseg_label[0], decode_predict[0], aux_predict[0],
                                   emb_l1[0], emb_l2[0], emb_l3[0], emb_l4[0], emb_h[0], attn[0], seq_name[0], epoch,
                                   is_train=True)
            elif args.backbone_type == "vit" or args.backbone_type == "convvit" or args.backbone_type == "swin_ecddp":
                vis_ft_semseg(args, events_voxel_grid[0], semseg_label[0], decode_predict[0], aux_predict[0],
                              emb_l1[0], emb_l2[0], emb_h[0], attn[0], seq_name[0], epoch,
                              is_train=True)  # 取当前batch_size的第0个
                # vis_ft_semseg_save(args, events_voxel_grid[0], label[0],
                #                           decode_predict_copy[0], decode_predict[0], aux_predict_copy[0], aux_predict[0],
                #                           emb_l1[0], emb_l2[0], emb_h[0], attn[0], data_iter_step)
            else:
                raise ValueError

        loss_total = args.decode_loss_weight * (decode_ce_loss + decode_dice_loss) + \
                     args.aux_loss_weight * (aux_ce_loss + aux_dice_loss)

        metric_logger.update(loss_total=loss_total.item())
        # metric_logger.update(decode_ce_loss=decode_ce_loss.item())
        # metric_logger.update(decode_dice_loss=decode_dice_loss.item())
        # metric_logger.update(decode_miou=decode_miou.item())
        # metric_logger.update(decode_macc=decode_macc.item())
        # metric_logger.update(aux_ce_loss=aux_ce_loss.item())
        # metric_logger.update(aux_dice_loss=aux_dice_loss.item())
        # metric_logger.update(aux_miou=aux_miou.item())
        # metric_logger.update(aux_macc=aux_macc.item())

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

        loss_total_reduce = misc.all_reduce_mean(loss_total.item())
        if log_writer is not None and (data_iter_step + 1) % args.log_freq == 0 and (data_iter_step + 1) % args.accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss_total', loss_total_reduce, epoch_1000x)
            # log_writer.add_scalar('decode_ce_loss', decode_ce_loss.item(), epoch_1000x)
            # log_writer.add_scalar('decode_dice_loss', decode_dice_loss.item(), epoch_1000x)
            # log_writer.add_scalar('decode_miou', decode_miou.item(), epoch_1000x)
            # log_writer.add_scalar('decode_macc', decode_macc.item(), epoch_1000x)
            # log_writer.add_scalar('aux_ce_loss', aux_ce_loss.item(), epoch_1000x)
            # log_writer.add_scalar('aux_dice_loss', aux_dice_loss.item(), epoch_1000x)
            # log_writer.add_scalar('aux_miou', aux_miou.item(), epoch_1000x)
            # log_writer.add_scalar('aux_macc', aux_macc.item(), epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    if args.visualize and (epoch + 1) % args.vis_train_freq == 0:
        if args.backbone_type == "vit_ecdp" or args.backbone_type == "convvit_ecdp":
            vis_ft_semseg_ecdp(args, events_voxel_grid[0], semseg_label[0], decode_predict[0], aux_predict[0],
                               out_embs[-1][0], attn[0], seq_name[0], epoch, is_train=True)
        elif args.backbone_type == "vit_mem":
            vis_ft_semseg_mem(args, events_voxel_grid[0], semseg_label[0], decode_predict[0], aux_predict[0],
                              out_embs[-1][0], attn[0], seq_name[0], epoch, is_train=True)
        elif args.backbone_type == "swin" or args.backbone_type == "swin_ecddp":
            vis_ft_semseg_swin(args, events_voxel_grid[0], semseg_label[0], decode_predict[0], aux_predict[0],
                               emb_l1[0], emb_l2[0], emb_l3[0], emb_l4[0], emb_h[0], attn[0], seq_name[0], epoch,
                               is_train=True)
        elif args.backbone_type == "vit" or args.backbone_type == "convvit" or args.backbone_type == "swin_ecddp":
            vis_ft_semseg(args, events_voxel_grid[0], semseg_label[0], decode_predict[0], aux_predict[0],
                       emb_l1[0], emb_l2[0], emb_h[0], attn[0], seq_name[0], epoch, is_train=True)  # 取当前batch_size的第0个
        else:
            raise ValueError

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def ft_semseg_val(args, model, data_loader, epoch, evrepsl_model=None):
    model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    infer_time = 0
    semseg_loss = SemsegLoss(args, num_classes=args.num_classes, ignore_index=args.ignore_label)
    for events_voxel_grid, semseg_label, seq_name in metric_logger.log_every(args, data_loader, args.print_freq, header):
        with torch.cuda.amp.autocast():
            events_voxel_grid = events_voxel_grid.to(args.device, non_blocking=True)
            semseg_label = semseg_label.to(args.device, non_blocking=True)

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

        decode_predict = resize(input=decode_predict, size=semseg_label.shape[2:], mode=args.sample_mode)
        aux_predict = resize(input=decode_predict, size=semseg_label.shape[2:], mode=args.sample_mode)

        # moiu, macc
        decode_confusion = semseg_compute_confusion(args, decode_predict, semseg_label)
        decode_miou = semseg_confusion_to_miou(decode_confusion)
        decode_macc = semseg_confusion_to_macc(decode_confusion)

        # ce_loss, dice_loss
        decode_ce_loss, decode_dice_loss = semseg_loss(decode_predict, semseg_label)

        if args.test_experiment and args.visualize:
            if args.backbone_type == "vit_ecdp" or args.backbone_type == "convvit_ecdp":
                vis_ft_semseg_ecdp(args, events_voxel_grid[0], semseg_label[0], decode_predict[0], aux_predict[0],
                              out_embs[-1][0], attn[0], seq_name[0], epoch, is_train=False)
            elif args.backbone_type == "vit_mem":
                vis_ft_semseg_mem(args, events_voxel_grid[0], semseg_label[0], decode_predict[0], aux_predict[0],
                                  out_embs[-1][0], attn[0], seq_name[0], epoch, is_train=False)
            elif args.backbone_type == "swin" or args.backbone_type == "swin_ecddp":
              vis_ft_semseg_swin(args, events_voxel_grid[0], semseg_label[0], decode_predict[0], aux_predict[0],
                               emb_l1[0], emb_l2[0], emb_l3[0], emb_l4[0], emb_h[0], attn[0], seq_name[0], epoch,
                               is_train=True)
            elif args.backbone_type == "vit" or args.backbone_type == "convvit":
                vis_ft_semseg(args, events_voxel_grid[0], semseg_label[0], decode_predict[0], aux_predict[0],
                           emb_l1[0], emb_l2[0], emb_h[0], attn[0], seq_name[0], epoch, is_train=False)  # 取当前batch_size的第0个
            else:
                raise ValueError

        loss_total = decode_ce_loss + decode_dice_loss

        metric_logger.update(loss_total=loss_total.item())
        # metric_logger.update(decode_ce_loss=decode_ce_loss.item())
        # metric_logger.update(decode_dice_loss=decode_dice_loss.item())
        metric_logger.update(miou=decode_miou.item())
        metric_logger.update(macc=decode_macc.item())

    if args.visualize and (epoch + 1) % args.vis_val_freq == 0:
        if args.backbone_type == "vit_ecdp" or args.backbone_type == "convvit_ecdp":
            vis_ft_semseg_ecdp(args, events_voxel_grid[0], semseg_label[0], decode_predict[0], aux_predict[0],
                               out_embs[-1][0], attn[0], seq_name[0], epoch, is_train=False)
        elif args.backbone_type == "vit_mem":
            vis_ft_semseg_mem(args, events_voxel_grid[0], semseg_label[0], decode_predict[0], aux_predict[0],
                              out_embs[-1][0], attn[0], seq_name[0], epoch, is_train=False)
        elif args.backbone_type == "swin" or args.backbone_type == "swin_ecddp":
            vis_ft_semseg_swin(args, events_voxel_grid[0], semseg_label[0], decode_predict[0], aux_predict[0],
                               emb_l1[0], emb_l2[0], emb_l3[0], emb_l4[0], emb_h[0], attn[0], seq_name[0], epoch,
                               is_train=True)
        elif args.backbone_type == "vit" or args.backbone_type == "convvit":
            vis_ft_semseg(args, events_voxel_grid[0], semseg_label[0], decode_predict[0], aux_predict[0],
                       emb_l1[0], emb_l2[0], emb_h[0], attn[0], seq_name[0], epoch, is_train=False)  # 取当前batch_size的第0个
        else:
            raise ValueError

    metric_logger.synchronize_between_processes()

    # print('* miou {miou.global_avg:.3f} macc {macc.global_avg:.3f} loss_total {loss_total.global_avg:.3f}'
    #       ' decode_ce_loss {decode_ce_loss.global_avg:.3f} decode_dice_loss {decode_dice_loss.global_avg:.3f}'
    #       .format(miou=metric_logger.miou, macc=metric_logger.macc, loss_total=metric_logger.loss_total,
    #               decode_ce_loss=metric_logger.decode_ce_loss, decode_dice_loss=metric_logger.decode_dice_loss))
    print('* miou {miou.global_avg:.3f} macc {macc.global_avg:.3f} loss_total {loss_total.global_avg:.3f}'
          .format(miou=metric_logger.miou, macc=metric_logger.macc, loss_total=metric_logger.loss_total))

    print("average inference time (ms): %.2f" % (infer_time / len(data_loader) * 1.e3))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

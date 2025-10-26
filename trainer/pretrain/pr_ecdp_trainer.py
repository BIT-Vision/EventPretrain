import math

import torch

from utils.lr_sched import adjust_learning_rate
from visualize.pr_visualize import vis_pr_ecdp
import utils.misc as misc


def adjust_ema_momentum(args, epoch):
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.ema_m)

    return m

def pr_ecdp_one_epoch(args, model, data_loader, optimizer, epoch, loss_scaler, log_writer=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch + 1)

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (events_image_q, events_image_k, clip_emb, image_name) in enumerate(
            metric_logger.log_every(args, data_loader, args.print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % args.accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        ema_m = adjust_ema_momentum(args, data_iter_step / len(data_loader) + epoch)
        with torch.cuda.amp.autocast():
            events_image_q = events_image_q.to(args.device, non_blocking=True)  # (2,224,224)
            events_image_k = events_image_k.to(args.device, non_blocking=True)  # (2,224,224)
            clip_emb = clip_emb.to(args.device, non_blocking=True)

            contrastive_loss_image, contrastive_loss_event, kl_loss, \
            emb_event_q_org, emb_image_q_org, emb_event_q, emb_image_q, clip_emb_org, clip_emb_proj, \
            mask_q, ids_restore_q, attn_q, mask_k, ids_restore_k, attn_k = \
                model(events_image_q, events_image_k, clip_emb, ema_m)

        if args.test_experiment and args.visualize:
            vis_pr_ecdp(args, events_image_q[0], events_image_k[0], emb_event_q_org[0], emb_image_q_org[0], emb_event_q[0], emb_image_q[0],
                        clip_emb_org[0], clip_emb_proj[0], mask_q[0], ids_restore_q[0], attn_q[0],
                        mask_k[0], ids_restore_k[0], attn_k[0], image_name[0], epoch)

        metric_logger.update(contrastive_loss_image=contrastive_loss_image.item())
        metric_logger.update(contrastive_loss_event=contrastive_loss_event.item())
        metric_logger.update(kl_loss=kl_loss.item())
        loss_total = args.lambda_image * contrastive_loss_image + args.lambda_event * contrastive_loss_event + \
                     args.lambda_kl * kl_loss
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

        contrastive_loss_image_reduce = misc.all_reduce_mean(contrastive_loss_image.item())
        contrastive_loss_event_reduce = misc.all_reduce_mean(contrastive_loss_event.item())
        kl_loss_reduce = misc.all_reduce_mean(kl_loss.item())
        if log_writer is not None and (data_iter_step + 1) % args.accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('contrastive_loss_image',
                                  contrastive_loss_image_reduce,
                                  epoch_1000x)
            log_writer.add_scalar('contrastive_loss_event',
                                  contrastive_loss_event_reduce,
                                  epoch_1000x)
            log_writer.add_scalar('kl_loss',
                                  kl_loss_reduce,
                                  epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    if args.visualize and (epoch + 1) % args.vis_train_freq == 0:
        vis_pr_ecdp(args, events_image_q[0], events_image_k[0], emb_event_q_org[0], emb_image_q_org[0], emb_event_q[0], emb_image_q[0],
                    clip_emb_org[0], clip_emb_proj[0], mask_q[0], ids_restore_q[0], attn_q[0],
                    mask_k[0], ids_restore_k[0], attn_k[0], image_name[0], epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

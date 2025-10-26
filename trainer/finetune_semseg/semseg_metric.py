import torch


def semseg_compute_confusion(args, target, predict):
    predict = predict.squeeze(1)
    target = torch.argmax(target, dim=1).squeeze(1)  # (2,11,440,640)->(2,1,440,640)->(2,440,640)

    mask = predict != args.ignore_label
    predict = predict[mask]
    target = target[mask]

    # hack for bincounting 2 arrays together
    x = target + args.num_classes * predict
    bincount_2d = torch.bincount(x.long(), minlength=args.num_classes ** 2)
    assert bincount_2d.numel() == args.num_classes ** 2, 'Internal error'
    confusion = bincount_2d.view((args.num_classes, args.num_classes)).long()

    return confusion

def semseg_confusion_to_miou(confusion):
    confusion = confusion.double()
    diag = confusion.diag()
    iou_per_class = 100 * diag / (confusion.sum(dim=1) + confusion.sum(dim=0) - diag).clamp(min=1e-12)
    miou = iou_per_class.mean()

    return miou

def semseg_confusion_to_macc(confusion):
    confusion = confusion.double()
    diag = confusion.diag()
    # macc = 100 * diag.sum() / (confusion.sum(dim=1).sum()).clamp(min=1e-12)
    acc_per_class = 100 * diag / (confusion.sum(dim=1)).clamp(min=1e-12)
    macc = acc_per_class.mean()

    return macc

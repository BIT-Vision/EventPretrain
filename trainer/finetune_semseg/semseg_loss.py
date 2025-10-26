import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemsegLoss(nn.Module):
    def __init__(self, args, num_classes=13, ignore_index=None):
        super(SemsegLoss, self).__init__()
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.dice_loss = DiceLoss(args, num_classes=num_classes, ignore_index=self.ignore_index)

    def forward(self, predict, target):  # (2,11,440,640) (2,1,440,640)
        ce_loss = self.ce_loss(predict.float(), target.squeeze(1))
        dice_loss = self.dice_loss(predict, target)

        return ce_loss, dice_loss


def make_one_hot(args, input, num_classes):
    """
        Convert class index tensor to one hot encoding tensor.
        Args:
             input: A tensor of shape [N, 1, *].
             num_classes: An int of number of class.
        Returns:
            A tensor of shape [N, num_classes, *].
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape, device=args.device)
    result = result.scatter_(1, input, 1)

    return result


class BinaryDiceLoss(nn.Module):
    """
        Dice loss of binary class.
        Args:
            smooth: A float number to smooth loss, and avoid NaN error, default: 1.
            p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2.
            predict: A tensor of shape [N, *].
            target: A tensor of shape same with predict.
        Returns:
            Loss tensor according to arg reduction.
        Raise:
            Exception if unexpected reduction.
    """
    def __init__(self, smooth=1, p=2):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target))*2 + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p)) + self.smooth

        dice = num / den
        loss = 1 - dice
        return loss


class DiceLoss(nn.Module):
    """
        Dice loss, need one hot encode input
        Args:
            weight: An array of shape [num_classes,].
            ignore_index: class index to ignore.
            predict: A tensor of shape [N, C, *].
            target: A tensor of same shape with predict.
            other args pass to BinaryDiceLoss.
        Return:
            same as BinaryDiceLoss.
    """
    def __init__(self, args, weight=None, num_classes=13, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.args = args
        self.weight = weight
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.kwargs = kwargs

    def forward(self, predict, target):
        mask = target != self.ignore_index
        target = target * mask
        target = make_one_hot(self.args, target, self.num_classes)  # (2,1,440,640)->(2,11,440,640)
        target = target * mask

        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)
        predict = predict * mask

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]

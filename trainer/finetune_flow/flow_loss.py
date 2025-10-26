import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowLoss(nn.Module):
    def __init__(self, args):
        super(FlowLoss, self).__init__()
        self.args = args
        self.l1_loss = F.l1_loss

    def forward(self, predict, target, target_valid):  # (2,2,260,346) (2,2,260,346)
        mag = torch.sum(target ** 2, dim=1, keepdim=True).sqrt()
        valid = (target_valid >= 0.5) & (mag < self.args.max_flow)
        valid = valid.repeat(1, 2, 1, 1)
        l1_loss = (predict[valid] - target[valid]).abs().mean()

        return l1_loss

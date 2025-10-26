import torch
import torch.nn as nn

from utils.reshape import resize


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU(inplace=True)):
        super().__init__()
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm_layer = norm_layer(out_channels)
        self.act_layer = act_layer

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.norm_layer(x)
        x = self.act_layer(x)

        return x


class BaseDecodeHead(nn.Module):
    def __init__(self, args, in_channels, channels, out_channels,
                 dropout_ratio=0.1, in_index=-1, input_transform='multiple_select'):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.out_channels = out_channels
        self.sample_mode = args.sample_mode

        self.dropout_ratio = dropout_ratio
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        self.in_index = in_index
        self.input_transform = input_transform

        self.conv_dense = nn.Conv2d(channels, self.out_channels, kernel_size=1)

    def cls_dense(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)

        output = self.conv_dense(feat)

        return output

class PPM(nn.ModuleList):
    """
        Pooling Pyramid Module used in PSPNet.
        Args:
            pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid Module.
            in_channels (int): Input channels.
            channels (int): Channels after modules, before conv_dense.
    """
    def __init__(self, sample_mode, pool_scales, in_channels, channels, **kwargs):
        super().__init__()
        self.pool_scales = pool_scales
        self.in_channels = in_channels
        self.channels = channels
        self.sample_mode = sample_mode

        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(self.in_channels, self.channels, 1, **kwargs)
                )
            )

    def forward(self, x):
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(ppm_out, size=x.size()[2:], mode=self.sample_mode)
            ppm_outs.append(upsampled_ppm_out)

        return ppm_outs


# decode_head
class UPerHead(BaseDecodeHead):
    def __init__(self, pool_scales=(1, 2, 3, 6), in_index=-1, **kwargs):
        super().__init__(**kwargs)
        self.in_index = in_index

        # PSP Module
        self.psp_modules = PPM(self.sample_mode, pool_scales, self.in_channels[-1], self.channels)

        self.psp_bottleneck = ConvModule(self.in_channels[-1] + len(pool_scales) * self.channels,
                                     self.channels, 3, padding=1)

        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(in_channels, self.channels, 1)
            fpn_conv = ConvModule(self.channels, self.channels, 3, padding=1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(len(self.in_channels) * self.channels,
                                         self.channels, 3, padding=1)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.psp_bottleneck(psp_outs)

        return output

    def _forward_feature(self, inputs):
        """
            Forward function for feature maps before classifying each pixel with ``self.cls_dense`` function.
            Args:
                inputs (list[Tensor]): List of multi-level img features.
            Returns:
                feats (Tensor): A tensor of shape (batch_size, self.channels, H, W) which is feature map for last layer of decoder head.
        """
        # build laterals
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(laterals[i], size=prev_shape, mode=self.sample_mode)

        # build outputs
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(fpn_outs[i], size=fpn_outs[0].shape[2:], mode=self.sample_mode)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)

        return feats

    def forward(self, inputs):  # inputs: (2,128,56,56) (2,256,28,28) (2,384,14,14) (2,384,6,6)
        output = self._forward_feature(inputs)  # (2,384,56,56)
        output = self.cls_dense(output)

        return output  # (2,11,56,56)


# auxiliary_head
class FCNHead(BaseDecodeHead):
    """
        Fully Convolution Networks for Semantic Segmentation used in FCNNet.
        Args:
            num_convs (int): Number of convs in the head. Default: 2.
            kernel_size (int): The kernel size for convs in the head. Default: 3.
            concat_input (bool): Whether concat the input and output of convs before classification layer.
            dilation (int): The dilation rate for convs in the head. Default: 1.
    """
    def __init__(self, num_convs=2, kernel_size=3, concat_input=True, dilation=1, **kwargs):
        super().__init__(**kwargs)
        self.concat_input = concat_input

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(ConvModule(self.in_channels, self.channels, kernel_size=kernel_size, padding=conv_padding))
        for i in range(num_convs - 1):
            convs.append(ConvModule(self.channels, self.channels, kernel_size=kernel_size, padding=conv_padding))
        self.convs = nn.Sequential(*convs)

        if concat_input:
            self.conv_cat = ConvModule(self.in_channels + self.channels, self.channels, kernel_size=kernel_size,
                                       padding=kernel_size // 2)

    def _forward_feature(self, inputs):
        """
            Forward function for feature maps before classifying each pixel with ``self.cls_dense`` function.
            Args:
                inputs (list[Tensor]): List of multi-level img features.
            Returns:
                feats (Tensor): A tensor of shape (batch_size, self.channels, H, W) which is feature map for last layer of decoder head.
        """
        x = inputs[self.in_index]
        feats = self.convs(x)
        if self.concat_input:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))

        return feats  # (2,256,14,14)

    def forward(self, inputs):
        output = self._forward_feature(inputs)  # (2,256,14,14)

        output = self.cls_dense(output)

        return output  # (2,11,14,14)


def finetune_decode_head_extend_small(args, **kwargs):
    model = UPerHead(
        args=args, in_channels=[128, 256, 384, 384], channels=384,
        in_index=[0, 1, 2, 3], pool_scales=(1, 2, 3, 6), **kwargs
    )
    return model

def finetune_decode_head_extend_small_swin(args, **kwargs):
    model = UPerHead(
        args=args, in_channels=[96, 192, 384, 768], channels=384,
        in_index=[0, 1, 2, 3], pool_scales=(1, 2, 3, 6), **kwargs
    )
    return model

def finetune_decode_head_extend_base(args, **kwargs):
    model = UPerHead(
        args=args, in_channels=[256, 384, 768, 768], channels=384,
        in_index=[0, 1, 2, 3], pool_scales=(1, 2, 3, 6), **kwargs
    )
    return model

def finetune_decode_head_small(args, **kwargs):
    model = UPerHead(
        args=args, in_channels=[384, 384, 384, 384], channels=384,
        in_index=[0, 1, 2, 3], pool_scales=(1, 2, 3, 6), **kwargs
    )
    return model

def finetune_decode_head_base(args, **kwargs):
    model = UPerHead(
        args=args, in_channels=[768, 768, 768, 768], channels=384,
        in_index=[0, 1, 2, 3], pool_scales=(1, 2, 3, 6), **kwargs
    )
    return model

def finetune_auxiliary_head_small(args, **kwargs):
    model = FCNHead(
        args=args, in_channels=384, channels=256,
        in_index=2, num_convs=1, kernel_size=3, concat_input=False, **kwargs
    )
    return model

def finetune_auxiliary_head_base(args, **kwargs):
    model = FCNHead(
        args=args, in_channels=768, channels=256,
        in_index=2, num_convs=1, kernel_size=3, concat_input=False, **kwargs
    )
    return model

def finetune_auxiliary_head_small_swin(args, **kwargs):
    model = FCNHead(
        args=args, in_channels=384, channels=256,
        in_index=2, num_convs=1, kernel_size=3, concat_input=False, **kwargs
    )
    return model

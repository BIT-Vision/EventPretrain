# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ELECTRA https://github.com/google-research/electra
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import json
import pdb


def param_groups_lrd(args, model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75, layer_grafted=False):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}
    if args.backbone_type == "vit" or args.backbone_type == "vit_mem" or args.backbone_type == "vit_ecdp":
        num_layers = len(model.backbone.vit_block)
    elif args.backbone_type == "convvit":
        num_layers = len(model.backbone.vit_block) + 2
    elif args.backbone_type == "swin":
        num_layers = len(model.backbone.swin_block)
    elif args.backbone_type == "swin_ecddp":
        num_layers = len(model.backbone.layers)
    else:
        num_layers = 0

    if layer_grafted:
        layer_scales = [0.01, 0.1, 1]
    else:
        layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit(n, num_layers, args.backbone_type, layer_grafted)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    if layer_decay != 1 or layer_grafted:
        print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers, backbone_type, layer_grafted):
    if layer_grafted:
        if name.startswith('backbone.pos_embed') or name.startswith('backbone.patch_embed'):
            return 0
        elif name.startswith('backbone.conv_block1') or name.startswith('backbone.conv_block2'):
            return 0
        elif name.startswith('backbone.vit_block'):
            block_id = int(name.split('.')[2])
            if block_id // 4 == 0:
                return 0
            elif block_id // 4 == 1:
                return 1
            else:
                return 2
        else:
            return 2
    else:
        if name.startswith('backbone.pos_embed') or name.startswith('backbone.patch_embed'):
            return 0
        elif name.startswith('backbone.vit_block'):
            if backbone_type == 'vit' or backbone_type == 'vit_mem' or backbone_type == 'vit_ecdp':
                return int(name.split('.')[2]) + 1
            elif backbone_type == 'convvit':
                return int(name.split('.')[2]) + 3
        elif name.startswith('backbone.conv_block1'):
            return 1
        elif name.startswith('backbone.conv_block2'):
            return 2
        else:
            return num_layers

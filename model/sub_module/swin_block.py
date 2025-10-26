import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


def get_coordinates(h, w, device='cpu'):
    coords_h = torch.arange(h, device=device)
    coords_w = torch.arange(w, device=device)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    return coords


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        # NOTE: the index is not used at pretraining and is kept for compatibility
        coords = get_coordinates(*window_size)  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, pos_idx=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # projection
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B_, nH, N, N

        # relative position bias
        assert pos_idx.dim() == 3, f"Expect the pos_idx/mask to be a 3-d tensor, but got{pos_idx.dim()}"
        rel_pos_mask = torch.masked_fill(torch.ones_like(mask), mask=mask.bool(), value=0.0)
        pos_idx_m = torch.masked_fill(pos_idx, mask.bool(), value=0).view(-1)
        relative_position_bias = self.relative_position_bias_table[pos_idx_m].view(
            -1, N, N, self.num_heads)  # nW, Wh*Ww, Wh*Ww,nH
        relative_position_bias = relative_position_bias * rel_pos_mask.view(-1, N, N, 1)

        nW = relative_position_bias.shape[0]
        relative_position_bias = relative_position_bias.permute(0, 3, 1, 2).contiguous()  # nW, nH, Wh*Ww, Wh*Ww
        attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + relative_position_bias.unsqueeze(0)

        # attention mask
        attn = attn + mask.view(1, nW, 1, N, N)
        attn = attn.view(B_, self.num_heads, N, N)

        # normalization
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # aggregation
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn  # (128,3,49,49) (32,6,49,49) (8,12,49,49) (2,24,49,49)

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, coords_prev, mask_prev):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        # gather patches lie within 2x2 local window
        mask = mask_prev.reshape(H // 2, 2, W // 2, 2).permute(0, 2, 1, 3).reshape(-1)
        coords = get_coordinates(H, W, device=x.device).reshape(2, -1).permute(1, 0)
        coords = coords.reshape(H // 2, 2, W // 2, 2, 2).permute(0, 2, 1, 3, 4).reshape(-1, 2)
        coords_vis_local = coords[mask].reshape(-1, 2)
        coords_vis_local = coords_vis_local[:, 0] * H + coords_vis_local[:, 1]
        idx_shuffle = torch.argsort(torch.argsort(coords_vis_local))

        x = torch.index_select(x, 1, index=idx_shuffle)
        x = x.reshape(B, L // 4, 4, C)  # 768 192 48  # 1536 384 96
        # row-first order to column-first order
        # make it compatible with Swin (https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L342)
        x = torch.cat([x[:, :, 0], x[:, :, 2], x[:, :, 1], x[:, :, 3]], dim=-1)

        # merging by a linear layer
        x = self.norm(x)
        x = self.reduction(x)

        mask_new = mask_prev.view(1, H // 2, 2, W // 2, 2).sum(dim=(2, 4))
        # assert torch.unique(mask_new).shape[0] == 2  # mask_ratio == 0: 1 mask_ratio != 0: 0
        mask_new = (mask_new > 0).reshape(1, -1)
        coords_new = get_coordinates(H // 2, W // 2, x.device).reshape(1, 2, -1)
        coords_new = coords_new.transpose(2, 1)[mask_new].reshape(1, -1, 2)
        return x, coords_new, mask_new

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask, rel_pos_idx):
        shortcut = x
        x = self.norm1(x)

        # W-MSA/SW-MSA
        x, attn = self.attn(x, mask=attn_mask, pos_idx=rel_pos_idx)  # B*nW, N_vis, C

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, attn

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"


def knapsack(W, wt):
    '''Args:
        W (int): capacity
        wt (tuple[int]): the numbers of elements within each window
    '''
    val = wt
    n = len(val)
    K = [[0 for w in range(W + 1)]
         for i in range(n + 1)]

    # Build table K[][] in bottom up manner
    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif wt[i - 1] <= w:
                K[i][w] = max(val[i - 1]
                              + K[i - 1][w - wt[i - 1]],
                              K[i - 1][w])
            else:
                K[i][w] = K[i - 1][w]

    # stores the result of Knapsack
    res = res_ret = K[n][W]

    # stores the selected indices
    w = W
    idx = []
    for i in range(n, 0, -1):
        if res <= 0:
            break
        # Either the result comes from the top (K[i-1][w])
        # or from (val[i-1] + K[i-1] [w-wt[i-1]]) as in Knapsack table.
        # If it comes from the latter one, it means the item is included.
        if res == K[i - 1][w]:
            continue
        else:
            # This item is included.
            idx.append(i - 1)
            # Since this weight is included, its value is deducted
            res = res - val[i - 1]
            w = w - wt[i - 1]

    return res_ret, idx[::-1]  # make the idx in an increasing order


def group_windows(group_size, num_ele_win):
    '''Greedily apply the DP algorithm to group the elements.
    Args:
        group_size (int): maximal size of the group
        num_ele_win (list[int]): number of visible elements of each window
    Outputs:
        num_ele_group (list[int]): number of elements of each group
        grouped_idx (list[list[int]]): the seleted indeices of each group
    '''
    wt = num_ele_win.copy()
    ori_idx = list(range(len(wt)))
    grouped_idx = []
    num_ele_group = []

    while len(wt) > 0:
        res, idx = knapsack(group_size, wt)
        num_ele_group.append(res)

        # append the selected idx
        selected_ori_idx = [ori_idx[i] for i in idx]
        grouped_idx.append(selected_ori_idx)

        # remaining idx
        wt = [wt[i] for i in range(len(ori_idx)) if i not in idx]
        ori_idx = [ori_idx[i] for i in range(len(ori_idx)) if i not in idx]

    return num_ele_group, grouped_idx


class GroupingModule:
    def __init__(self, window_size, shift_size, group_size=None):
        self.window_size = window_size
        self.shift_size = shift_size
        assert shift_size >= 0 and shift_size < window_size

        self.group_size = group_size or self.window_size ** 2
        self.attn_mask = None
        self.rel_pos_idx = None

    def _get_group_id(self, coords):
        group_id = coords.clone()  # (1,1536,2)
        group_id += (self.window_size - self.shift_size) % self.window_size  # shift_size: 0
        group_id = group_id // self.window_size
        group_id = group_id[0, :, 0] * group_id.shape[1] + group_id[0, :, 1]  # (N_vis, )
        return group_id

    def _get_attn_mask(self, group_id):
        pos_mask = (group_id == -1)
        pos_mask = torch.logical_and(pos_mask[:, :, None], pos_mask[:, None, :])
        gid = group_id.float()
        attn_mask_float = gid.unsqueeze(2) - gid.unsqueeze(1)
        attn_mask = torch.logical_or(attn_mask_float != 0, pos_mask)
        attn_mask_float.masked_fill_(attn_mask, -100.)
        return attn_mask_float

    def _get_rel_pos_idx(self, coords):
        # num_groups, group_size, group_size, 2
        rel_pos_idx = coords[:, :, None, :] - coords[:, None, :, :]
        rel_pos_idx += self.window_size - 1
        rel_pos_idx[..., 0] *= 2 * self.window_size - 1
        rel_pos_idx = rel_pos_idx.sum(dim=-1)
        return rel_pos_idx

    def _prepare_masking(self, coords):
        # coords: (B, N_vis, 2)
        group_id = self._get_group_id(coords)  # (N_vis, )
        attn_mask = self._get_attn_mask(group_id.unsqueeze(0))
        rel_pos_idx = self._get_rel_pos_idx(coords[:1])

        # do not shuffle
        self.idx_shuffle = None
        self.idx_unshuffle = None

        return attn_mask, rel_pos_idx

    def _prepare_grouping(self, coords):
        # find out the elements within each local window
        # coords: (B, N_vis, 2)
        group_id = self._get_group_id(coords)  # (N_vis, )
        idx_merge = torch.argsort(group_id)
        group_id = group_id[idx_merge].contiguous()  # group_id 排序
        exact_win_sz = torch.unique_consecutive(group_id, return_counts=True)[1].tolist()  # 对应数字重复次数

        # group the windows by DP algorithm
        self.group_size = min(self.window_size ** 2, max(exact_win_sz))  # 49 FIXME
        num_ele_group, grouped_idx = group_windows(self.group_size, exact_win_sz)

        # pad the splits
        idx_merge_spl = idx_merge.split(exact_win_sz)
        group_id_spl = group_id.split(exact_win_sz)
        shuffled_idx, attn_mask = [], []
        for num_ele, gidx in zip(num_ele_group, grouped_idx):
            pad_r = self.group_size - num_ele
            # shuffle indices: (group_size)
            sidx = torch.cat([idx_merge_spl[i] for i in gidx], dim=0)
            shuffled_idx.append(F.pad(sidx, (0, pad_r), value=-1))
            # attention mask: (group_size)
            amask = torch.cat([group_id_spl[i] for i in gidx], dim=0)
            attn_mask.append(F.pad(amask, (0, pad_r), value=-1))

        # shuffle indices: (num_groups * group_size, )
        self.idx_shuffle = torch.cat(shuffled_idx, dim=0)
        # unshuffle indices that exclude the padded indices: (N_vis, )
        self.idx_unshuffle = torch.argsort(self.idx_shuffle)[-sum(num_ele_group):]
        self.idx_shuffle[self.idx_shuffle == -1] = 0  # index_select does not permit negative index

        # attention mask: (num_groups, group_size, group_size)
        attn_mask = torch.stack(attn_mask, dim=0)
        attn_mask = self._get_attn_mask(attn_mask)

        # relative position indices: (num_groups, group_size, group_size)
        coords_shuffled = coords[0][self.idx_shuffle].reshape(-1, self.group_size, 2)
        rel_pos_idx = self._get_rel_pos_idx(coords_shuffled)  # num_groups, group_size, group_size
        rel_pos_mask = torch.ones_like(rel_pos_idx).masked_fill_(attn_mask.bool(), 0)
        rel_pos_idx = rel_pos_idx * rel_pos_mask

        return attn_mask, rel_pos_idx  # (35,49,49) (35,49,49)

    def prepare(self, coords, num_tokens):
        if num_tokens <= 2 * self.window_size ** 2:
            self._mode = 'masking'
            return self._prepare_masking(coords)
        else:
            self._mode = 'grouping'
            return self._prepare_grouping(coords)

    def group(self, x):
        if self._mode == 'grouping':
            self.ori_shape = x.shape
            x = torch.index_select(x, 1, self.idx_shuffle)  # (B, nG*GS, C)
            x = x.reshape(-1, self.group_size, x.shape[-1])  # (B*nG, GS, C)
        return x

    def merge(self, x):
        if self._mode == 'grouping':
            B, N, C = self.ori_shape
            x = x.reshape(B, -1, C)  # (B, nG*GS, C)
            x = torch.index_select(x, 1, self.idx_unshuffle)  # (B, N, C)
        return x


class BasicBlock(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.window_size = window_size
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        else:
            self.shift_size = window_size // 2

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, coords, patch_mask):
        # prepare the attention mask and relative position bias
        group_block = GroupingModule(self.window_size, 0)  # shift_size: 0
        mask, pos_idx = group_block.prepare(coords, num_tokens=x.shape[1])  # coords: x,y坐标
        if self.window_size < min(self.input_resolution):
            group_block_shift = GroupingModule(self.window_size, self.shift_size)  # shift_size: 3
            mask_shift, pos_idx_shift = group_block_shift.prepare(coords, num_tokens=x.shape[1])
        else:
            # do not shift
            group_block_shift = group_block
            mask_shift, pos_idx_shift = mask, pos_idx

        # forward with grouping/masking
        for i, blk in enumerate(self.blocks):
            gblk = group_block if i % 2 == 0 else group_block_shift
            attn_mask = mask if i % 2 == 0 else mask_shift
            rel_pos_idx = pos_idx if i % 2 == 0 else pos_idx_shift
            x = gblk.group(x)
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask, rel_pos_idx)
            else:
                x, attn = blk(x, attn_mask, rel_pos_idx)
            x = gblk.merge(x)

        # patch merging
        if self.downsample is not None:
            x_down, coords_down, patch_mask_down = self.downsample(x, coords, patch_mask)
            # (2,1536,96) (1,1536,2) (1,3136) -> (2,384,192) (1,384,2) (1,784) -> (2,96,384) (1,96,2) (1,196) -> (2,24,768) (1,24,2) (1,49) * 2
            return x, coords, patch_mask, x_down, coords_down, patch_mask_down, attn
        else:
            return x, coords, patch_mask, attn

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, window_size={self.window_size}," \
               f"shift_size={self.shift_size}, depth={self.depth}"

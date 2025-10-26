import torch
import torch.nn.functional as F


def emb2frame(args, emb, chans):  # emb: (b,l,d) -> frame: (b,c,H,W)
    hight = width = int(emb.shape[1] ** .5)
    assert hight * width == emb.shape[1]

    frame = emb.reshape(shape=(emb.shape[0], hight, width, args.patch_size, args.patch_size, chans))
    frame = torch.einsum('bhwpqc->bchpwq', frame)
    frame = frame.reshape(shape=(emb.shape[0], chans, hight * args.patch_size, width * args.patch_size))

    return frame  # (b,1,224,224)

def frame2emb(patch_size, frame):  # frame: (b,c,H,W) -> emb: (b,l,d)
    emb = frame.reshape(shape=(frame.shape[0], frame.shape[1],
                                frame.shape[2] // patch_size, patch_size,
                                frame.shape[2] // patch_size, patch_size))
    emb = torch.einsum('bchpwq->bhwpqc', emb)
    emb = emb.reshape(shape=(emb.shape[0], emb.shape[1] * emb.shape[2],
                               emb.shape[3] * emb.shape[4] * emb.shape[5]))
    return emb  # (b,196,384)

def emb2patch_frame(emb):  # emb: (b,l,c) -> patch_frame: (b,c,h,w)
    hight = width = int(emb.shape[1] ** .5)
    assert hight * width == emb.shape[1]

    patch_frame = emb.reshape(shape=(emb.shape[0], hight, width, emb.shape[-1]))
    patch_frame = torch.einsum('bhwc->bchw', patch_frame)

    return patch_frame  # (b,384,14,14)

def patch_frame2emb(patch_frame):  # patch_frame: (b,c,h,w) -> emb: (b,l,c)
    emb = patch_frame.reshape(shape=(patch_frame.shape[0], patch_frame.shape[1],
                                     patch_frame.shape[2] * patch_frame.shape[3]))
    emb = torch.einsum('bcl->blc', emb)

    return emb  # (b,196,384)

def resize(input, size, scale_factor=None, mode='bilinear', align_corners=None):
    output = F.interpolate(input, size, scale_factor, mode, align_corners)

    return output

def resize_flow(args, input, size, scale_factor=None, mode='bilinear', align_corners=None):
    org_h, org_w = input.shape[-2], input.shape[-1]
    output = F.interpolate(input, size, scale_factor, mode, align_corners)

    new_h, new_w = size[0], size[1]
    output = torch.einsum('bchw->bhwc', output)
    output = output * torch.tensor([new_w / org_w, new_h / org_h]).to(args.device)
    output = torch.einsum('bhwc->bchw', output)

    return output

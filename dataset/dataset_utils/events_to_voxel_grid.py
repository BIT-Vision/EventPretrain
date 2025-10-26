import torch


def events_to_voxel_grid(args, events, size, is_txyp=False):
    height, width = size[0], size[1]

    with torch.no_grad():
        events_tensor = torch.from_numpy(events)
        assert (events_tensor.shape[1] == 4)

        voxel_grid = torch.zeros(args.num_bins, height, width, dtype=torch.float32).flatten()
        if events_tensor.device.type == "cuda":
            voxel_grid = voxel_grid.cuda()

        if is_txyp:
            first_stamp = events_tensor[0, 0]
            last_stamp = events_tensor[-1, 0]
        else:
            first_stamp = events_tensor[0, 2]
            last_stamp = events_tensor[-1, 2]

        deltaT = last_stamp - first_stamp

        if deltaT == 0:
            deltaT = 1.0

        if is_txyp:
            xs = events_tensor[:, 1].long()
            ys = events_tensor[:, 2].long()
            ts = (args.num_bins - 1) * (events_tensor[:, 0] - first_stamp) / deltaT  # 0 -> 4
        else:
            xs = events_tensor[:, 0].long()
            ys = events_tensor[:, 1].long()
            ts = (args.num_bins - 1) * (events_tensor[:, 2] - first_stamp) / deltaT  # 0 -> 4
        ps = events_tensor[:, 3].float()
        ps[ps == 0] = -1  # polarity should be +1 / -1

        tis = torch.floor(ts)
        tis_long = tis.long()
        dts = ts - tis
        vals_left = ps * (1.0 - dts.float())
        vals_right = ps * dts.float()

        valid_indices = tis < args.num_bins
        valid_indices &= tis >= 0
        voxel_grid.index_add_(dim=0,
                              index=xs[valid_indices] + ys[valid_indices]
                                    * width + tis_long[valid_indices] * width * height,
                              source=vals_left[valid_indices])

        valid_indices = (tis + 1) < args.num_bins
        valid_indices &= tis >= 0

        voxel_grid.index_add_(dim=0,
                              index=xs[valid_indices] + ys[valid_indices] * width
                                    + (tis_long[valid_indices] + 1) * width * height,
                              source=vals_right[valid_indices])

        voxel_grid = voxel_grid.view(args.num_bins, height, width)

    return voxel_grid

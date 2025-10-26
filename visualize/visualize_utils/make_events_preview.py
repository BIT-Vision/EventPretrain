import numpy as np
import copy

import torch


def make_events_preview(events_voxel_grid):
    if events_voxel_grid.shape[0] == 5 or events_voxel_grid.shape[0] == 6:
        # events: [C x H x W]
        sum_events = torch.sum(events_voxel_grid, dim=0).numpy()

        # positive events: blue, negative events: red
        event_frame = np.ones((sum_events.shape[0], sum_events.shape[1], 3),
                               dtype=np.uint8) * 255  # (224, 224, 3)
        b = event_frame[:, :, 0]
        g = event_frame[:, :, 1]
        r = event_frame[:, :, 2]

        # pos
        b[sum_events > 0] = 0
        g[sum_events > 0] = 0
        r[sum_events > 0] = 255

        # neg
        b[sum_events < 0] = 255
        g[sum_events < 0] = 0
        r[sum_events < 0] = 0

    else:
        if events_voxel_grid.shape[0] == 2:
            events_voxel_grid[1, :, :] = -events_voxel_grid[1, :, :]
        elif events_voxel_grid.shape[0] == 3:
            events_voxel_grid[2, :, :] = -events_voxel_grid[2, :, :]
        else:
            raise ValueError

        # events: [C x H x W]
        sum_events = torch.sum(events_voxel_grid, dim=0).numpy()

        # positive events: blue, negative events: red
        event_frame = np.ones((sum_events.shape[0], sum_events.shape[1], 3),
                              dtype=np.uint8) * 255  # (224, 224, 3)
        b = event_frame[:, :, 0]
        g = event_frame[:, :, 1]
        r = event_frame[:, :, 2]

        # pos
        b[sum_events > 0] = 0
        g[sum_events > 0] = 0
        r[sum_events > 0] = 255

        # neg
        b[sum_events < 0] = 255
        g[sum_events < 0] = 0
        r[sum_events < 0] = 0

    return torch.from_numpy(event_frame)

def make_events_preview_norm(events_voxel_grid):
    # events: [C x H x W]
    if events_voxel_grid.dim() == 3:
        sum_events = torch.sum(events_voxel_grid, dim=0).detach().cpu().numpy()
        sum_events_change = torch.sum(events_voxel_grid, dim=0).detach().cpu().numpy()
    else:
        sum_events = events_voxel_grid.detach().cpu().numpy()
        sum_events_change = events_voxel_grid.detach().cpu().numpy()

    if len(sum_events_change[sum_events >= 0]) > 0:
        sum_events_change[sum_events >= 0] = ((sum_events_change[sum_events >= 0] - sum_events_change[sum_events >= 0].min()) / (
            sum_events_change[sum_events >= 0].max() - sum_events_change[sum_events >= 0].min())) * 0.5 + 0.5
    if len(sum_events_change[sum_events <= 0]) > 0:
        sum_events_change[sum_events <= 0] = (sum_events_change[sum_events <= 0] - sum_events_change[sum_events <= 0].min()) / (
            sum_events_change[sum_events <= 0].max() - sum_events_change[sum_events <= 0].min()) * 0.5
    if len(sum_events_change[sum_events == 0]) > 0:
        sum_events_change[sum_events == 0] = 0.5

    return torch.from_numpy(sum_events_change)

def make_events_preview_rgb_norm(events_voxel_grid):
    # events: [C x H x W]
    if events_voxel_grid.shape[0] == 5 or events_voxel_grid.shape[0] == 6:
        # events: [C x H x W]
        sum_events = torch.sum(events_voxel_grid, dim=0).numpy()

    else:
        if events_voxel_grid.shape[0] == 2:
            events_voxel_grid[1, :, :] = -events_voxel_grid[1, :, :]
        elif events_voxel_grid.shape[0] == 3:
            events_voxel_grid[2, :, :] = -events_voxel_grid[2, :, :]
        else:
            raise ValueError

        # events: [C x H x W]
        sum_events = torch.sum(events_voxel_grid, dim=0).numpy()

    sum_events_change = copy.deepcopy(sum_events)

    if len(sum_events_change[sum_events >= 0]) > 0:
        sum_events_change[sum_events >= 0] = ((sum_events_change[sum_events >= 0] - sum_events_change[
            sum_events >= 0].min()) / (sum_events_change[sum_events >= 0].max() - sum_events_change[sum_events >= 0].min()))
    if len(sum_events_change[sum_events <= 0]) > 0:
        sum_events_change[sum_events <= 0] = (sum_events_change[sum_events <= 0] - sum_events_change[
            sum_events <= 0].min()) / (sum_events_change[sum_events <= 0].max() - sum_events_change[sum_events <= 0].min()) - 1
    if len(sum_events_change[sum_events == 0]) > 0:
        sum_events_change[sum_events == 0] = 0

    # positive events: blue, negative events: red
    event_frame = np.ones((sum_events.shape[0], sum_events.shape[1], 3),
                          dtype=np.uint8) * 255  # (224, 224, 3)
    b = event_frame[:, :, 0]
    g = event_frame[:, :, 1]
    r = event_frame[:, :, 2]

    # pos
    b[sum_events_change > 0] = 150 * (1 - sum_events_change[sum_events_change > 0])
    g[sum_events_change > 0] = 150 * (1 - sum_events_change[sum_events_change > 0])
    r[sum_events_change > 0] = 255

    # neg
    b[sum_events_change < 0] = 255
    g[sum_events_change < 0] = 150 * (1 + sum_events_change[sum_events_change < 0])
    r[sum_events_change < 0] = 150 * (1 + sum_events_change[sum_events_change < 0])

    return torch.from_numpy(event_frame)

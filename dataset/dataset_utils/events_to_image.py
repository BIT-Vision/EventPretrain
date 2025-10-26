import numpy as np

import torch


def events_to_image_ecdp(args, events, size, is_txyp=False):
    height, width = size[0], size[1]

    with torch.no_grad():
        events_tensor = torch.from_numpy(events)
        assert (events_tensor.shape[1] == 4)

        pos_events = events_tensor[events_tensor[:, 3] == 1]
        neg_events = events_tensor[events_tensor[:, 3] == 0]
        if len(neg_events) == 0:
            neg_events = events_tensor[events_tensor[:, 3] == -1]

        if is_txyp:
            pos_image = torch.bincount(pos_events[:, 1].long() + pos_events[:, 2].long() * width,
                                       minlength=height * width).reshape(height, width)
            neg_image = torch.bincount(neg_events[:, 1].long() + neg_events[:, 2].long() * width,
                                       minlength=height * width).reshape(height, width)
        else:
            pos_image = torch.bincount(pos_events[:, 0].long() + pos_events[:, 1].long() * width,
                                       minlength=height * width).reshape(height, width)
            neg_image = torch.bincount(neg_events[:, 0].long() + neg_events[:, 1].long() * width,
                                       minlength=height * width).reshape(height, width)

        pos_neg_image = torch.stack([pos_image, neg_image], dim=2)
        pos_neg_image = torch.einsum('hwc->chw', pos_neg_image).float()

    return pos_neg_image


def events_to_image_mem(args, events, size, is_txyp=False):
    height, width = size[0], size[1]

    with torch.no_grad():
        events_tensor = torch.from_numpy(events)
        assert (events_tensor.shape[1] == 4)

        pos_events = events_tensor[events_tensor[:, 3] == 1]
        neg_events = events_tensor[events_tensor[:, 3] == 0]
        if len(neg_events) == 0:
            neg_events = events_tensor[events_tensor[:, 3] == -1]

        if is_txyp:
            pos_image = torch.bincount(pos_events[:, 1].long() + pos_events[:, 2].long() * width,
                                       minlength=height * width).reshape(height, width)
            neg_image = torch.bincount(neg_events[:, 1].long() + neg_events[:, 2].long() * width,
                                       minlength=height * width).reshape(height, width)
        else:
            pos_image = torch.bincount(pos_events[:, 0].long() + pos_events[:, 1].long() * width,
                                       minlength=height * width).reshape(height, width)
            neg_image = torch.bincount(neg_events[:, 0].long() + neg_events[:, 1].long() * width,
                                       minlength=height * width).reshape(height, width)

        tss_image = torch.zeros((height, width), dtype=torch.int64)
        pos_neg_image = torch.stack([pos_image, tss_image, neg_image], dim=2)
        pos_neg_image = torch.einsum('hwc->chw', pos_neg_image).float()

    return pos_neg_image


def remove_hot_pixel_mem(hist, num_stds=10):
    hist_flatten = hist[0::2, :, :].flatten()  # (2*H*W)

    mean, std = torch.mean(hist[0::2, :, :]), torch.std(hist[0::2, :, :])
    threshold_filter = mean + num_stds * std
    hot_pixel_inds = torch.atleast_1d(torch.squeeze(torch.argwhere(hist_flatten > threshold_filter)))

    hot_pixel_index_2d = np.asarray(np.unravel_index(hot_pixel_inds, hist.shape)).T
    hist[0::2, hot_pixel_index_2d[:, 1], hot_pixel_index_2d[:, 2]] = 0

    return hist

def events_to_EvRep(event_xs, event_ys, event_timestamps, event_polarities, resolution=(320, 240)):
    """
    Convert event-based data into an EvRep representation using more efficient matrix operations.

    :param event_xs: Array of x-coordinates of events
    :param event_ys: Array of y-coordinates of events
    :param event_timestamps: Array of timestamps of events
    :param event_polarities: Array of polarities of events (assumed to be from {0, 1})
    :param resolution: Tuple (width, height) representing the resolution of the output grid
    :return: An EvRep representation of shape (3, height, width)
    """
    width, height = resolution

    # Initialize the three channels: spatial, polarity, and temporal
    E_C = np.zeros((height, width), dtype=np.int32)  # Event spatial channel
    E_I = np.zeros((height, width), dtype=np.int32)  # Event polarity channel
    E_T_sum = np.zeros((height, width), dtype=np.float32)  # For sum of timestamp deltas
    E_T_sq_sum = np.zeros((height, width), dtype=np.float32)  # For sum of squared deltas

    # Normalize event polarities to {-1, 1}
    event_polarities = np.where(event_polarities == 0, -1, event_polarities)

    # Bin events into the grid (spatial and polarity channels)
    np.add.at(E_C, (event_ys, event_xs), 1)  # Count of events at each pixel
    np.add.at(E_I, (event_ys, event_xs), event_polarities)  # Net polarity of events at each pixel

    # Sort events by pixel for temporal statistics (approximation using binning)
    sort_indices = np.lexsort((event_timestamps, event_ys, event_xs))
    sorted_xs = event_xs[sort_indices]
    sorted_ys = event_ys[sort_indices]
    sorted_timestamps = event_timestamps[sort_indices]

    # Calculate deltas for consecutive events at each pixel
    delta_timestamps = np.diff(sorted_timestamps, prepend=sorted_timestamps[0])

    # Efficient temporal processing by binning consecutive deltas
    np.add.at(E_T_sum, (sorted_ys, sorted_xs), delta_timestamps)
    np.add.at(E_T_sq_sum, (sorted_ys, sorted_xs), delta_timestamps ** 2)

    # Calculate standard deviation for temporal channel
    E_T_counts = E_C.clip(min=1)  # Avoid division by zero
    delta_mean = E_T_sum / E_T_counts
    E_T = np.sqrt(np.maximum((E_T_sq_sum / E_T_counts) - delta_mean ** 2, 0))
    E_T = E_T.clip(max=1000)

    # Stack the channels to form the EvRep representation
    EvRep = np.stack([E_C, E_I, E_T], axis=0)

    return EvRep

import numpy as np


# events
def get_random_index(args, events, is_train, seed=None):  # 均为 xytp
    if seed is not None:
        np.random.seed(seed)

    if is_train:
        fix_events_num = args.fix_events_num
    else:
        fix_events_num = args.val_fix_events_num
    if events.shape[0] > fix_events_num:
        start_index = np.random.randint(0, events.shape[0] - fix_events_num)
        end_index = start_index + fix_events_num
    else:
        start_index = 0
        end_index = events.shape[0]

    return start_index, end_index  # 左闭右开

def events_reshape(events, sensor_w, sensor_h, input_w, input_h):
    events[:, 0] *= (input_w / sensor_w)
    events[:, 1] *= (input_h / sensor_h)

    return events

def erase_and_add_events(args, events, size=None):
    sensor_h, sensor_w = size[0], size[1]

    if int(0.01 * events.shape[0]) > 0:
        # erase events
        erase_num = np.random.randint(int(0.001 * events.shape[0]), int(0.01 * events.shape[0]))
        erase_index = np.random.choice(np.arange(events.shape[0]), size=erase_num, replace=False)
        erase_index = np.sort(erase_index)

        # add correlated events
        add_num = np.random.randint(int(0.001 * events.shape[0]), int(0.01 * events.shape[0]))
        add_correlated_events = np.concatenate((
            events[:, [0]] + np.random.normal(0, 1.5, size=(events.shape[0], 1)),
            events[:, [1]] + np.random.normal(0, 1.5, size=(events.shape[0], 1)),
            events[:, [2]] + np.random.normal(0, 0.001, size=(events.shape[0], 1)),
            events[:, [3]]
        ), 1)

        add_index = np.random.choice(np.arange(add_correlated_events.shape[0]), size=add_num, replace=False)
        add_events = add_correlated_events[add_index]
        add_events[:, 0] = np.clip(add_events[:, 0], 0, sensor_w - 1)
        add_events[:, 1] = np.clip(add_events[:, 1], 0, sensor_h - 1)

        events = np.delete(events, erase_index, axis=0)
        events = np.concatenate((events, add_events))
        events = events[events[:, 2].argsort()]  # 按时间顺序排序

    return events

def add_noise_events(args, events, size):
    sensor_h, sensor_w = size[0], size[1]

    # add correlated events
    add_num = np.random.randint(int(0.1 * events.shape[0]), int(0.5 * events.shape[0]))
    add_correlated_events = np.concatenate((
        np.random.randint(0, sensor_w, size=(events.shape[0], 1)),
        np.random.randint(0, sensor_h, size=(events.shape[0], 1)),
        np.random.uniform(events[0, 2], events[-1, 2], size=(events.shape[0], 1)),
        np.random.randint(0, 2, size=(events.shape[0], 1))
    ), 1)

    add_index = np.random.choice(np.arange(add_correlated_events.shape[0]), size=add_num, replace=False)
    add_events = add_correlated_events[add_index]
    add_events[:, 0] = np.clip(add_events[:, 0], 0, sensor_w - 1)
    add_events[:, 1] = np.clip(add_events[:, 1], 0, sensor_h - 1)

    events = np.concatenate((events, add_events))
    events = events[events[:, 2].argsort()]  # 按时间顺序排序

    return events

# augment
def events_augment(args, events, size, seed=None):
    if seed is not None:
        np.random.seed(seed)

    events = erase_and_add_events(args, events, size=size)

    return events

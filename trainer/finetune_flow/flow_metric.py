import torch


def flow_compute_mag(matrix, mask=None):
    mag = torch.sum(matrix ** 2, dim=1).sqrt()  # (2,260,346)

    if mask is None:
        return mag
    else:
        return mag[mask == 1]

def flow_compute_epe(predict, target, mask=None):
    epe = flow_compute_mag(predict - target, mask=mask)  # (2,260,346)

    return epe

def flow_compute_aee(epe):
    epe = epe.view(-1)  # (179920)
    aee = epe.mean()

    return aee

def flow_compute_outlier(epe, mag):
    epe = epe.view(-1)  # (179920)
    mag = mag.view(-1)  # (179920)

    outlier = ((epe > 3.0) & ((epe / mag) > 0.05)).float().mean() * 100

    return outlier

def flow_compute_aee_outlier(predict, target, mask=None):
    epe = flow_compute_epe(predict, target, mask=mask)  # (2,260,346)
    mag = flow_compute_mag(target, mask=mask)  # (2,260,346)

    aee = flow_compute_aee(epe)
    outlier = flow_compute_outlier(epe, mag)  # (179920)

    return aee, outlier

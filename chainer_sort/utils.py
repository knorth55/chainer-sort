from __future__ import division

import numpy as np


def bbox2z_bbox(bbox):
    if bbox.shape[1] != 4:
        raise ValueError

    y_center = (bbox[:, 2] + bbox[:, 0]) / 2
    x_center = (bbox[:, 3] + bbox[:, 1]) / 2
    size = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
    ratio = (bbox[:, 2] - bbox[:, 0]) / (bbox[:, 3] - bbox[:, 1])
    z_bbox = np.concatenate(
        (y_center[:, None], x_center[:, None],
         size[:, None], ratio[:, None]),
        axis=1)
    return z_bbox


def z_bbox2bbox(z_bbox):
    if z_bbox.shape[1] != 4:
        raise ValueError

    height = np.sqrt(z_bbox[:, 2] * z_bbox[:, 3])
    width = z_bbox[:, 2] / height
    y_min = z_bbox[:, 0] - (height / 2)
    x_min = z_bbox[:, 1] - (width / 2)
    y_max = z_bbox[:, 0] + (height / 2)
    x_max = z_bbox[:, 1] + (width / 2)
    bbox = np.concatenate(
        (y_min[:, None], x_min[:, None],
         y_max[:, None], x_max[:, None]),
        axis=1)
    return bbox

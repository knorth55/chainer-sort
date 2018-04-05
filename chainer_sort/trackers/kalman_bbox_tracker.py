# this code is modified from original SORT code
# https://github.com/abewley/sort


import numpy as np

from filterpy.kalman import KalmanFilter

from chainer_sort.utils import bbox2z_bbox
from chainer_sort.utils import z_bbox2bbox


class KalmanBboxTracker(object):

    def __init__(self, bbox):
        if len(bbox) > 1:
            raise ValueError

        self.filter = KalmanFilter(dim_x=7, dim_z=4)
        self.filter.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0],
             [0, 1, 0, 0, 0, 1, 0],
             [0, 0, 1, 0, 0, 0, 1],
             [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 1]])
        self.filter.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0]])

        self.filter.R[2:, 2:] *= 10.
        self.filter.P[4:, 4:] *= 1000.
        self.filter.P *= 10.
        self.filter.Q[-1, -1] *= 0.01
        self.filter.Q[4:, 4:] *= 0.01
        self.filter.x[:4] = bbox2z_bbox(bbox).T

        self.time_since_update = 0
        self.history = []
        # self.hits = 0
        self.hit_streak = 0
        # self.age = 0

    def get_state(self):
        bbox = z_bbox2bbox(self.filter.x[:4].T)
        return bbox

    def update(self, bbox):
        if len(bbox) > 1:
            raise ValueError

        self.time_since_update = 0
        self.history = []
        # self.hits += 1
        self.hit_streak += 1
        self.filter.update(bbox2z_bbox(bbox).T)

    def predict(self):
        if (self.filter.x[2] + self.filter.x[6]) <= 0:
            self.filter.x[6] = 0.
        self.filter.predict()
        # self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        bbox = self.get_state()
        self.history.append(bbox)
        return bbox

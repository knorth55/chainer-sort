# Modified work by Shingo Kitagawa (@knorth55)
#
# Original works of SORT from https://github.com/abewley/sort
# ---------------------------------------------------------------------
# Copyright (C) 2016 Alex Bewley alex@dynamicdetection.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# ---------------------------------------------------------------------


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
        self.hit_streak = 0

    def get_state(self):
        bbox = z_bbox2bbox(self.filter.x[:4].T)
        return bbox

    def update(self, bbox):
        if len(bbox) > 1:
            raise ValueError

        self.time_since_update = 0
        self.history = []
        self.hit_streak += 1
        self.filter.update(bbox2z_bbox(bbox).T)

    def predict(self):
        if (self.filter.x[2] + self.filter.x[6]) <= 0:
            self.filter.x[6] = 0.
        self.filter.predict()
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        bbox = self.get_state()
        self.history.append(bbox)
        return bbox

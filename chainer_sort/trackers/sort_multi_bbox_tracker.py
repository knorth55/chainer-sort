import numpy as np

from chainer_sort.trackers import KalmanBboxTracker
from chainer_sort.utils import iou_linear_assignment


class SORTMultiBboxTracker(object):

    def __init__(self, max_age=1, min_hit_streak=3):
        self.max_age = max_age
        self.min_hit_streak = min_hit_streak
        self.trackers = []
        self.frame_count = 0
        self.tracker_num = 0

    def update(self, det_bboxes):
        self.frame_count += 1
        pred_bboxes = []
        valid_trackers = []
        for tracker in self.trackers:
            pred_bbox = tracker.predict()[0]
            if np.all(~np.isnan(pred_bbox)) or np.all(~np.isinf(pred_bbox)):
                valid_trackers.append(tracker)
                pred_bboxes.append(pred_bbox)
        self.trackers = valid_trackers
        if len(pred_bboxes) > 0:
            pred_bboxes = np.concatenate(pred_bboxes)
        else:
            pred_bboxes = np.array(pred_bboxes).reshape((-1, 4))
        assert len(self.trackers) == len(pred_bboxes)

        matched_det_indices, matched_pred_indices = iou_linear_assignment(
            det_bboxes, pred_bboxes)

        # update matched trackers
        for i, tracker in enumerate(self.trackers):
            matched_det_index = matched_det_indices[matched_pred_indices == i]
            tracker.update(det_bboxes[matched_det_index, :])

        # create new trackers for unmatched detections
        for det_index, det_bbox in enumerate(det_bboxes):
            if det_index not in matched_det_indices:
                tracker = KalmanBboxTracker(det_bbox[None])
                tracker.id = self.tracker_num
                self.tracker_num += 1
                self.trackers.append(tracker)

        new_trackers = []
        res_bboxes = []
        res_inst_id = []
        for tracker in self.trackers:
            res_bbox = tracker.get_state()[0]
            if tracker.time_since_update < 1 \
                and (tracker.hit_streak >= self.min_hit_streak
                     or self.frame_count <= self.min_hit_streak):
                res_bboxes.append(res_bbox)
                res_inst_id.append(tracker.id)
            if tracker.time_since_update <= self.max_age:
                new_trackers.append(tracker)
        self.trackers = new_trackers
        res_bboxes = np.array(res_bboxes, dtype=float).reshape((-1, 4))
        res_inst_id = np.array(res_inst_id, dtype=int)
        return res_bboxes, res_inst_id

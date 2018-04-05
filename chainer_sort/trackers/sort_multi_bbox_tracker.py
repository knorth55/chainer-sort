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
        self.inst_ids = []

    def update(self, det_bboxes):
        self.frame_count += 1
        pred_bboxes = []
        valid_trackers = []
        for i, tracker in enumerate(self.trackers):
            pred_bbox = tracker.predict()
            if np.all(~np.isnan(pred_bbox)) and np.all(~np.isinf(pred_bbox)):
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
        # create new trackers for unmatched detections
        new_trackers = []
        trk_det_indices = []
        trk_bboxes = []
        trk_inst_ids = []
        for det_index, det_bbox in enumerate(det_bboxes):
            # matched
            if det_index in matched_det_indices:
                pred_index = matched_pred_indices[
                    matched_det_indices == det_index]
                tracker = self.trackers[int(pred_index)]
                tracker.update(det_bbox[None])
            # not matched
            else:
                tracker = KalmanBboxTracker(det_bbox[None])
                tracker.id = self.tracker_num
                self.tracker_num += 1

            if tracker.time_since_update < 1 \
                and (tracker.hit_streak >= self.min_hit_streak
                     or self.frame_count <= self.min_hit_streak):
                trk_det_indices.append(det_index)
                trk_bbox = tracker.get_state()[0]
                trk_bboxes.append(trk_bbox)
                if tracker.id in self.inst_ids:
                    trk_inst_id = self.inst_ids.index(tracker.id)
                else:
                    self.inst_ids.append(tracker.id)
                    trk_inst_id = len(self.inst_ids)
                trk_inst_ids.append(trk_inst_id)

            if tracker.time_since_update <= self.max_age:
                new_trackers.append(tracker)
        self.trackers = new_trackers

        trk_det_indices = np.array(trk_det_indices, dtype=int)
        trk_bboxes = np.array(trk_bboxes, dtype=float).reshape((-1, 4))
        trk_inst_ids = np.array(trk_inst_ids, dtype=int)
        return trk_det_indices, trk_bboxes, trk_inst_ids

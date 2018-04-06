import numpy as np

import chainer

from chainer_sort.trackers import SORTMultiBboxTracker


class SORTMultiObjectTracking(object):

    def __init__(self, dectector,
                 detector_label_names, tracking_label_names=None):
        self.bbox_tracker = SORTMultiBboxTracker()
        self.detector = dectector
        self.detector_label_names = detector_label_names
        self.tracking_label_names = tracking_label_names

    def predict(self, img):
        if len(img) > 1:
            raise ValueError
        bboxes, labels, scores = self.detector.predict(img)
        bbox, label, score = bboxes[0], labels[0], scores[0]
        bbox = chainer.cuda.to_cpu(bbox)
        label = chainer.cuda.to_cpu(label)
        score = chainer.cuda.to_cpu(score)

        det_bbox = []
        det_label = []
        det_score = []
        for bb, lbl, sc in zip(bbox, label, score):
            if self.detector_label_names[lbl] in self.tracking_label_names:
                det_bbox.append(bb[None])
                det_label.append(lbl)
                det_score.append(sc)
        if len(det_bbox) > 0:
            det_bbox = np.concatenate(det_bbox)
        else:
            det_bbox = np.array(det_bbox).reshape((0, 4))
        det_label = np.array(det_label)
        det_score = np.array(det_score)

        trk_indices, trk_bbox, trk_inst_id = self.bbox_tracker.update(det_bbox)
        trk_label = det_label[trk_indices]
        trk_score = det_score[trk_indices]
        return trk_bbox[None], trk_label[None], \
            trk_score[None], trk_inst_id[None]

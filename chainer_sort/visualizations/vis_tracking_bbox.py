from __future__ import division

import numpy as np

from chainercv.visualizations.vis_image import vis_image


def _default_cmap(label):
    """Color map used in PASCAL VOC"""
    r, g, b = 0, 0, 0
    i = label
    for j in range(8):
        if i & (1 << 0):
            r |= 1 << (7 - j)
        if i & (1 << 1):
            g |= 1 << (7 - j)
        if i & (1 << 2):
            b |= 1 << (7 - j)
        i >>= 3
    return r, g, b


def vis_tracking_bbox(
        img, bbox, inst_id, label=None, score=None,
        label_names=None, alpha=1.0, ax=None):

    from matplotlib import pyplot as plot
    ax = vis_image(img, ax=ax)

    assert len(bbox) == len(inst_id)
    if len(bbox) == 0:
        return ax

    for i, (bb, inst_i) in enumerate(zip(bbox, inst_id)):
        bb = np.round(bb).astype(np.int32)
        y_min, x_min, y_max, x_max = bb
        color = np.array(_default_cmap(inst_i + 1)) / 255.

        ax.add_patch(plot.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            fill=False, edgecolor=color, linewidth=3))

        caption = []
        caption.append('{}'.format(inst_i))

        if label is not None and label_names is not None:
            lb = label[i]
            if not (0 <= lb < len(label_names)):
                raise ValueError('No corresponding name is given')
            caption.append(label_names[lb])
        if score is not None:
            sc = score[i]
            caption.append('{:.2f}'.format(sc))

        ax.text((x_max + x_min) / 2, y_min,
                ': '.join(caption),
                style='italic',
                bbox={'facecolor': color, 'alpha': alpha},
                fontsize=8, color='white')

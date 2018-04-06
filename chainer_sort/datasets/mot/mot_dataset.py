import os

import chainer
from chainercv.utils import read_image

from chainer_sort.datasets.mot import mot_utils
from chainer_sort.datasets.mot.mot_utils import mot_map_names
from chainer_sort.datasets.mot.mot_utils import mot_sequence_names


class MOTDataset(chainer.dataset.DatasetMixin):

    def __init__(
        self, data_dir='auto', year='2015', split='train', sequence='c2',
    ):
        if split not in ['train', 'val', 'trainval']:
            raise ValueError(
                'please pick split from \'train\', \'trainval\', \'val\'')

        if data_dir == 'auto':
            data_dir = mot_utils.get_mot(year, split)

        id_list_file = os.path.join(
            data_dir, 'annotations/{0}.txt'.format(split))
        ids = [id_.strip() for id_ in open(id_list_file)]
        if sequence in mot_map_names:
            if (year == '2015' and sequence not in ['c2', 'c3', 'c4']) \
                    or (year == '2016' and sequence != 'c5') \
                    or (year == '2017' and sequence != 'c9'):
                raise ValueError
            sequences = mot_utils.get_sequences(split, sequence)
            self.ids = [id_ for id_ in ids if id_.split('_')[1] in sequences]
        elif sequence.startswith(tuple(mot_sequence_names[year])):
            self.ids = [
                id_ for id_ in ids if id_.split('_')[1].startswith(sequence)
            ]
        else:
            raise ValueError

        self.data_dir = data_dir

        self.id2inst_id, self.id2bbox = None, None
        if split != 'val':
            self.id2inst_id, self.id2bbox = mot_utils.load_gt(
                self.data_dir, self.ids)

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        data_id = self.ids[i]
        split_d, seq_d, frame = data_id.split('_')
        img_file = os.path.join(
            self.data_dir, split_d, seq_d, 'img1/{}.jpg'.format(frame))
        img = read_image(img_file, color=True)
        if self.id2inst_id is None and self.id2bbox is None:
            inst_id, bbox = None, None
        else:
            inst_id = self.id2inst_id[data_id]
            bbox = self.id2bbox[data_id]
        return img, bbox, inst_id

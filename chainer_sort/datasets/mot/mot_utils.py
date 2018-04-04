import numpy as np
import os

from chainer.dataset import download
from chainercv import utils


root = 'pfnet/chainercv/mot'
dev_urls = 'http://motchallenge.net/data/devkit.zip'
urls = {
    '2015': 'http://motchallenge.net/data/2DMOT2015.zip',
}

mot_map_names = ['c2', 'c3', 'c4', 'c5', 'c8', 'c9', 'c10']
mot_sequence_names = {
    '2015': [
        'TUD-Stadtmitte',
        'TUD-Campus',
        'PETS09-S2L1',
        'ETH-Bahnhof',
        'ETH-Sunnyday',
        'ETH-Pedcross2',
        'ADL-Rundle-6',
        'ADL-Rundle-8',
        'KITTI-13',
        'KITTI-17',
        'Venice-2',
        'TUD-Crossing',
        'PETS09-S2L2',
        'ETH-Jelmoli',
        'ETH-Linthescher',
        'ETH-Crossing',
        'AVG-TownCentre',
        'ADL-Rundle-1',
        'ADL-Rundle-3',
        'KITTI-16',
        'KITTI-19',
        'Venice-1',
    ]
}


def load_gt(data_dir, data_ids):
    gt_dict = {}
    id2bbox = {}
    id2inst_id = {}

    for data_id in data_ids:
        split_d, seq_d, img_name = data_id.split('_')
        frame_id = int(img_name)
        if split_d != 'train':
            id2inst_id[data_id] = None
            id2bbox[data_id] = None
            continue
        if seq_d not in gt_dict:
            gt_dict[seq_d] = _load_gt(data_dir, split_d, seq_d)
        if frame_id in gt_dict[seq_d]:
            inst_id, bbox = gt_dict[seq_d][frame_id]
        else:
            inst_id, bbox = np.empty((0, )), np.empty((0, 4))
        id2inst_id[data_id] = inst_id
        id2bbox[data_id] = bbox
    return id2inst_id, id2bbox


def _load_gt(data_dir, split_d, seq_d):
    gt_path = os.path.join(data_dir, split_d, seq_d, 'gt/gt.txt')

    gt_dict = {}
    gt_data = [map(float, x.split(',')[:6]) for x in open(gt_path).readlines()]
    gt_data = np.array(gt_data)
    for frame_id in np.unique(gt_data[:, 0].astype(np.int32)):
        gt_d = gt_data[gt_data[:, 0] == frame_id]
        inst_id = gt_d[:, 1].astype(np.int32)
        bbox = gt_d[:, 2:].astype(np.float32)
        bbox[:, 2:4] += bbox[:, :2]
        bbox = bbox[:, [1, 0, 3, 2]]
        gt_dict[frame_id] = (inst_id, bbox)
    return gt_dict


def get_sequence_map(split, map_name):
    if split == 'train':
        splits = ['train']
    elif split == 'val':
        splits = ['test']
    elif split == 'trainval':
        if map_name in ['c2', 'c3', 'c4']:
            splits = ['train', 'test']
        else:
            splits = ['all']
    else:
        raise ValueError

    seq_map = []
    data_root = download.get_dataset_directory(root)
    seq_path = os.path.join(
        data_root, 'motchallenge-devkit/motchallenge/seqmaps')
    for sp in splits:
        seqmap_path = os.path.join(
            seq_path, '{0}-{1}.txt'.format(map_name, sp))
        with open(seqmap_path, 'r') as f:
            seq_m = f.read().split('\n')
        seq_map.extend(seq_m[1:-1])
    return seq_map


def get_mot(year, split):
    if year not in urls:
        raise ValueError

    data_root = download.get_dataset_directory(root)
    base_path = os.path.join(data_root, '2DMOT{}'.format(year))
    anno_path = os.path.join(base_path, 'annotations')
    anno_txt_path = os.path.join(anno_path, '{}.txt'.format(split))

    # if os.path.exists(anno_txt_path):
    #     return base_path
    #
    # download_file_path = utils.cached_download(urls[year])
    # ext = os.path.splitext(urls[year])[1]
    # utils.extractall(download_file_path, data_root, ext)

    download_devfile_path = utils.cached_download(dev_urls)
    dev_ext = os.path.splitext(dev_urls)[1]
    utils.extractall(download_devfile_path, data_root, dev_ext)

    if not os.path.exists(anno_path):
        os.mkdir(anno_path)

    if split == 'train':
        split_dirs = ['train']
    elif split == 'val':
        split_dirs = ['test']
    elif split == 'trainval':
        split_dirs = ['train', 'test']
    else:
        raise ValueError

    data_ids = []
    for split_d in split_dirs:
        seq_dirs = sorted(os.listdir(os.path.join(base_path, split_d)))
        for seq_d in seq_dirs:
            img_dir = os.path.join(base_path, split_d, seq_d, 'img1')
            img_names = sorted(os.listdir(img_dir))
            for img_name in img_names:
                data_id = '{0}_{1}_{2}'.format(
                    split_d, seq_d, img_name.split('.')[0])
                data_ids.append(data_id)

    with open(anno_txt_path, 'w') as anno_f:
        anno_f.write('\n'.join(data_ids))

    return base_path

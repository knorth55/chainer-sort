chainer-sort - SORT
===================
![Build Status](https://travis-ci.org/knorth55/chainer-sort.svg?branch=master)

This is [Simple, Online, Realtime Tracking of Multiple Objects](https://arxiv.org/abs/1602.00763) implementation for chainer and chainercv.

This repository provides MOT dataset, SORT tracker class and SORT examples with FasterRCNN and SSD.

[\[arXiv\]](https://arxiv.org/abs/1602.00763), [\[Original repo\]](https://github.com/abewley/sort)

<img src="./static/sort_faster_rcnn_example.gif" width="50%">

Notification
------------

- This repository is the implementation of [SORT](https://arxiv.org/abs/1602.00763), not [DeepSORT](https://arxiv.org/abs/1703.070402)
- SORT is based on Kalman filter and Hangarian algorithm and does not use deep learning techniques.
- In this repo, we use deep learning techniques (FasterRCNN and SSD) for object detection part.

Requirement
-----------

- [Chainer](https://github.com/chainer/chainer)
- [ChainerCV](https://github.com/chainer/chainercv)
- [FilterPy](https://github.com/rlabbe/filterpy)

Installation
------------

We recommend to use [Anacoda](https://anaconda.org/).

```bash
# Requirement installation
conda create -n chainer-sort python=2.7
source activate chainer-sort

git clone https://github.com/knorth55/chainer-sort
cd chainer-sort/
pip install -e .
```

Demo
----

```bash
cd examples/mot/
python demo.py --display
```

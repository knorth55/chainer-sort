import pkg_resources

from chainer_sort import datasets  # NOQA
from chainer_sort import models  # NOQA
from chainer_sort import trackers  # NOQA
from chainer_sort import utils  # NOQA
from chainer_sort import visualizations  # NOQA


__version__ = pkg_resources.get_distribution('chainer_sort').version

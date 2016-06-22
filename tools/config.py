import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.TRAIN = edict()


__C.TEST = edict()


__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..'))
__C.SAMPLEPATH = osp.join(__C.ROOT_DIR, 'data', 'vggface', 'vgg_list.txt')
__C.POSITIVE_NUM = 5
__C.IMAGEPATH = osp.join(__C.ROOT_DIR, 'data', 'vggface', 'dlib-affine-sz:224')
__C.MEANFILE = osp.join(__C.ROOT_DIR, 'data', 'vggface', 'vgg_mean.npy')

__C.GPU_ID = 0
__C.RNG_SEED = 3

__C.personPerBatch = 12
__C.imagesPerPerson = 10
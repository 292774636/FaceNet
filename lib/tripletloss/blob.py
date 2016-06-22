# --------------------------------------------------------
# TRIPLET LOSS
# Copyright (c) 2015 Pinguo Tech.
# Written by David Lu
# --------------------------------------------------------

"""Blob helper functions."""

import numpy as np
import cv2
from config import cfg

def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob

def prep_im_for_blob(im):
    mean_array = np.load(cfg.MEANFILE)
    target_size = 224
    im = cv2.resize(im, (224,224),
                    interpolation=cv2.INTER_LINEAR)
    im = im.astype(np.float32, copy=False)

    im[:, :, 0] = im[:, :, 0] - mean_array[0, :, :]
    im[:, :, 1] = im[:, :, 1] - mean_array[1, :, :]
    im[:, :, 2] = im[:, :, 2] - mean_array[2, :, :]

    im = im / 255.0
    #print im
    return im

# --------------------------------------------------------
# TRIPLET LOSS
# Copyright (c) 2015 Pinguo Tech.
# Written by David Lu
# --------------------------------------------------------

"""The data layer used during training a VGG_FACE network by triplet loss.
"""


import caffe
import numpy as np
from numpy import *
import yaml
from sklearn import preprocessing
from caffe._caffe import RawBlobVec



class Norm2Layer(caffe.Layer):
    """norm2 layer used for L2 normalization."""

    def setup(self, bottom, top):
        """Setup the TripletDataLayer."""
        #assert bottom[0].num % 3 == 0
        pass
        

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        #N = bottom[0].data.shape[0]
        #print bottom[0].shape
        top[0].reshape(*(bottom[0].shape))
        M = bottom[0].data.shape[1]
        self.b = np.sum((bottom[0].data[...]**2), axis=1) 
        self.b = np.tile(self.b, (M, 1))
        self.b = self.b.T
        self.c = self.b**0.5
        #print self.b.shape
        top[0].data[...] = bottom[0].data[...]/(self.c)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        
        a = self.c - ((bottom[0].data[...]**2)/self.c)
        diff = a / self.b
        bottom[0].diff[...] = diff * top[0].diff[...]
        

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

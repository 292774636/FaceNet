# --------------------------------------------------------
# TRIPLET LOSS
# Copyright (c) 2015 Pinguo Tech.
# Written by David Lu
# --------------------------------------------------------

"""The data layer used during training to train the network.
   This is a example for online triplet selection
   Each minibatch contains a set of archor-positive pairs, random select negative exemplar
"""

import caffe
import numpy as np
from numpy import *
import yaml
from sampledata import sampledata
import random
import cv2
from blob import prep_im_for_blob, im_list_to_blob
from config import cfg
import os.path as osp

class DataLayer(caffe.Layer):
    """Sample data layer used for training."""    

    def GenTripletList(self, num_triplet):
        #N = label.shape[0]
        triplet_list = []
        while len(triplet_list) < num_triplet:
            a_ind = random.randint(0, 4)
            p_ind = random.randint(0, 4)
            n_ind = random.randint(5, self._batch_size-1)
            if a_ind == p_ind:
                p_ind = max(0, a_ind - 1)
                if p_ind == 0:
                    p_ind = a_ind + 1
                else:
                    p_ind = a_ind - 1
            triplet_list.append([a_ind, p_ind, n_ind])
        return triplet_list

    def _get_next_minibatch(self):
        #sample people
        #class_list_sample = self.data_container.class_list_sample
        classes = random.sample(range(len(self.data_container.class_list_sample)), cfg.personPerBatch)
        #print 'classes', classes

        nSamplePerClass = np.zeros((cfg.personPerBatch, ), dtype = np.int32)
        sample = []
        
        for i in xrange(cfg.personPerBatch):
            nSample = np.minimum(len(self.data_container.class_list_sample[classes[i]]), int(cfg.imagesPerPerson))
            nSamplePerClass[i] = nSample

        #print nSamplePerClass

        for i in xrange(cfg.personPerBatch):
            cls = classes[i]
            nSample = nSamplePerClass[i]
            shuffle = range(len(self.data_container.class_list_sample[classes[i]]))
            random.shuffle(shuffle)

            for j in xrange(nSample):
                imgName = self.data_container.class_list_sample[cls][shuffle[j]]
                sample.append(imgName)
        #print sample
        #######################################
        im_blob,labels_blob = self._get_image_blob(sample)
        #triplet_label = self.GenTripletList(self._triplet)

        blobs = {'data': im_blob,
		         #'labels': labels_blob,
                 #'triplet_label': triplet_label
                 'num_per_class': nSamplePerClass
                 }
        return blobs

    def _get_image_blob(self,sample):
        im_blob = []
        labels_blob = []
        for i in range(len(sample)):
            img_name = osp.join(cfg.IMAGEPATH, sample[i])
            #print img_name
            im = cv2.imread(img_name)
            personname = sample[i].split('/')[0]
            #print str(i)+':'+personname+','+str(len(sample))
            labels_blob.append(self.data_container._sample_label[personname])
            im = prep_im_for_blob(im)
            im_blob.append(im)

        # Create a blob to hold the input images
        blob = im_list_to_blob(im_blob)
        return blob,labels_blob

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)
        self._batch_size = layer_params['batch_size']
        self._name_to_top_map = {
            'data': 0,
            #'labels': 1,
            #'triplet_label': 1
            'num_per_class': 1
            }
        self._triplet = 30 
        self.data_container =  sampledata()


        self._index = 0
        #print self._index
        # data blob: holds a batch of N images, each with 3 channels
        # The height and width (100 x 100) are dummy values
        top[0].reshape(1, 3, 224, 224)

        #top[1].reshape(self._batch_size)

        #top[1].reshape(self._triplet, 3)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob

        #print top[1].data[...]

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

if __name__ == '__main__':
    pass

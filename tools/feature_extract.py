#!/usr/bin/env python
"""
extract enbedding of lfw dataset
"""
import _init_paths
import os.path as osp
import os
import sys
import caffe
import cv2
import numpy as np
import cPickle
from config import cfg


if __name__ == '__main__':
    lfw_img_dir = osp.join(cfg.ROOT_DIR, 'data', 'lfw', 'dlib-affine-sz:224')

    #caffemodel = os.path.join(cfg.ROOT_DIR, 'output', 'vgg', '44', 'facenet_140000_iter_120000.caffemodel')
    caffemodel = os.path.join(cfg.ROOT_DIR, 'output', 'vgg', 'triplet_v2', 'facenet_triplet_iter_70000.caffemodel')
    prototxt = os.path.join(cfg.ROOT_DIR, 'model', 'facenet_deploy.prototxt')
    #prototxt = os.path.join(cfg.ROOT_DIR, 'model', 'cls_deploy.prototxt')
    #init caffe
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print ('Loaded network {:s}'.format(caffemodel))

    mean_arr = np.load(osp.join(cfg.ROOT_DIR, 'data', 'vggface', 'vgg_mean.npy'))

    i = 0
    np.set_printoptions(threshold='nan')
    with open('data/lfw_list.txt', 'r') as f:
        num_img = len(f.readlines())
    print num_img
    imdb = np.ndarray((num_img, 128), dtype=np.float32)
    name_list = []

    with open('data/lfw_list.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            #print line
            id = int(line.split(' ')[1])
            #identity_name = line.split('/')[0]
            img_name = line.split(' ')[0]
            img_name_sub = img_name.split('/')[1]
            img_name_abs = osp.join(lfw_img_dir, img_name)
            print 'extracting...' + img_name_abs

            img = cv2.imread(img_name_abs)
            img_resize = cv2.resize(img, (224, 224))
            img_resize = img_resize.astype(np.float32, copy=False)

            #substruct mean
            img_resize[:, :, 0] = img_resize[:, :, 0] - mean_arr[0, :, :]
            img_resize[:, :, 1] = img_resize[:, :, 1] - mean_arr[1, :, :]
            img_resize[:, :, 2] = img_resize[:, :, 2] - mean_arr[2, :, :]

            img_resize = img_resize / 255.0

            blob = np.zeros((1, 224, 224, 3), dtype=np.float32)
            blob[0, :, :, :] = img_resize
            channel_swap = (0, 3, 1, 2)
            blob = blob.transpose(channel_swap)
            net.blobs['data'].reshape(*(blob.shape))
            net.blobs['data'].data[...] = blob.astype(np.float32, copy=False)
            #print net.blobs['data'].data[0, 0,0,0]
            #forward_kwargs = {'data': blob}

            res = net.forward()
            ft = net.blobs['norm2'].data[...]
            #print ft.shape
            #print ft[0, 4089]


            #tmp['feature'] = ft[0, :]

            imdb[i, :] = ft[0, :]
            name_list.append(img_name_sub)
            i += 1


    with open('evaluation/vgg/triplet_v2/iter_70000/lfw_embedding.pkl', 'wb') as fid:
        cPickle.dump(imdb, fid, cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(name_list, fid, cPickle.HIGHEST_PROTOCOL)
    print 'wrote feature to imdb'

    #np.set_printoptions(threshold='nan')
    #print(feature_db)

import _init_paths
from config import cfg
import numpy as np
import caffe
import os

if __name__ == '__main__':
    #caffemodel = os.path.join(cfg.ROOT_DIR, 'output', 'vgg', '44', 'facenet_340000_iter_320000.caffemodel')
    #prototxt = os.path.join(cfg.ROOT_DIR, 'model', 'facenet_train_val.prototxt')

    caffemodel = os.path.join(cfg.ROOT_DIR, 'output', 'vgg', 'Inception', 'wdecay_iter_320000.caffemodel')
    prototxt = os.path.join(cfg.ROOT_DIR, 'model', 'Inception-Resnet2.prototxt')

    #init caffe
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print ('Loaded network {:s}'.format(caffemodel))

    test_iter = 100
    accuracy = 0
    _accuracy = 0


    for i in xrange(test_iter):
        out = net.forward()
        
        _accuracy = net.blobs['loss3/top-5'].data[...]
        print _accuracy
        accuracy += _accuracy

    accuracy = accuracy / test_iter
    print accuracy

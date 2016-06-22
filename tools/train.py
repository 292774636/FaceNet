import _init_paths
from config import cfg
import numpy as np
import caffe
from utils.timer import Timer
import os.path as osp
import cPickle
from pylab import *
import matplotlib.pyplot as plt

from caffe.proto import caffe_pb2
import google.protobuf as pb2

class SolverWrapper(object):
    def __init__(self, solver_prototxt,
                 pretrain_model=None):

        self.solver = caffe.SGDSolver(solver_prototxt)

        if pretrain_model is not None:
            print ('loading pretrained model from {:s}').format(pretrain_model)
            self.solver.net.copy_from(pretrain_model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)


    def train_model(self, max_iters):
        #display = self.solver_param.display #40
        #test_iter = 1
        #test_interval = 1
        #_accuracy = 0
        #accuracy = 0

        timer = Timer()
        while self.solver.iter < max_iters:
            #print self.solver.iter
            #make one SGD update
            timer.tic()
            self.solver.step(1)
            timer.toc()
            """
            _train_loss += self.solver.net.blobs['euclidean_loss'].data
            if (self.solver.iter-1) % display == 0:
                train_loss[(self.solver.iter-1) // display] = _train_loss / display
                _train_loss = 0
            """
            if self.solver.iter % (self.solver_param.display) == 0:
                print ('speed {:.3f}s / iter').format(timer.average_time)
            """
            if self.solver.iter % test_interval == 0:
                for test_it in range(test_iter):
                    self.solver.test_nets[0].forward()
                    _accuracy += self.solver.test_nets[0].blobs['loss3/top-5'].data
                accuracy = _accuracy / test_iter
                f.write(str(self.solver.iter) + ' ' + str(accuracy) + '\n')
                _accuracy = 0
            """
                
            
        """
        _, ax1 = plt.subplots()
        ax1.plot(display * arange(len(train_loss)), train_loss, 'g')
        ax1.set_xlabel('iteration')
        ax1.set_ylabel('loss')
        plt.show()
        """

    def snapshot(self):

        net = self.solver.net

        filename = (self.solver_param.snapshot_prefix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = osp.join(cfg.OUTPUT_DIR, filename)

        net.save(str(filename))
        print ('wrote snapshot to: {:s}').format(filename)



def _init_caffe(cfg):

    np.random.seed(cfg.RNG_SEED)
    caffe.set_random_seed(cfg.RNG_SEED)

    caffe.set_mode_gpu()
    caffe.set_device(cfg.GPU_ID)

if __name__ == '__main__':
    solver_prototxt = osp.join(cfg.ROOT_DIR, 'model', 'inception_resnet_v2_solver.prototxt')
    pretrain_model = osp.join(cfg.ROOT_DIR, 'output', 'vgg', 'facenet_iter_200000.caffemodel')
    max_iters = 500000
    #max_iters = 5


    _init_caffe(cfg)

    sw = SolverWrapper(solver_prototxt,
                       pretrain_model=None)


    #print ('Solving...')
    #sw.train_model(max_iters)
    #print ('done solving')





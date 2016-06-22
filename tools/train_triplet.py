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

        display = self.solver_param.display
        #train_loss = zeros(ceil(max_iters * 1.0 / display))
        _train_loss = 0
        timer = Timer()
        while self.solver.iter < max_iters:
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
            if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = self.solver.iter
                self.snapshot()

        if last_snapshot_iter != self.solver.iter:
            self.snapshot()
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
    solver_prototxt = osp.join(cfg.ROOT_DIR, 'model', 'facenet_solver_triplet.prototxt')
    pretrain_model = osp.join(cfg.ROOT_DIR, 'output', 'vgg', '44', 'facenet_140000_iter_200000.caffemodel')
    max_iters = 500000


    _init_caffe(cfg)

    sw = SolverWrapper(solver_prototxt,
                       pretrain_model=pretrain_model)

    #net = sw.solver.net
    #out = net.forward()
    #print net.blobs['loss'].data[...]


    print ('Solving...')
    sw.train_model(max_iters)
    print ('done solving')


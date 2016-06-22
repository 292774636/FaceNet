import caffe
import numpy as np
from caffe._caffe import RawBlobVec

class MulConstant(caffe.Layer):

	def setup(self, bottom, top):
		self.alpha = 0.1

	def forward(self, bottom, top):
		top[0].reshape(*(bottom[0].shape))
		top[0].data[...] = bottom[0].data[...] * self.alpha

	def backward(self, top, propagate_down, bottom):
		bottom[0].diff[...] = self.alpha * top[0].diff[...]

	def reshape(self, bottom, top):
		pass

import caffe
import numpy as np
import random
from config import cfg
from scipy.spatial.distance import cdist
import itertools

class TripletLoss:

	def __init__(self):
		self.margin = 0.2

	def _forward(self, x_a, x_p, x_n):
		a_p = x_a - x_p
		a_n = x_a - x_n
		ap = np.sum(a_p**2, axis = 1)
		an = np.sum(a_n**2, axis = 1)

		self.dist = ap - an + self.margin
		b = np.maximum(0, self.dist)
		loss = np.sum(b, axis = 0)
		return loss

	def _gradient(self, x_a, x_p, x_n):
		#n = x_a.shape[0]
		Fa = (2*(x_n - x_p))
		Fp = (2*(x_p - x_a))
		Fn = (2*(x_a - x_n))
		Fa[self.dist<0,:] = 0
		Fp[self.dist<0,:] = 0
		Fn[self.dist<0,:] = 0
		#Fa[self.dist>0.8,:]= 0
		#Fp[self.dist>0.8,:]= 0
		#Fn[self.dist>0.8,:]= 0
		return Fa, Fp, Fn

	def _numerical_gradient(self, x_a,x_p,x_n):
	    (m,n) = x_a.shape
	    eps = 1e-4
	    gq = np.zeros((m,n))
	    gi = np.zeros((m,n))
	    gj = np.zeros((m,n))
	    for i in range(m):
	        for j in range(n):
	            q1 = x_a.copy()
	            q2 = x_a.copy()
	            q1[i,j] +=eps
	            q2[i,j] -=eps
	            l1 = self._forward(q1, x_p, x_n)
	            l2 = self._forward(q2, x_p, x_n)
	            gq[i,j] = (l1-l2)/(2*eps)
	    for i in range(m):
	        for j in range(n):
	            i1 = x_p.copy()
	            i2 = x_p.copy()
	            i1[i,j] +=eps
	            i2[i,j] -=eps
	            l1 = self._forward(x_a, i1, x_n)
	            l2 = self._forward(x_a, i2, x_n)
	            gi[i,j] = (l1-l2)/(2*eps)
	    for i in range(m):
	        for j in range(n):
	            j1 = x_n.copy()
	            j2 = x_n.copy()
	            j1[i,j] +=eps
	            j2[i,j] -=eps
	            l1 = self._forward(x_a, x_p, j1)
	            l2 = self._forward(x_a, x_p, j2)
	            gj[i,j] = (l1-l2)/(2*eps)
	    return (gq, gi, gj)

class AdaptorLayer:

	def __init__(self):
	    self._triple = TripletLoss()

	def _forward(self, feature, ranklist):
	    self.ranklist = ranklist.astype(np.int32, copy=True)
	    self.hq = feature[self.ranklist[:,0],:]
	    self.hi = feature[self.ranklist[:,1],:]
	    self.hj = feature[self.ranklist[:,2],:]
	    #print('ranklist',self.ranklist,self.hq.shape, self.hi, self.hj)
	    L = self._triple._forward(self.hq, self.hi, self.hj)
	    #print ('l',L)
	    return (L)  

	def _gradient(self, M):
	    (Fq,Fi,Fj) = self._triple._gradient(self.hq, self.hi, self.hj)
	    #M = self.ranklist.max() + 1
	    #print 'M', M
	    N = Fq.shape[1]
	    dx = np.zeros((M,N))
	    for i in range (M):
	        dx[i,:] = np.sum(Fq[self.ranklist[:,0]==i],axis = 0)+np.sum(Fi[self.ranklist[:,1]==i],axis = 0)+np.sum(Fj[self.ranklist[:,2]==i],axis = 0)
	    return dx

	def _numerical_gradient(self, feature, ranklist):
	    S = AdaptorLayer()
	    (m,n) = feature.shape
	    eps = 1e-4
	    grad = np.zeros((m,n))
	    for i in range(m):
	        for j in range (n):
	            g1 = feature.copy()
	            g2 = feature.copy()
	            g1[i,j] +=eps
	            g2[i,j] -=eps
	            l1 = S._forward(g1, ranklist)
	            l2 = S._forward(g2, ranklist)
	            grad[i,j] = (l1-l2)/(2*eps)
	    print ('check',grad)
	    return grad

def GenTripletList(feature, y):
	"""
		feature -- embeddings
		y -- labels
	"""
	margin = 0.2
	(M, N) = feature.shape
	#print 'feature', feature
	#print (M, N)
	#print 'y', y
	pairwise_distances = cdist(feature, feature, 'sqeuclidean')
	#print 'pairwise_distances', pairwise_distances

	a_p = []
	triplets = []

	for label in np.unique(y):
		a_p.extend(list(itertools.combinations(np.where(y==label)[0], 2)))
	#print 'a_p', a_p

	for a_i, p_i in a_p:
		#print (a_i, p_i)
		a_p_dist = pairwise_distances[a_i, p_i]

		negatives_idx = np.where(y!=y[a_i])[0]
		#print negatives_idx

		diff = pairwise_distances[a_i, negatives_idx] - a_p_dist
		#print 'a_p_dist', a_p_dist
		#print 'diff', diff

		soft_negatives = np.where((diff>0.) & (diff<margin))[0]
		#print soft_negatives

		if len(soft_negatives) > 0:
			n_i = negatives_idx[soft_negatives[diff[soft_negatives].argmin()]]
			triplets.append(np.array((a_i, p_i, n_i)))

	return np.array(triplets, dtype=np.int32)

class TripletLayer(caffe.Layer):

	def setup(self, bottom, top):
	    top[0].reshape(1)

	def reshape(self, bottom, top):
	    top[0].reshape(1)

	def forward(self, bottom, top):
	    self._adaptor = AdaptorLayer()
	    #(N, M) = bottom[1].shape
	    top[0].reshape(1)
	    self.triplet_list = GenTripletList(bottom[0].data, bottom[1].data)
	    N = len(self.triplet_list)
	    top[0].data[0] = self._adaptor._forward(bottom[0].data, self.triplet_list)/N

	def backward(self, top, propagate_down, bottom):
	    #print 'fn', bottom[0].shape[0]
	    M = bottom[0].shape[0]
	    N = len(self.triplet_list)
	    loss = top[0].data[0]
	    bottom[0].diff[...] = self._adaptor._gradient(M)*loss/N


if __name__ == '__main__':
	"""
	M = 3
	N = 4
	hq = np.random.rand(M,N)*2-1
	hi = np.random.rand(M,N)*2-1
	hj = np.random.rand(M,N)*2-1
	print('hq', hq)
	print('hi', hi)
	print('hj', hj)
	L = TripletLoss()
	l = L._forward(hq, hi, hj)
	print ('l',l)
	(gradq,gradi,gradj) = L._gradient(hq, hi, hj)
	(ngradq, ngradi, ngradj) = L._numerical_gradient(hq,hi, hj)
	print(gradq, ngradq)
	print(gradi, ngradi)
	print(gradj, ngradj)

    """
	M = 6
	N = 4
	feature = np.random.rand(M,N)*2-1
	ranklist = np.random.randint(0, M, (2*M, 3))

	print ('ranklist', ranklist)
	print ('feature',feature)
	F = AdaptorLayer()
	L = F._forward(feature, ranklist)

	G = F._gradient()
	print G
	check = F._numerical_gradient(feature, ranklist)

	print check
    

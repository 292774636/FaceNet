#import caffe
import numpy as np
import random


def GenTripletList(num_triplet):
	#N = label.shape[0]
	triplet_list = []
	while len(triplet_list) < num_triplet:
		a_ind = random.randint(0, 4)
		p_ind = random.randint(0, 4)
		n_ind = random.randint(5, 30-1)
		if a_ind == p_ind:
			p_ind = max(0, a_ind - 1)
			if p_ind == 0:
				p_ind = a_ind + 1
			else:
				p_ind = a_ind - 1
		triplet_list.append([a_ind, p_ind, n_ind])
	return triplet_list




if __name__ == '__main__':
	print(GenTripletList(30))


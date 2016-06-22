import _init_paths
from config import cfg
import numpy as np
import cPickle
from sklearn import preprocessing

def L2_Distance(a, b):

    #c = a - b
    #return np.dot(c.T, c)
    return np.linalg.norm(a - b)

def eval_accuracy(dis_p, dis_n, thres):
    accuracy = 0.0
    count = 0

    for i in xrange(len(dis_p)):
        if dis_p[i] <= thres:
            count += 1

    for i in xrange(len(dis_n)):
        if dis_n[i] >= thres:
            count +=1

    totol = len(dis_p) + len(dis_n)
    accuracy = float(count) / totol
    return accuracy

if __name__ == '__main__':

    """
    pairs_filename = '../data/pairs.txt'

    with open(pairs_filename, 'r') as fid:
        pair_list = fid.readlines()

    pair_list.pop(0)
    pos_pair = []
    neg_pair = []
    tmp = {}

    for fold in xrange(10):
        for p in xrange(300):
            i = 600 * fold + p
            split = pair_list[i].strip('\n').split('\t')
            tmp['img1'] = split[0] + '_' + split[1].zfill(4) + '.jpg'
            tmp['img2'] = split[0] + '_' + split[2].zfill(4) + '.jpg'
            tmp['label'] = 1    #same person
            pos_pair.append(tmp)
            tmp = {}
            #print tmp['img2']

        for n in xrange(300):
            i = 600 * fold + 300 + n
            split = pair_list[i].strip('\n').split('\t')
            tmp['img1'] = split[0] + '_' + split[1].zfill(4) + '.jpg'
            tmp['img2'] = split[2] + '_' + split[3].zfill(4) + '.jpg'
            tmp['label'] = 0
            neg_pair.append(tmp)
            tmp = {}
            #print split


    filename_pos = '../data/pos_pairs.pkl'
    filename_neg = '../data/neg_pairs.pkl'

    with open(filename_pos, 'wb') as fid:
        cPickle.dump(pos_pair, fid, cPickle.HIGHEST_PROTOCOL)
    print 'wrote pairs to {:s}'.format(filename_pos)

    with open(filename_neg, 'wb') as fid:
        cPickle.dump(neg_pair, fid, cPickle.HIGHEST_PROTOCOL)
    print 'wrote pairs to {:s}'.format(filename_neg)

    """
    filename_pos = 'data/pos_pairs.pkl'
    filename_neg = 'data/neg_pairs.pkl'
    filename_imdb = 'data/lfw_embedding.pkl'

    with open(filename_pos, 'rb') as fid:
        pos_pair = cPickle.load(fid)

    with open(filename_neg, 'rb') as fid:
        neg_pair = cPickle.load(fid)


    with open(filename_imdb, 'rb') as fid:
        imdb = cPickle.load(fid)
        name_list = cPickle.load(fid)

    """
    min_max_scaler = preprocessing.MinMaxScaler()
    imdb_norm = min_max_scaler.fit_transform(imdb)
    with open('../data/imdb_norm.pkl', 'wb') as fid:
        cPickle.dump(imdb_norm, fid, cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(name_list, fid, cPickle.HIGHEST_PROTOCOL)
    #print neg_pair
"""

    dis = np.ndarray((2, len(pos_pair)), dtype=np.float32)

    np.set_printoptions(threshold='nan')
    for i in xrange(len(pos_pair)):
    #for i in xrange(2):
        img1 = pos_pair[i]['img1']
        img2 = pos_pair[i]['img2']
        #print img1
        #print img2
        ind1 = name_list.index(img1)
        ind2 = name_list.index(img2)
        dis[0, i] = L2_Distance(imdb[ind1], imdb[ind2])


    for i in xrange(len(neg_pair)):
        img1 = neg_pair[i]['img1']
        img2 = neg_pair[i]['img2']
        #print img1
        #print img2
        ind1 = name_list.index(img1)
        ind2 = name_list.index(img2)
        dis[1, i] = L2_Distance(imdb[ind1], imdb[ind2])

    print np.mean(dis, axis=1)
    #print len(imdb.max(axis=1))
    accuracy = eval_accuracy(dis[0, :], dis[1, :], 0.19)

    print accuracy





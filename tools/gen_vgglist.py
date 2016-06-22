import _init_paths
import os.path as osp
import os
import sys
import caffe
import cv2
import numpy as np
import cPickle

if __name__ == '__main__':
    VGG_FACE_PATH = '/home/huangrui/Desktop/face/FaceNet/data/vggface/dlib-affine-sz_299'
    name_list = os.listdir(VGG_FACE_PATH)
    i = 0
    n = 0
    #print name_list

    with open('vallist_299.txt', 'w') as f:
        for person_name in name_list:
            print person_name
            img_list = os.listdir(osp.join(VGG_FACE_PATH, person_name))
            n = 0
            for img_name in img_list:
                if n % 100 == 0:
                    line = person_name + '/' + img_name + ' ' + str(i) + '\n'
                    f.write(line)
                n += 1
            i += 1

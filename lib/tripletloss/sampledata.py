# --------------------------------------------------------
# TRIPLET LOSS
#
#
# --------------------------------------------------------

import os
import codecs
from config import cfg

class sampledata():

    global _sample_person
    global _sample_negative
    global _sample
    global _sample_label

    def __init__(self):
        self._sample_person = {}
        self._sample_negative = {}
        self._sample = []
        self._sample_label = {}
        self.class_list_sample = []
        lines = open(cfg.SAMPLEPATH,'r')
        #lines = open('data/val.txt','r')
        i = -1
        for line in lines:
            line = line.strip('\n')
            personname = line.split('/')[0]
            picname = line.split(' ')[0]
            self._sample.append(picname)
            if personname in self._sample_person.keys():
                self._sample_person[personname].append(picname)
                self.class_list_sample[i].append(picname)
            else:
                i += 1
                self._sample_person[personname] = []
                self._sample_person[personname].append(picname)
                self._sample_label[personname] = len(self._sample_person) - 1
                self.class_list_sample.append([])
                self.class_list_sample[i].append(picname)
                #print self.class_list_sample[i]
                
        #print self.class_list_sample
        

if __name__ == '__main__':

    sample = sampledata()
    print sample._sample_label['0000099']

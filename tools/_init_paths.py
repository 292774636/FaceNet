#!/usr/bin/env python

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

caffe_path = osp.join(this_dir, '..', '..','..', 'caffe', 'python')
lib_path = osp.join(this_dir, '..', 'lib')
tool_path = osp.join(this_dir, '..', 'tools')

add_path(caffe_path)
add_path(lib_path)
add_path(tool_path)
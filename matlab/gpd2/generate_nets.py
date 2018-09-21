# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 20:25:43 2015

@author: rplatt
"""

import numpy as np
import matplotlib.pyplot as plt

# Make sure that caffe is on the python path:
caffe_root = '/home/rplatt/projects/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

from caffe import layers as L
from caffe import params as P

import sys

foldername = sys.argv[1]

# GLOG_logtostderr=1 ./build/tools/convert_imageset ~/data/grasp_images/ ~/data/grasp_images/train_val.txt ~/data/grasp_images/val_lmdb --gray --shuffle

def lenet(lmdb, batch_size):
  # our version of LeNet: a series of linear and simple nonlinear transformations
  n = caffe.NetSpec()
  n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb, transform_param=dict(scale=1./255), ntop=2)
  n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
  #~ n.conv1 = L.Convolution(n.data, kernel_size=10, num_output=40, weight_filler=dict(type='xavier')) # double # neurons
  n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
  #~ n.conv2 = L.Convolution(n.pool1, kernel_size=10, num_output=100, weight_filler=dict(type='xavier')) # double # neurons
  n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
  n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)

  #~ n.conv3 = L.Convolution(n.pool2, kernel_size=5, num_output=50, weight_filler=dict(type='xavier')) # double # neurons
  #~ n.pool3 = L.Pooling(n.conv3, kernel_size=2, stride=2, pool=P.Pooling.MAX)

  n.ip1 = L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
  #~ n.ip1 = L.InnerProduct(n.pool3, num_output=500, weight_filler=dict(type='xavier'))
  n.relu1 = L.ReLU(n.ip1, in_place=True)
  n.ip2 = L.InnerProduct(n.relu1, num_output=2, weight_filler=dict(type='xavier'))
  n.accuracy = L.Accuracy(n.ip2, n.label)
  
  n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
  #~ n.ip2softmax = L.Softmax(n.ip2)
  #~ n.loss = L.MultinomialLogisticLoss(n.ip2, n.label)
  
  return n.to_proto()


print "---- Generating train.prototxt, test.prototxt, and lenet_solver.prototxt ----"
foldpath = './data/' + foldername + '/'

with open('./data/' + foldername + '/train.prototxt','w') as f:
  f.write(str(lenet('./data/' + foldername + '/train_lmdb', 64)))
  
with open('./data/' + foldername + '/test.prototxt','w') as f:
  f.write(str(lenet('./data/' + foldername + '/test_lmdb', 100)))

solver_in = open('./lenet_auto_solver.prototxt', 'r')
solver_out = open('./data/' + foldername + '/solver.prototxt', 'w+')
solver_out.write('# The train/test net protocol buffer definition\n')
solver_out.write('train_net: \"./data/' + foldername + '/train.prototxt\"\n')
solver_out.write('test_net: \"./data/' + foldername + '/test.prototxt\"\n')
lines_solver = solver_in.readlines()
lines_solver = lines_solver[3 : len(lines_solver)]
for ll in lines_solver:
  if ll.find('snapshot_prefix') >= 0:
    solver_out.write('snapshot_prefix: \"./data/' + foldername + '/lenet\"\n')
  else:
    solver_out.write(ll)
solver_out.close()
solver_in.close()

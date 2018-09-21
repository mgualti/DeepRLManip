# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 20:41:09 2015

@author: rplatt
"""

# python train_lenet.py bb_singleobj_bottles_orthnorm/softsoap_gold_clsLearning ./data/bb_category_bottles_orthnorm/lenet_iter_25000_015.caffemodel

import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import sys

# Make sure that caffe is on the python path:
import sys
import caffe

foldername = sys.argv[1]

print foldername

caffe.set_device(0)
caffe.set_mode_gpu()

# examples/mnist/lenet_black_and_decker_lithium_drill_driver_unboxed_solver.prototxt

acc = []
loss = []
objects = []

print "--------------------"
solver = caffe.SGDSolver('./data/' + foldername + '/solver.prototxt')

if size(sys.argv) > 2:
    caffemodelFile = sys.argv[2]
    solver.net.copy_from(caffemodelFile)

niter = 300
test_interval = 1
# losses will also be stored in the log
train_loss = zeros(niter)
test_acc = []
output = zeros((niter, 8, 10))

print 'Testing initial network ...'
correct = 0
solver.step(1)  # SGD by Caffe
for test_it in range(100):
	solver.test_nets[0].forward()
	correct += sum(solver.test_nets[0].blobs['ip2'].data.argmax(1)
				   == solver.test_nets[0].blobs['label'].data)
# test_acc[it // test_interval] = correct / 1e4
test_acc.append(correct / 1e4)

# the main solver loop
for it in range(niter):
  solver.step(100)  # SGD by Caffe
  
  # store the train loss
  train_loss[it] = solver.net.blobs['loss'].data
  
  # store the output on the first test batch
  # (start the forward pass at conv1 to avoid loading new data)
  solver.test_nets[0].forward(start='conv1')
  # output[it] = solver.test_nets[0].blobs['ip2'].data[:8]
  
  # run a full test every so often
  # (Caffe can also do this for us and write to a log, but we show here
  #  how to do it directly in Python, where more complicated things are easier.)
  if it % test_interval == 0:
    print 'Iteration', it, 'testing...'
    correct = 0
    for test_it in range(100):
        solver.test_nets[0].forward()
        correct += sum(solver.test_nets[0].blobs['ip2'].data.argmax(1)
                       == solver.test_nets[0].blobs['label'].data)
    # test_acc[it // test_interval] = correct / 1e4
    test_acc.append(correct / 1e4)

# plot train loss and test accuracy
# plt.figure()
# _, ax1 = subplots()
# ax2 = ax1.twinx()
# ln1 = ax1.plot(arange(niter), train_loss, 'b', label = 'train loss')
# ln2 = ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r', label = 'test accuracy')
# ax1.set_xlabel('iteration')
# ax1.set_ylabel('train loss')
# ax2.set_ylabel('test accuracy')
# ln = ln1 + ln2
# labels = [l.get_label() for l in ln]
# plt.legend(ln, labels, loc = 0)
# plt.title('')
# plt.show(block=False)

with open('./data/' + foldername + '/solver_results.txt', "w") as file_results:
  file_results.write('accuracy:\n')
  for i in range(0, len(test_acc)):
    file_results.write(str(test_acc[i]) + '\n')
#  file_results.write('\n')

with open('./data/' + foldername + '/solver_activations.txt', "w") as file_activations:
    
  for thres_iter in range(0,51):
    #~ thres = thres_iter * 0.2
    thres = thres_iter * 0.3
    
    num_tp = 0
    num_tn = 0
    num_fp = 0
    num_fn = 0
    for test_it in range(100):
      solver.test_nets[0].forward()
    
#      estimates = solver.test_nets[0].blobs['ip2'].data.argmax(1)
      estimates = solver.test_nets[0].blobs['ip2'].data[:,1] > thres + solver.test_nets[0].blobs['ip2'].data[:,0]
      labels = solver.test_nets[0].blobs['label'].data
    
      num_tp = num_tp + sum((estimates == 1) & (labels == 1))
      num_tn = num_tn + sum((estimates == 0) & (labels == 0))
      num_fp = num_fp + sum((estimates == 1) & (labels == 0))
      num_fn = num_fn + sum((estimates == 0) & (labels == 1))
    
    precision = float(num_tp) / (float(num_tp) + float(num_fp))
    recall = float(num_tp) / (float(num_tp) + float(num_fn))
    tp_rate = float(num_tp) / (float(num_tp) + float(num_fn))
    fp_rate = float(num_fp) / (float(num_tn) + float(num_fp))
	  
	  
    #~ file_activations.write(str([thres, precision, recall]) + '\n')
    file_activations.write(str(thres) + ', ' + str(precision) + ', ' + str(recall) + '\n')
    #~ file_activations.write(str([tp_rate, fp_rate]) + '\n')
  
  file_activations.close()    

#acc.append(test_acc[len(test_acc) - 1])
# store results
  
  #~ # store results
  #~ with open('./data/' + foldername + '/results_cnn_grasp_images.txt', "w+") as file_results:
#~ for i in range(0, len(acc)):
  #~ file_results.write(str(objects[i]) + ' ' + str(acc[i]) + '\n')

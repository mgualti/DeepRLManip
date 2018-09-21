import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import sys
import matplotlib.pyplot as plt
import scipy.io as sio

# Make sure that caffe is on the python path:
#caffe_root = '/home/rplatt/projects/caffe/'
caffe_root = '/home/baxter/software/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_device(0)
caffe.set_mode_gpu()

net = caffe.Net('./data/CAFFEfiles/test.prototxt', './data/CAFFEfiles/lenet_iter_1000.caffemodel', caffe.TEST)
net.forward()

predictionList = []
predictionProbsList = []
with open('./data/temp/test.txt', "r") as fh:
	lines = fh.readlines()
	for line in lines:
     
     
           #~ filename = line[0:-8] + '.mat'
           filename = line[0:-3]
           print 'Loading ' + filename
           mat = sio.loadmat('./data/temp/jpgs/' + filename)  
           im = mat['im']
#            im = mpimg.imread(prefix + '/jpgs/' + filename)
           im = np.rollaxis(im,2) # make first dimension channel
     
#           filename = line[:-3]
#           img = plt.imread('./data/temp/jpgs/' + filename)
#           imgbgr = array([img.transpose(2,0,1)[2], img.transpose(2,0,1)[1], img.transpose(2,0,1)[0]])
#           imgbgr = array([im.transpose(2,0,1)[14], im.transpose(2,0,1)[13], im.transpose(2,0,1)[12], im.transpose(2,0,1)[11], im.transpose(2,0,1)[10], im.transpose(2,0,1)[9], im.transpose(2,0,1)[8], im.transpose(2,0,1)[7], im.transpose(2,0,1)[6]], im.transpose(2,0,1)[5], im.transpose(2,0,1)[4], im.transpose(2,0,1)[3], im.transpose(2,0,1)[2], im.transpose(2,0,1)[1], im.transpose(2,0,1)[0]])
#           imgbgr = array([im[14],im[13],im[12],im[11],im[10],im[9],im[8],im[7],im[6],im[5],im[4],im[3],im[2],im[1],im[0]])
           
           #~ imgbgr = im[0:3,...]
           imgbgr = im
           net.blobs['data'].data[0] = imgbgr
           net.forward(start='conv1')
		
           predictionProbs = net.blobs['ip2'].data[0].copy()
           prediction = net.blobs['ip2'].data[0].argmax(0)
		#~ print 'prediction: ' + str(prediction)
		#~ print 'predictionProbs: ' + str(predictionProbs)
		
           predictionList.append(prediction)
           predictionProbsList.append(predictionProbs)

#~ print predictionProbsList

#~ with open('./data/temp/predictionList.txt', "w") as fh:
	#~ for prediction in predictionList:
		#~ fh.write(str(prediction) + '\n')
with open('./data/temp/predictionList.txt', "w") as fh:
	for predictionProbs in predictionProbsList:
		#~ fh.write(str(predictionProbs) + '\n')
		fh.write(str(predictionProbs[0]) + ', ' + str(predictionProbs[1]) + '\n')



# Usage: python generate_lmdb_rob2.py bb_singleobj_test_orthnorm/band_aid_clear_strips_clsLearning
# Run from ./projects/gpd directory
# create train and test LMDBs given the jpgs directory and train.txt, test.txt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylab import *
import sys
import scipy.io as sio
import lmdb
import caffe


def loadData(fileName, imgChannels):
	
	# Calculate number of .MAT files to import
    with open(fileName, "r") as file_in:  
        numData = sum(1 for line in file_in)
        if imgChannels == range(8,10): # this is code for the kappler representation
            data = np.zeros((numData,size(imgChannels)+1,60,60), dtype=np.uint8)
        elif imgChannels == range(0,16): # this is code for eliminating occlusion data
            data = np.zeros((numData,12,60,60), dtype=np.uint8)
        else:
            data = np.zeros((numData,size(imgChannels),60,60), dtype=np.uint8)
#        data = np.zeros((numData,15,60,60), dtype=np.uint8)
#        data = np.zeros((numData,5,60,60), dtype=np.uint8)
#        data = np.zeros((numData,3,60,60), dtype=np.uint8)
        labels = np.zeros((numData))
		  
	# Import all the .MAT files into <imacc>
    with open(fileName, "r") as file_in:  
        ii = 0
        for line in file_in:
            label = line[-2:]
            #~ filename = line[0:-8] + '.mat'
            #~ filename = line[0:-8] + '.jpeg'
            filename = line[0:-3]
            print 'Loading ' + filename + ', ' + label
            mat = sio.loadmat(prefix + '/jpgs/' + filename)  
            im = mat['im']
#            im = mpimg.imread(prefix + '/jpgs/' + filename)
            im = np.rollaxis(im,2) # make first dimension channel
            
            # the following image is created to duplicate what was used in the Kappler papers
            imFree = 255*(1 - (np.maximum(im[8,...],im[9,...])>0))

            # if we are using the Kappler data, then add an extra dimension containing the free data            
            if imgChannels == range(8,10): # this is code for the kappler representation
                data[ii] = r_[im[imgChannels,...],imFree[np.newaxis,...]]
            elif imgChannels == range(0,16): # this is code for eliminating occlusion data
                data[ii] = r_[im[range(0,4),...],im[range(5,9),...],im[range(10,14),...]]
            else:
                data[ii] = im[imgChannels,...]
#            data[ii] = im[5:10,...]
#            data[ii] = im[0:5,...]
#            data[ii] = im[0:3,...] # test using only a single rgb image
            labels[ii] = label

            ii = ii + 1

    # randomly permute the data
#    randIdx = np.random.permutation(range(0,data.shape[0]))

#    return [data[randIdx],labels[randIdx]]
    return [data,labels]


def writeData2LMDB(data,labels,lmdbName):
    
    # We need to prepare the database for the size. We'll set it 10 times
    # greater than what we theoretically need. There is little drawback to
    # setting this too big. If you still run into problem after raising
    # this, you might want to try saving fewer entries in a single
    # transaction.
    map_size = data.nbytes * 10
    env = lmdb.open(lmdbName, map_size=map_size)
    
    with env.begin(write=True) as txn:
    # txn is a Transaction object
    
        randIdx = np.random.permutation(range(0,data.shape[0]))
#        for i in range(labels.shape[0]):
        j=1
        for i in randIdx:
            
            print 'Adding elt ' + str(i) + ' to the LMDB'
            
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = data.shape[1]
            datum.height = data.shape[2]
            datum.width = data.shape[3]
            datum.data = data[i].tobytes()  # or .tostring() if numpy < 1.9
            datum.label = int(labels[i])
            #~ str_id = '{:08}'.format(i)
            str_id = '{:08}'.format(j)
            j=j+1
            txn.put(str_id.encode('ascii'), datum.SerializeToString()) # The encode is only essential in Python 3
        
        
        
# Make sure that caffe is on the python path:
caffe_root = '/home/rplatt/projects/caffe/'
sys.path.insert(0, caffe_root + 'python')

foldername = sys.argv[1]
rangeStart = int(sys.argv[2])
rangeEnd = int(sys.argv[3])
#~ foldername = '3dnet_cans'
#~ rangeStart = 0
#~ rangeEnd = 3

imgChannels = range(rangeStart,rangeEnd)

prefix = './data/' + foldername

[data, labels] = loadData(prefix + '/train.txt',imgChannels)  
writeData2LMDB(data,labels,prefix + '/train_lmdb')

[data, labels] = loadData(prefix + '/test.txt',imgChannels)  
writeData2LMDB(data,labels,prefix + '/test_lmdb')

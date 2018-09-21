
import numpy as np
import sys
import os
import shutil

def trainThisObject(objPath,rangeStart,rangeEnd):

    cmd = 'rm -r ./data/' + objPath + '/train_lmdb'
    os.system(cmd)
    cmd = 'rm -r ./data/' + objPath + '/test_lmdb'
    os.system(cmd)
    
    cmdLmdb = 'python generate_lmdb_rob2.py ' + objPath + ' ' + str(rangeStart) + ' ' + str(rangeEnd)
    cmdNets = 'python generate_nets_train.py ' + objPath
    
    os.system(cmdLmdb)
    os.system(cmdNets)

    cmdTrain = 'python train_lenet.py ' + objPath
    #~ cmdTrain = 'python train_lenet.py ' + objPath + ' ./data/3dnet/lenet_iter_30000_015.caffemodel'
    #~ cmdTrain = 'python train_lenet.py ' + objPath + ' ./data/3dnet/lenet_iter_30000_03.caffemodel'
    #~ cmdTrain = 'python train_lenet.py ' + objPath + ' ./data/bb_boxes_cans_bottles_balanced/lenet_iter_20000_015.caffemodel'
    #~ cmdTrain = 'python train_lenet.py ' + objPath + ' ./data/3dnet_all/lenet_iter_30000_03.caffemodel'
    os.system(cmdTrain)
    #~ shutil.copyfile('./data/' + objPath + '/solver_results.txt', './data/' + objPath + '/solver_results_pretrainOld3DNET03.txt')
    #~ shutil.copyfile('./data/' + objPath + '/solver_activations.txt', './data/' + objPath + '/solver_activations_pretrainOld3DNET03.txt')
    
    #~ cmdTrain = 'python train_lenet.py ' + objPath
    #~ os.system(cmdTrain)
    #~ shutil.copyfile('./data/' + objPath + '/solver_results.txt', './data/' + objPath + '/solver_results_nopretrain.txt')
    #~ shutil.copyfile('./data/' + objPath + '/solver_activations.txt', './data/' + objPath + '/solver_activations_nopretrain.txt')

    #~ shutil.copyfile('./data/' + objPath + '/solver_results.txt', './data/' + objPath + '/solver_results' + str(rangeStart) + str(rangeEnd) + '.txt')
    #~ shutil.copyfile('./data/' + objPath + '/solver_activations.txt', './data/' + objPath + '/solver_activations' + str(rangeStart) + str(rangeEnd) + '.txt')

    #~ shutil.copyfile('./data/' + objPath + '/lenet_iter_20000.caffemodel', './data/' + objPath + '/lenet_iter_20000_' + str(rangeStart) + str(rangeEnd) + '.caffemodel')
    #~ shutil.copyfile('./data/' + objPath + '/lenet_iter_20000.solverstate', './data/' + objPath + '/lenet_iter_20000_' + str(rangeStart) + str(rangeEnd) + '.solverstate')


#~ objPath = 'bb_singleobj_test_orthnorm/pepto_bismol_clsLearning'
#~ objPath = 'bb_singleobj_test_orthnorm/softsoap_gold_clsLearning'
#~ objPath = 'bb_singleobj_test_orthnorm/white_rain_sensations_apple_blossom_hydrating_body_wash_clsLearning'
#~ objPath = ''
objPath = sys.argv[1]

#~ trainThisObject(objPath,0,16) # all w/o occlusion
#~ trainThisObject(objPath,8,10) # just kappler
#~ trainThisObject(objPath,0,5) # just orth2curve part
trainThisObject(objPath,0,15) # all
#~ trainThisObject(objPath,0,3)

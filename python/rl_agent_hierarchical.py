'''Reinforcement learning (RL) agent and related utility functions.'''

# python
import os
from copy import copy
from time import time
# scipy
from matplotlib import pyplot
from numpy.random import choice, permutation, rand, randint
from numpy import argmax, arange, array, exp, flipud, logical_not, isinf, mean, pi, stack, where, zeros
# caffe
import h5py
import caffe
from caffe.proto import caffe_pb2
# self
import hand_descriptor
from rl_agent import RlAgent
from hand_descriptor import HandDescriptor

# AGENT ============================================================================================

class RlAgentHierarchical(RlAgent):

  def __init__(self, rlEnvironment, gpuId, hasHistory, nSamples):
    '''Initializes agent in the given environment.'''

    RlAgent.__init__(self, rlEnvironment, gpuId)

    # parameters
    self.nSamples = nSamples
    self.hasHistory = hasHistory
    self.emptyStateImage = zeros((3, 60, 60), dtype='float')
    self.emptyStateVector = zeros(6, dtype='float')
    self.emptyState = [self.emptyStateImage, self.emptyStateVector]
    self.minValue = -float('inf'); self.maxValue = float('inf')

    # network parameters
    self.conv1aOutputs = 12 # LeNet 20 outputs
    self.conv1sOutputs = 12 # LeNet 20 outputs
    self.conv2aOutputs = 24 # LeNet 50 outputs
    self.conv2sOutputs = 24 # LeNet 50 outputs
    self.ip1aOutputs = 200 # LeNet 500 outputs
    self.ip1sOutputs = 200 # LeNet 500 outputs
    self.ip1Outputs = 50 # MarcNet 60 outputs
    self.ip2Outputs = 50 # MarcNet 60 outputs

  def ComputeMinAndMaxReturns(self, D):
    '''Sets minValue and maxValue to what is in the database.'''

    values = [D[idx][2] for idx in xrange(len(D))]
    self.minValue = min(values) if isinf(self.minValue) else min(self.minValue, min(values))
    self.maxValue = max(values) if isinf(self.maxValue) else max(self.maxValue, max(values))
    return values

  def DownsampleData(self, D, batchSize):
    '''Samples data to a batch size, uniformly at random, and calls LabelData.'''

    if len(D) <= batchSize:
      return D

    idxs = choice(len(D), batchSize, replace=False)
    return [D[i] for i in idxs]

  def EvaluateActions(self, state, actionImage, actionVectors):
    '''Run forward propagation and get approximated action-values.'''

    for i in xrange(actionVectors.shape[0]):
      self.caffeNet.blobs["actionImage"].data[i] = actionImage
      self.caffeNet.blobs["actionVector"].data[i] = actionVectors[i]
    if self.hasHistory:
      for i in xrange(actionVectors.shape[0]):
        self.caffeNet.blobs["stateImage"].data[i] = state[0]
        self.caffeNet.blobs["stateVector"].data[i] = state[1]

    self.caffeNet.forward()
    values = self.caffeNet.blobs["ip3"].data.flatten()
    return values[0:self.nSamples]

  def GenerateFileNameLists(self, fileNameListTrain, fileNameListTest):
    '''Generates two text files pointing to train.h5 and test.h5, needed by caffe.'''

    with open(fileNameListTrain, 'w') as fileHandle:
      fileHandle.write(self.caffeDir + "train.h5")
    with open(fileNameListTest, 'w') as fileHandle:
      fileHandle.write(self.caffeDir + "test.h5")

  def GenerateSolverPrototxt(self):
    '''Generates the solver prototxt file for training in Caffe.'''

    solver = caffe_pb2.SolverParameter()

    # net files
    solver.train_net = self.caffeTrainModelFileName
    solver.test_net.append(self.caffeTestModelFileName)

    # solver
    solver.type = "SGD"
    solver.solver_mode = caffe_pb2.SolverParameter.GPU

    # intervals
    solver.test_iter.append(100)
    solver.test_interval = 100000
    solver.snapshot = 3000
    solver.snapshot_prefix = self.caffeWeightsFilePrefix[:-6]

    # learning rate
    solver.lr_policy = "step"
    solver.base_lr = 0.00025
    solver.gamma = 0.5
    solver.stepsize = 1000
    solver.display = 100
    solver.max_iter = 450000
    solver.momentum = 0.9
    solver.weight_decay = 0.0005

    # write to file
    with open(self.caffeSolverFileName, 'w') as solverFile:
      solverFile.write(str(solver))

  def GenerateModelFile(self, netSpec, modelFileName):
    '''Generates model prototxt file for caffe. (Assumes inputs are already generated).'''

    # ACTION CNN
    netSpec.conv1a = caffe.layers.Convolution(netSpec.actionImage, num_output=self.conv1aOutputs,
      kernel_size=5, group=3, weight_filler={"type":"xavier"}, param=[{"decay_mult":1},
      {"decay_mult":0}])
    netSpec.relu1a = caffe.layers.ReLU(netSpec.conv1a, in_place=True)
    netSpec.pool1a = caffe.layers.Pooling(netSpec.conv1a, pool=caffe.params.Pooling.MAX,
      kernel_size=2, stride=2)
    netSpec.conv2a = caffe.layers.Convolution(netSpec.pool1a, num_output=self.conv2aOutputs,
      kernel_size=5, group=3, weight_filler={"type":"xavier"}, param=[{"decay_mult":1},
      {"decay_mult":0}]) # LeNet 50 outputs
    netSpec.relu2a = caffe.layers.ReLU(netSpec.conv2a, in_place=True)
    netSpec.pool2a = caffe.layers.Pooling(netSpec.conv2a, pool=caffe.params.Pooling.MAX,
      kernel_size=2, stride=2)
    netSpec.ip1a = caffe.layers.InnerProduct(netSpec.pool2a, num_output=self.ip1aOutputs,
      weight_filler={"type":"xavier"},  param=[{"decay_mult":1}, {"decay_mult":0}])
    netSpec.relu3a = caffe.layers.ReLU(netSpec.ip1a, in_place=True)

    # STATE CNN
    if self.hasHistory:
      netSpec.conv1s = caffe.layers.Convolution(netSpec.stateImage, num_output=self.conv1sOutputs,
        kernel_size=5, group=3, weight_filler={"type":"xavier"}, param=[{"decay_mult":1},
        {"decay_mult":0}])
      netSpec.relu1s = caffe.layers.ReLU(netSpec.conv1s, in_place=True)
      netSpec.pool1s = caffe.layers.Pooling(netSpec.conv1s, pool=caffe.params.Pooling.MAX,
        kernel_size=2, stride=2)
      netSpec.conv2s = caffe.layers.Convolution(netSpec.pool1s, num_output=self.conv2sOutputs,
        kernel_size=5, group=3, weight_filler={"type":"xavier"}, param=[{"decay_mult":1},
        {"decay_mult":0}])
      netSpec.relu2s = caffe.layers.ReLU(netSpec.conv2s, in_place=True)
      netSpec.pool2s = caffe.layers.Pooling(netSpec.conv2s, pool=caffe.params.Pooling.MAX,
        kernel_size=2, stride=2)
      netSpec.ip1s = caffe.layers.InnerProduct(netSpec.pool2s, num_output=self.ip1sOutputs,
        weight_filler={"type":"xavier"},  param=[{"decay_mult":1}, {"decay_mult":0}])
      netSpec.relu3s = caffe.layers.ReLU(netSpec.ip1s, in_place=True)

    # CONNECTION
    bottomLayers = [netSpec.ip1a, netSpec.actionVector]
    if self.hasHistory: bottomLayers = [netSpec.ip1s, netSpec.stateVector] + bottomLayers
    netSpec.concat = caffe.layers.Concat(*bottomLayers)

    # MARCNET
    netSpec.ip1 = caffe.layers.InnerProduct(netSpec.concat, num_output=self.ip1Outputs,
      weight_filler={"type":"xavier"},  param=[{"decay_mult":1}, {"decay_mult":0}])
    netSpec.relu1 = caffe.layers.ReLU(netSpec.ip1, in_place=True)
    netSpec.ip2 = caffe.layers.InnerProduct(netSpec.ip1, num_output=self.ip2Outputs,
      weight_filler={"type":"xavier"},  param=[{"decay_mult":1}, {"decay_mult":0}])
    netSpec.relu2 = caffe.layers.ReLU(netSpec.ip2, in_place=True)
    netSpec.ip3 = caffe.layers.InnerProduct(netSpec.ip2, num_output=1,
      weight_filler={"type":"xavier"},  param=[{"decay_mult":1}, {"decay_mult":0}])

    # LOSS
    if hasattr(netSpec, "label"):
      netSpec.loss = caffe.layers.EuclideanLoss(netSpec.ip3, netSpec.label)

    # WRITE TO FILE
    with open(modelFileName, 'w') as modelFile:
      modelFile.write(str(netSpec.to_proto()))

  def InitializeCaffe(self, caffeDirPostfix):
    '''Generates prototxt files for Caffe network specifications.'''

    # set file names
    levelString = str(self.level)
    self.caffeDir = os.getcwd() + "/caffe/" + caffeDirPostfix
    self.caffeTrainDataFileName = self.caffeDir + "train.h5"
    self.caffeTestDataFileName = self.caffeDir + "test.h5"
    fileNameListTest = self.caffeDir + "fileNameList-test.txt"
    fileNameListTrain = self.caffeDir + "fileNameList-train.txt"
    self.caffeWeightsFilePrefix = self.caffeDir + "level" + levelString + "_iter_"
    self.caffeSolverFileName = self.caffeDir + "solver-level" + levelString + ".prototxt"
    self.caffeDeployModelFileName = self.caffeDir + "deploy-level" + levelString + ".prototxt"
    self.caffeTrainModelFileName = self.caffeDir + "train-level" + levelString + ".prototxt"
    self.caffeTestModelFileName = self.caffeDir + "test-level" + levelString + ".prototxt"

    # generate deploy model file
    netDeploy = caffe.NetSpec()
    netDeploy.actionImage = caffe.layers.Input(
      input_param={'shape':{'dim':[self.nSamples, 3, 60, 60]}})
    netDeploy.actionVector = caffe.layers.Input(
      input_param={'shape':{'dim':[self.nSamples, 6]}})
    if self.hasHistory:
      netDeploy.stateImage = caffe.layers.Input(
        input_param={'shape':{'dim':[self.nSamples, 3, 60, 60]}})
      netDeploy.stateVector = caffe.layers.Input(
        input_param={'shape':{'dim':[self.nSamples, 6]}})
    self.GenerateModelFile(netDeploy, self.caffeDeployModelFileName)

    # generate train model file
    netTrain = caffe.NetSpec()
    netTrain.actionImage = caffe.layers.HDF5Data(source=fileNameListTrain, batch_size=32)
    netTrain.actionVector = caffe.layers.HDF5Data(source=fileNameListTrain, batch_size=32)
    if self.hasHistory:
      netTrain.stateImage = caffe.layers.HDF5Data(source=fileNameListTrain, batch_size=32)
      netTrain.stateVector = caffe.layers.HDF5Data(source=fileNameListTrain, batch_size=32)
    netTrain.label = caffe.layers.HDF5Data(source=fileNameListTrain, batch_size=32)
    self.GenerateModelFile(netTrain, self.caffeTrainModelFileName)

    # generate test model file
    netTest = caffe.NetSpec()
    netTest.actionImage = caffe.layers.HDF5Data(source=fileNameListTest, batch_size=32)
    netTest.actionVector = caffe.layers.HDF5Data(source=fileNameListTest, batch_size=32)
    if self.hasHistory:
      netTest.stateImage = caffe.layers.HDF5Data(source=fileNameListTest, batch_size=32)
      netTest.stateVector = caffe.layers.HDF5Data(source=fileNameListTest, batch_size=32)
    netTest.label = caffe.layers.HDF5Data(source=fileNameListTest, batch_size=32)
    self.GenerateModelFile(netTest, self.caffeTestModelFileName)

    # generate solver
    self.GenerateSolverPrototxt()

    # generate lists pointing to train.h5 and test.h5
    self.GenerateFileNameLists(fileNameListTrain, fileNameListTest)

    # initialize network
    self.caffeNet = caffe.Net(self.caffeDeployModelFileName, caffe.TEST)

  def LoadExperienceDatabase(self):
    '''Loads the experience database from an HDF5 file.'''

    startTime = time()

    fileName = self.caffeDir + "ExperienceDatabase" + str(self.level) + ".h5"

    experienceDatabase = []
    with h5py.File(fileName, 'r') as fileHandle:

      # load datasets
      dai  = fileHandle["actionImage"]
      dav  = fileHandle["actionVector"]
      if self.hasHistory:
        dsi  = fileHandle["stateImage"]
        dsv  = fileHandle["stateVector"]
      dr = fileHandle["reward"]

      # save data
      rewards = zeros(dr.shape[0])
      for i in xrange(dr.shape[0]):
        s = (dsi[i,:,:,:], dsv[i,:]) if self.hasHistory else self.emptyState
        a = (dai[i,:,:,:], dav[i,:])
        r = dr[i]; rewards[i] = r
        experienceDatabase.append((s, a, r))

      # update range of values
      self.minValue = min(rewards)
      self.maxValue = max(rewards)
      if self.minValue == self.maxValue:
        self.minValue = -float('inf')
        self.maxValue = float('inf')

    print("Took {}s to load experience database.".format(time()-startTime))
    print("Loaded experiences from {} successfully.".format(fileName))

    return experienceDatabase

  def LoadNetworkWeights(self):
    '''Loads the network weights from the specified file name.'''

    weightsFileName = self.caffeWeightsFilePrefix + "3000.caffemodel"
    self.caffeNet = caffe.Net(self.caffeDeployModelFileName, caffe.TEST, weights=weightsFileName)
    print("Loaded file " + weightsFileName + " successfully.")

  def PlotImages(self, s, a):
    '''Produces plots of the state image.'''

    # Setup

    Is = copy(s[0])
    Ia = copy(a[0])
    coords = copy(a[1])

    coordsIsPoint = self.level == 0 or self.level == 1 or self.level == 5

    fig = pyplot.figure()

    # Plot State Images

    for i in xrange(3):
      pyplot.subplot(2, 3, i+1)
      pyplot.imshow(Is[i,:,:], cmap='gray')

    # Plot Action Images

    for i in xrange(3):

      Ir = copy(Ia[i,:,:])
      Ig = copy(Ia[i,:,:])
      Ib = copy(Ia[i,:,:])

      if coordsIsPoint:
        # draw the point the robot chose
        offsets = [(0,0),(1,0),(-1,0),(0,1),(0,-1),\
                         (2,0),(-2,0),(0,2),(0,-2),
                         (3,0),(-3,0),(0,3),(0,-3)]
        for offset in offsets:
          idx = int(coords[2*i+0]) + offset[0]
          jdx = int(coords[2*i+1]) + offset[1]
          if idx >= 0 and jdx >= 0 and idx < Ir.shape[0] and jdx < Ir.shape[1]:
            Ir[idx, jdx] = 1.0; Ig[idx, jdx] = 0.0; Ib[idx, jdx] = 0.0
      else:
        # draw the angle the robot chose
        idx = int(((coords[i]+pi) / (2*pi)) * (Ib.shape[0]-1))
        idxs = [idx-1, idx, idx+1]
        for idx in idxs:
          if idx >= 0 and idx < Ir.shape[1]:
            Ir[:, idx] = 1.0

      Irgb = stack((Ir, Ig, Ib), 2)
      pyplot.subplot(2, 3, i+4)
      pyplot.imshow(Irgb)

    fig.suptitle("(Top.) State Image. (Bottom.) Action image.")

    for i in xrange(6):
      fig.axes[i].set_xticks([])
      fig.axes[i].set_yticks([])

    pyplot.show(block=True)

  def PlotValues(self, action, values):
    '''Plots values as a function of the action.'''

    print action.shape
    print values.shape

    pyplot.plot(action, values, '-x')
    pyplot.xlabel("Action")
    pyplot.ylabel("Value")
    pyplot.title("Action-Values")
    pyplot.show(block=True)

  def PruneDatabase(self, replayDatabase, maxEntries):
    '''Removes oldest items in the database until the size is no more than maxEntries.'''

    self.ComputeMinAndMaxReturns(replayDatabase)

    if len(replayDatabase) <= maxEntries:
      return replayDatabase

    return replayDatabase[len(replayDatabase)-maxEntries:]

  def SaveExperienceDatabase(self, D):
    '''Saves the experience database to an HDF5 file.'''

    if len(D) == 0: return

    startTime = time()

    # delete training files (prevents crashing due to space shortage)
    if os.path.isfile(self.caffeTrainDataFileName):
      os.remove(self.caffeTrainDataFileName)
    if os.path.isfile(self.caffeTestDataFileName):
      os.remove(self.caffeTestDataFileName)

    # determine data sizes
    databaseFileName = self.caffeDir + "ExperienceDatabase" + str(self.level) + ".h5"
    stateImageShape = D[0][0][0].shape
    stateVectorShape = D[0][0][1].shape
    actionImageShape = D[0][1][0].shape
    actionVectorShape = D[0][1][1].shape

    with h5py.File(databaseFileName, 'w') as fileHandle:

      # create datasets
      dai  = fileHandle.create_dataset("actionImage", (len(D),)+actionImageShape, 'f')
      dav  = fileHandle.create_dataset("actionVector", (len(D),)+actionVectorShape, 'f')
      if self.hasHistory:
        dsi  = fileHandle.create_dataset("stateImage", (len(D),)+stateImageShape, 'f')
        dsv  = fileHandle.create_dataset("stateVector", (len(D),)+stateVectorShape, 'f')
      dr   = fileHandle.create_dataset("reward", (len(D),), 'f')

      # save data
      for i, d in enumerate(D):
        dai[i,:] = d[1][0]
        dav[i,:] = d[1][1]
        if self.hasHistory:
          dsi[i,:] = d[0][0]
          dsv[i,:] = d[0][1]
        dr[i]    = d[2]

    print("Took {}s to save experience database.".format(time()-startTime))

  def SelectIndexEpsilonGreedy(self, state, actionImage, actionChoices, epsilon):
    '''Evaluates actions or (with probability epsilon) randomly chooses an action.'''

    if rand() < epsilon:
      bestIdx = randint(actionChoices.shape[0])
      bestValue = float('NaN')
      #bestValue = self.EvaluateActions(state, actionImage, actionChoices[bestIdx])
    else:
      values = self.EvaluateActions(state, actionImage, actionChoices)
      bestIdx = argmax(values)
      bestValue = values[bestIdx]

    #if self.level == 3: self.PlotValues(actionChoices[:,1]*(180/pi), values)

    return bestIdx, bestValue

  def SelectIndexSoftmax(self, values, tau):
    '''Select index into values according to softmax distribution and temperature parameter tau.
    - Input values: Values returned from Q-function.
    - Input tau: Number in (0, Inf) specifying how much exploration should take place.
      (Lower values means more determinism.)
    - Returns: Index into values representing the value that was selected.
    '''

    # correct values outside feasible range
    values[values < self.minValue] = self.minValue
    values[values > self.maxValue] = self.maxValue

    pChoice = exp(values / tau) / sum(exp(values / tau)) # softmax
    return choice(len(values), p=pChoice)

  def SelectIndexSoftTieBreaking(self, values, tau):
    '''Select index into values breaking ties randomly.
    - Input values: Values returned from Q-function.
    - Input tau: Number in [0, 1] specifying how much exploration should take place. A value of 0
      means only exact ties are broken (less exploration). Larger values indicate the range within
      which two values are considered equal (more exploration).
    - Returns: Index into values representing the value that was selected.
    '''

    # correct values outside feasible range
    values[values < self.minValue] = self.minValue
    values[values > self.maxValue] = self.maxValue

    bestIdxs = where(values >= (max(values)-tau))[0]
    return bestIdxs[randint(len(bestIdxs))]

  def SelectIndicesTopN(self, values, n):
    '''Selects indicies of the top n values. Returns indices and values.'''

    idxs = values.argsort()
    bestIdxs = idxs[-n:]
    return bestIdxs, values[bestIdxs]

  def SenseAndActSample(self, hand, prevDescs, cloudTree, tau):
    '''Senses the current state and determines a sampling of next actions.'''

    states = []; actions = []; descriptors = []; values = []
    for prevDesc in prevDescs:
      s, a, d, v = self.SenseAndAct(hand, prevDesc, cloudTree, tau)
      states.append(s); actions.append(a); descriptors.append(d); values.append(v)

    return states, actions, descriptors, array(values)

  def Train(self, Dl, recordLoss=True, stepSize=100, nIterations=3000):
    '''Trains the network on the provided data labels.
      - Input Dl: List of tuples with (state,action,value).
      - Input recordLoss: If true, saves train and test return values (takes longer).
      - Input stepSize: Records loss values this often.
      - Input nIterations: Number of training iterations to run.
      - Returns: train loss, test loss.
    '''

    if len(Dl) == 0:
      return

    # 1. Write data to files
    startTime = time()

    # split data into train/test
    nTest = int(len(Dl)/4.0)
    nTrain = len(Dl) - nTest

    # shuffle data
    pdxs = permutation(len(Dl))
    idxs = pdxs[0:nTrain]
    jdxs = pdxs[nTrain:]

    # determine sizes
    sampleActionImage = Dl[0][1][0]
    nia = sampleActionImage.shape
    nva = len(Dl[0][1][1])
    if self.hasHistory:
      sampleStateImage = Dl[0][0][0]
      nis = sampleStateImage.shape
      nvs = len(Dl[0][0][1])

    # write training data to file
    with h5py.File(self.caffeTrainDataFileName, 'w') as fileHandle:

      Ia = fileHandle.create_dataset("actionImage", (nTrain, nia[0], nia[1], nia[2]), 'f')
      Va = fileHandle.create_dataset("actionVector", (nTrain, nva), 'f')
      if self.hasHistory:
        Is = fileHandle.create_dataset("stateImage", (nTrain, nis[0], nis[1], nis[2]), 'f')
        Vs = fileHandle.create_dataset("stateVector", (nTrain, nvs), 'f')
      L = fileHandle.create_dataset("label", (nTrain,), 'f')

      for i, idx in enumerate(idxs):
        Ia[i, :, :, :] = Dl[idx][1][0]
        Va[i, :] = Dl[idx][1][1]
        L[i] = Dl[idx][2]
      if self.hasHistory:
        for i, idx in enumerate(idxs):
          Is[i, :, :, :] = Dl[idx][0][0]
          Vs[i, :] = Dl[idx][0][1]

    # write test data to file
    with h5py.File(self.caffeTestDataFileName, 'w') as fileHandle:

      Ia = fileHandle.create_dataset("actionImage", (nTest, nia[0], nia[1], nia[2]), 'f')
      Va = fileHandle.create_dataset("actionVector", (nTest, nva), 'f')
      if self.hasHistory:
        Is = fileHandle.create_dataset("stateImage", (nTest, nis[0], nis[1], nis[2]), 'f')
        Vs = fileHandle.create_dataset("stateVector", (nTest, nvs), 'f')
      L = fileHandle.create_dataset("label", (nTest,), 'f')

      for j, jdx in enumerate(jdxs):
        Ia[j, :, :, :] = Dl[jdx][1][0]
        Va[j, :] = Dl[jdx][1][1]
        L[j] = Dl[jdx][2]
      if self.hasHistory:
        for j, jdx in enumerate(jdxs):
          Is[j, :, :, :] = Dl[jdx][0][0]
          Vs[j, :] = Dl[jdx][0][1]

    print("Took {}s to write to hdf5 files.".format(time()-startTime))

    # 2. Optimize
    startTime = time()

    weightsFileName = self.caffeWeightsFilePrefix + str(nIterations) + ".caffemodel"
    solver = caffe.SGDSolver(self.caffeSolverFileName)

    if self.caffeFirstTrain:
      self.caffeFirstTrain = False
    else:
      solver.net.copy_from(weightsFileName)

    trainLoss = []; testLoss = []

    if recordLoss:

      for iteration in xrange(int(nIterations/stepSize)):
        solver.step(stepSize)
        loss = float(solver.net.blobs["loss"].data)
        trainLoss.append(loss)

        loss = 0
        for testIteration in xrange(stepSize):
          solver.test_nets[0].forward()
          loss += float(solver.test_nets[0].blobs["loss"].data)
        loss /= stepSize
        testLoss.append(loss)

    else:

      solver.step(nIterations)

    self.caffeNet = caffe.Net(self.caffeDeployModelFileName, caffe.TEST, weights=weightsFileName)

    print("Training took {}s.".format(time()-startTime))
    return trainLoss, testLoss
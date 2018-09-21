#!/usr/bin/env python
'''Trains the robot to locate stable placements.'''

# python
import os
import time
# scipy
from scipy.io import savemat
from numpy import array, mean
from numpy.random import randint
from scipy.spatial import cKDTree
# self
from rl_environment_placing import RlEnvironmentPlacing
from rl_agent_level0 import RlAgentLevel0
from rl_agent_level1 import RlAgentLevel1
from rl_agent_level2 import RlAgentLevel2
from rl_agent_level3 import RlAgentLevel3
from rl_agent_level4 import RlAgentLevel4
from rl_agent_level5 import RlAgentLevel5
import point_cloud

def main():
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================

  # system
  gpuId = 0
  nAgents = 6

  # objects
  nObjects = 10
  objectFolder = os.getcwd() + "/../../Data/RaveObjects/RectangularBlocks"
  graspColor = [0,0,1]

  # reaching
  nActionSamples = [500, 1000, 361, 181, 361, 303]
  stateImageWHD = [0.09, 0.09, 0.22]

  # view
  viewCenter = array([0,0,0])
  viewKeepout = 0.70
  viewWorkspace = [(-1.0,1.0),(-1.0,1.0),(-1.0,1.0)]
  viewWorkspaceNoTable = [(-1.0,1.0),(-1.0,1.0),(0.002,1.0)]

  # learning
  nTrainingRounds = 200
  nScenes = 100
  nEpisodes = 20
  minEpsilon = 0.05
  unbiasOnRound = nTrainingRounds-5
  maxExperiences = 50000
  trainingBatchSize = maxExperiences
  minTrainBatchSize = 1024

  # visualization/saving
  saveFileName = "results.mat"
  recordLoss = True
  loadNetwork = False
  loadDatabase = False
  showViewer = False
  showSteps = False
  plotImages = False

  # visualize policy
  visualizePolicy = False
  if visualizePolicy:
    nEpisodes = 1
    unbiasOnRound = 0
    loadNetwork = True
    loadDatabase = False
    showViewer = True
    showSteps = True
    plotImages = True

  # INITIALIZATION =================================================================================

  rlEnv = RlEnvironmentPlacing(showViewer)

  experiences = []
  rlAgents = []
  for i in xrange(nAgents):
    rlAgent = eval("RlAgentLevel" + str(i) + \
      "(rlEnv, gpuId, False, nActionSamples[i], \"weights-grasp-only-blocks/\")")
    rlAgents.append(rlAgent)
    experiences.append([])
  rlAgents[0].SetSamplingWorkspace(-viewWorkspaceNoTable[2][1], -viewWorkspaceNoTable[2][0])

  if loadNetwork:
    for rlAgent in rlAgents:
      rlAgent.LoadNetworkWeights()
      rlAgent.caffeFirstTrain = False

  if loadDatabase:
    for i, rlAgent in enumerate(rlAgents):
      experiences[i] = rlAgent.LoadExperienceDatabase()

  # RUN TEST =======================================================================================

  avgReturn = []; epsilonRound = []; databaseSize = []; roundTime = []; trainLoss = []; testLoss = []

  for i in xrange(len(rlAgents)):
    trainLoss.append([]); testLoss.append([])

  for trainingRound in xrange(nTrainingRounds):

    # initialization
    iterationStartTime = time.time()
    Return = []

    # cool exploration and check if it's time to unbias data
    if trainingRound >= unbiasOnRound:
      epsilon = 0.0
    else:
      epsilon = max(minEpsilon, 1.0 - float(len(experiences[ 0]))/float(maxExperiences))

    epsilonRound.append(epsilon)

    for scene in xrange(nScenes):

      # place random object in random orientation on table
      rlAgents[0].MoveHandToHoldingPose()
      objHandles = rlEnv.PlaceObjects(randint(2, nObjects+1), objectFolder)
      if showSteps: raw_input("Placed objects.")

      # get a point cloud
      cloudScene, cloudTreeScene = rlAgents[0].GetDualCloud(viewCenter, viewKeepout, viewWorkspace)
      rlAgents[0].PlotCloud(cloudScene)
      if showSteps: raw_input("Acquired point cloud.")

      for episode in xrange(nEpisodes):

        states = []; actions = []; graspDesc = None

        # grasp an object
        for rlAgent in rlAgents:
          s, a, graspDesc, v = rlAgent.SenseAndAct(None, graspDesc, cloudTreeScene, epsilon)
          rlAgents[0].PlotDescriptors([graspDesc], graspColor)
          if plotImages: rlAgent.PlotImages(s, a)
          states.append(s); actions.append(a)

        # evaluate grasp
        r = rlEnv.TransitionGraspHalfConditions(graspDesc, rlAgents[-1], objHandles)
        if showSteps: raw_input("Grasp received reward {}.".format(r))

        # save experiences
        for i in xrange(nAgents):
          experiences[i].append((states[i], actions[i], r))

        # cleanup this episode
        print("Episode {}.{}.{} had return {}".format(trainingRound, scene, episode, r))
        Return.append(r)

      # cleanup this scene
      rlEnv.RemoveObjectSet(objHandles)

    # Train each agent.
    for i, rlAgent in enumerate(rlAgents):
      if len(experiences[i]) < minTrainBatchSize: continue
      experiences[i] = rlAgent.PruneDatabase(experiences[i], maxExperiences)
      Dl = rlAgent.DownsampleData(experiences[i], trainingBatchSize)
      loss = rlAgent.Train(Dl, recordLoss=recordLoss)
      trainLoss[i].append(loss[0]); testLoss[i].append(loss[1])

    # Save results
    avgReturn.append(mean(Return))
    databaseSize.append(len(experiences[0]))
    roundTime.append(time.time()-iterationStartTime)

    saveData = {"gpuId":gpuId, "nAgents":nAgents, "nObjects":nObjects, "objectFolder":objectFolder,
      "viewCenter":viewCenter, "viewKeepout":viewKeepout, "viewWorkspace":viewWorkspace,
      "viewWorkspaceNoTable":viewWorkspaceNoTable, "nActionSamples":nActionSamples,
      "stateImageWHD":stateImageWHD, "nTrainingRounds":nTrainingRounds, "nScenes":nScenes,
      "nEpisodes":nEpisodes, "epsilonRound":epsilonRound, "minEpsilon":minEpsilon,
      "unbiasOnRound":unbiasOnRound, "maxExperiences":maxExperiences,
      "trainingBatchSize":trainingBatchSize, "minTrainBatchSize":minTrainBatchSize,
      "avgReturn":avgReturn, "databaseSize":databaseSize, "roundTime":roundTime,
      "trainLoss0": trainLoss[ 0], "testLoss0": testLoss[ 0],
      "trainLoss1": trainLoss[ 1], "testLoss1": testLoss[ 1],
      "trainLoss2": trainLoss[ 2], "testLoss2": testLoss[ 2],
      "trainLoss3": trainLoss[ 3], "testLoss3": testLoss[ 3],
      "trainLoss4": trainLoss[ 4], "testLoss4": testLoss[ 4],
      "trainLoss5": trainLoss[ 5], "testLoss5": testLoss[ 5]}
    savemat(saveFileName, saveData)

    # Backup experience database
    if (trainingRound == nTrainingRounds-1) or (trainingRound % 10 == 9):
      for i, rlAgent in enumerate(rlAgents):
        rlAgent.SaveExperienceDatabase(experiences[i])

if __name__ == "__main__":
  main()
  exit()

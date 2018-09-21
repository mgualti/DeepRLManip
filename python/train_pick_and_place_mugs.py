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
  nObjects = 5
  objectFolder = os.getcwd() + "/../../Data/RaveObjects/Mugs"
  graspColor = [0,0,1]
  placeColor = [1,0,0]

  # reaching
  nActionSamples = [500, 1000, 361, 181, 361, 303]
  stateImageWHD = [0.20, 0.20, 0.20]

  # view
  viewCenter = array([0,0,0])
  viewKeepout = 0.70
  viewWorkspace = [(-1.0,1.0),(-1.0,1.0),(-1.0,1.0)]
  viewWorkspaceNoTable = [(-1.0,1.0),(-1.0,1.0),(0.002,1.0)]

  # learning
  nTrainingRounds = 250
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
  visualizePolicy = True
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
  rlAgentsGrasping = []
  for i in xrange(nAgents):
    rlAgent = eval("RlAgentLevel" + str(i) + \
      "(rlEnv, gpuId, False, nActionSamples[i], \"weights-grasp-mugs/\")")
    rlAgentsGrasping.append(rlAgent)
    experiences.append([])
  rlAgentsGrasping[0].SetSamplingWorkspace(-viewWorkspaceNoTable[2][1], -viewWorkspaceNoTable[2][0])

  rlAgentsPlacing = []
  for i in xrange(nAgents):
    rlAgent = eval("RlAgentLevel" + str(i) +
      "(rlEnv, gpuId, True, nActionSamples[i], \"weights-place-mugs/\")")
    rlAgentsPlacing.append(rlAgent)
    experiences.append([])
  rlAgentsPlacing[0].SetSamplingWorkspace(-viewWorkspaceNoTable[2][0], viewWorkspaceNoTable[2][0])

  rlAgents = rlAgentsGrasping + rlAgentsPlacing

  if loadNetwork:
    for rlAgent in rlAgents:
      rlAgent.LoadNetworkWeights()
      rlAgent.caffeFirstTrain = False

  if loadDatabase:
    for i, rlAgent in enumerate(rlAgents):
      experiences[i] = rlAgent.LoadExperienceDatabase()

  # RUN TEST =======================================================================================

  avgReturn = []; avgPlaceReward = []; avgGraspReward = [];
  epsilonGraspRound = []; epsilonPlaceRound = []
  graspDatabaseSize = []; placeDatabaseSize = []; roundTime = [];
  trainLoss = []; testLoss = []

  for i in xrange(len(rlAgents)):
    trainLoss.append([]); testLoss.append([])

  for trainingRound in xrange(nTrainingRounds):

    # initialization
    iterationStartTime = time.time()
    Return = []; placeReward = []; graspReward = []

    # cool exploration and check if it's time to unbias data
    if trainingRound >= unbiasOnRound:
      epsilonGrasp = 0.0
      epsilonPlace = 0.0
    else:
      epsilonGrasp = max(minEpsilon, 1.0 - float(len(experiences[ 0]))/float(maxExperiences))
      epsilonPlace = max(minEpsilon, 1.0 - float(len(experiences[-1])/float(maxExperiences)))

    epsilonGraspRound.append(epsilonGrasp)
    epsilonPlaceRound.append(epsilonPlace)

    for scene in xrange(nScenes):

      # place random object in random orientation on table
      rlAgents[0].MoveHandToHoldingPose()
      objHandles = rlEnv.PlaceObjects(randint(1, nObjects+1), objectFolder)
      if showSteps: raw_input("Placed objects.")

      # get a point cloud
      cloudScene, cloudTreeScene = rlAgentsPlacing[0].GetDualCloud(
        viewCenter, viewKeepout, viewWorkspace)
      rlAgentsPlacing[0].PlotCloud(cloudScene)
      if showSteps: raw_input("Acquired point cloud.")
      cloudSceneNoTable = point_cloud.FilterWorkspace(viewWorkspaceNoTable, cloudScene)
      cloudTreeSceneNoTable = cKDTree(cloudSceneNoTable) if cloudSceneNoTable.shape[0] > 0 else None

      for episode in xrange(nEpisodes):

        states = []; actions = []; graspDesc = None; placeDesc = None

        # grasp an object
        for rlAgent in rlAgentsGrasping:
          s, a, graspDesc, v = rlAgent.SenseAndAct(None, graspDesc, cloudTreeScene, epsilonGrasp)
          rlAgentsGrasping[0].PlotDescriptors([graspDesc], graspColor)
          if plotImages: rlAgent.PlotImages(s, a)
          states.append(s); actions.append(a)

        # evaluate grasp
        rGrasp = rlEnv.TransitionObjectTopGraspHalfConditions(
          graspDesc, rlAgentsGrasping[-1], objHandles)
        if showSteps: raw_input("Grasp received reward {}.".format(rGrasp))
        graspReward.append(rGrasp)

        # if grasp was a success, try a place
        if rGrasp > 0.00:

          rlEnv.SetObjectPoses(objHandles)
          rlEnv.GraspObject(rlAgentsPlacing[0], graspDesc, objHandles)
          if showSteps: raw_input("Removed object.")

          # update hand contents
          graspDesc.imW = stateImageWHD[0]
          graspDesc.imH = stateImageWHD[1]
          graspDesc.imD = stateImageWHD[2]
          graspDesc.GenerateDepthImage(cloudSceneNoTable, cloudTreeSceneNoTable)

          # get a point cloud
          cloud, cloudTree = rlAgentsPlacing[0].GetDualCloud(viewCenter, viewKeepout, viewWorkspace)
          rlAgentsPlacing[0].PlotCloud(cloud)
          if showSteps: raw_input("Acquired point cloud.")

          # act at each level in the hierarchy
          for rlAgent in rlAgentsPlacing:
            s, a, placeDesc, v = rlAgent.SenseAndAct(
              graspDesc, placeDesc, cloudTree, epsilonPlace)
            rlAgentsPlacing[0].PlotDescriptors([placeDesc], placeColor)
            if plotImages: rlAgent.PlotImages(s, a)
            states.append(s); actions.append(a)
          if showSteps: raw_input("Decided place pose.")

          # check place and finish
          rPlace = rlEnv.TransitionPlaceUprightOnTable(
            placeDesc, rlAgentsPlacing[-1], showSteps)
          if showSteps: raw_input("Place received reward {}.".format(rPlace))
          placeReward.append(rPlace)
          rlEnv.ResetObjectPoses(objHandles)

        else: # grasp was a failure
          rPlace = 0.0

        # save experiences
        for i in xrange(nAgents):
          experiences[i].append((states[i], actions[i], rGrasp + rPlace))
        if len(states) > nAgents:
          for i in xrange(nAgents, 2*nAgents):
            experiences[i].append((states[i], actions[i], rPlace))

        # cleanup this episode
        r = rGrasp + rPlace
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
    if len(placeReward) == 0: placeReward.append(0)
    avgPlaceReward.append(mean(placeReward))
    avgGraspReward.append(mean(graspReward))
    graspDatabaseSize.append(len(experiences[0]))
    placeDatabaseSize.append(len(experiences[-1]))
    roundTime.append(time.time()-iterationStartTime)

    saveData = {"gpuId":gpuId, "nAgents":nAgents, "nObjects":nObjects, "objectFolder":objectFolder,
      "viewCenter":viewCenter, "viewKeepout":viewKeepout, "viewWorkspace":viewWorkspace,
      "viewWorkspaceNoTable":viewWorkspaceNoTable, "nActionSamples":nActionSamples,
      "stateImageWHD":stateImageWHD, "nTrainingRounds":nTrainingRounds, "nScenes":nScenes,
      "nEpisodes":nEpisodes, "epsilonGraspRound":epsilonGraspRound,
      "epsilonPlaceRound":epsilonPlaceRound, "minEpsilon":minEpsilon, "unbiasOnRound":unbiasOnRound,
      "maxExperiences":maxExperiences, "trainingBatchSize":trainingBatchSize,
      "minTrainBatchSize":minTrainBatchSize, "avgReturn":avgReturn, "avgPlaceReward":avgPlaceReward,
      "avgGraspReward":avgGraspReward, "graspDatabaseSize":graspDatabaseSize,
      "placeDatabaseSize":placeDatabaseSize, "roundTime":roundTime,
      "trainLoss0": trainLoss[ 0], "testLoss0": testLoss[ 0],
      "trainLoss1": trainLoss[ 1], "testLoss1": testLoss[ 1],
      "trainLoss2": trainLoss[ 2], "testLoss2": testLoss[ 2],
      "trainLoss3": trainLoss[ 3], "testLoss3": testLoss[ 3],
      "trainLoss4": trainLoss[ 4], "testLoss4": testLoss[ 4],
      "trainLoss5": trainLoss[ 5], "testLoss5": testLoss[ 5],
      "trainLoss6": trainLoss[ 6], "testLoss6": testLoss[ 6],
      "trainLoss7": trainLoss[ 7], "testLoss7": testLoss[ 7],
      "trainLoss8": trainLoss[ 8], "testLoss8": testLoss[ 8],
      "trainLoss9": trainLoss[ 9], "testLoss9": testLoss[ 9],
      "trainLoss10":trainLoss[10], "testLoss10":testLoss[10],
      "trainLoss11":trainLoss[11], "testLoss11":testLoss[11]}
    savemat(saveFileName, saveData)

    # Backup experience database
    if (trainingRound == nTrainingRounds-1) or (trainingRound % 10 == 9):
      for i, rlAgent in enumerate(rlAgents):
        rlAgent.SaveExperienceDatabase(experiences[i])

if __name__ == "__main__":
  main()
  exit()

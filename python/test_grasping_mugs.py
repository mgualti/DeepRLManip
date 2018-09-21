#!/usr/bin/env python
'''Trains the robot to locate stable placements.'''

# python
import os
import time
# scipy
from scipy.io import savemat
from numpy import argmax, array, mean
from numpy.random import randint
# self
from rl_environment_placing import RlEnvironmentPlacing
from rl_agent_level0 import RlAgentLevel0
from rl_agent_level1 import RlAgentLevel1
from rl_agent_level2 import RlAgentLevel2
from rl_agent_level3 import RlAgentLevel3
from rl_agent_level4 import RlAgentLevel4
from rl_agent_level5 import RlAgentLevel5

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

  # reaching
  nActionSamples = [500, 1000, 361, 181, 361, 303]
  #nActionSamples = [100, 500, 361, 181, 361, 303]
  nTrials = 1
  #nTrials = 10

  # view
  viewCenter = array([0,0,0])
  viewKeepout = 0.70
  viewWorkspace = [(-1.0,1.0),(-1.0,1.0),(-1.0,1.0)]
  viewWorkspaceNoTable = [(-1.0,1.0),(-1.0,1.0),(0.002,1.0)]

  # learning
  nEpisodes = 1000
  epsilon = 0.0

  # visualization/saving
  saveFileName = "results.mat"
  showViewer = False
  showSteps = False
  plotImages = False

  # visualize policy
  visualizePolicy = True
  if visualizePolicy:
    nEpisodes = 1
    showViewer = True
    showSteps = True
    plotImages = True

  # INITIALIZATION =================================================================================

  rlEnv = RlEnvironmentPlacing(showViewer)

  experiences = []
  rlAgents = []
  for i in xrange(nAgents):
    rlAgent = eval("RlAgentLevel" + str(i) + \
      "(rlEnv, gpuId, False, nActionSamples[i], \"weights-grasp-only-mugs/\")")
    rlAgents.append(rlAgent)
    experiences.append([])
  rlAgents[0].SetSamplingWorkspace(-viewWorkspaceNoTable[2][1], -viewWorkspaceNoTable[2][0])

  for rlAgent in rlAgents:
    rlAgent.LoadNetworkWeights()
    rlAgent.caffeFirstTrain = False

  # RUN TEST =======================================================================================

  antipodal = []; perfect = []; episodeTime = []

  for episode in xrange(nEpisodes):

    # initialization
    episodeStartTime = time.time()

    # place random object in random orientation on table
    rlAgents[0].MoveHandToHoldingPose()
    objHandles = rlEnv.PlaceObjects(randint(2, nObjects+1), objectFolder)
    if showSteps: raw_input("Placed objects.")

    # get a point cloud
    cloudScene, cloudTreeScene = rlAgents[0].GetDualCloud(viewCenter, viewKeepout, viewWorkspace)
    rlAgents[0].PlotCloud(cloudScene)
    if showSteps: raw_input("Acquired point cloud.")
    # grasp an object
    graspDescs = [None]*nTrials
    for rlAgent in rlAgents:
      states, actions, graspDescs, values = rlAgent.SenseAndActSample(
        None, graspDescs, cloudTreeScene, epsilon)
      bestIdx = argmax(values)
      rlAgents[0].PlotDescriptors(graspDescs, graspColor)
      if plotImages: rlAgent.PlotImages(states[bestIdx], actions[bestIdx])
      if showSteps: raw_input("Showing grasp samples.")
    graspDesc = graspDescs[bestIdx]

    # evaluate grasp
    ant, antAndCf = rlEnv.TestGrasp(graspDesc, rlAgents[-1], objHandles)

    # cleanup this episode
    print("Episode {}, antipodal={}, antipodal+collisionFree={}".format(episode, ant, antAndCf))
    antipodal.append(ant); perfect.append(antAndCf)
    rlEnv.RemoveObjectSet(objHandles)

    # Save results
    episodeTime.append(time.time()-episodeStartTime)

    saveData = {"gpuId":gpuId, "nAgents":nAgents, "nObjects":nObjects, "objectFolder":objectFolder,
      "viewCenter":viewCenter, "viewKeepout":viewKeepout, "viewWorkspace":viewWorkspace,
      "viewWorkspaceNoTable":viewWorkspaceNoTable, "nActionSamples":nActionSamples,
      "nTrials":nTrials, "nEpisodes":nEpisodes, "antipodal":antipodal, "perfect":perfect}
    savemat(saveFileName, saveData)

if __name__ == "__main__":
  main()
  exit()

#!/usr/bin/env python
'''Trains the robot to locate stable grasps.'''

# python
import time
# scipy
from scipy.io import savemat
from numpy import array, mean
# self
from rl_environment_grasping import RlEnvironmentGrasping
from rl_agent import RlAgent
# matlab
from grasp_proxy_matlab import GraspProxyMatlab

def main():
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================

  # system
  gpuId = 0

  # objects
  nObjects = 10
  objectFolder = "/home/mgualti/Data/RaveObjects/RectangularBlocks"

  # view
  viewCenter = array([0,0,0])
  viewKeepout = 0.70
  viewWorkspace = [(-1.0,1.0),(-1.0,1.0),(-1.0,1.0)]

  # learning
  nEpisodes = 1000
  gpdScoreThresh = -float('Inf')
  gpdNSamples = 500

  # visualization/saving
  saveFileName = "results.mat"
  showViewer = False
  showSteps = False

  # INITIALIZATION =================================================================================

  rlEnv = RlEnvironmentGrasping(showViewer)
  rlAgent = RlAgent(rlEnv, gpuId)
  gpd = GraspProxyMatlab()

  # RUN TEST =======================================================================================

  antipodal = []; perfect = []; episodeTime = []

  for episode in xrange(nEpisodes):

    # Initialization
    episodeStartTime = time.time()

    # place random object in random orientation on table
    objHandles = rlEnv.PlaceObjects(nObjects, objectFolder)
    if showSteps: raw_input("Placed objects.")

    cloud, cloudTree, viewPoints, viewPointIndices = rlAgent.GetDualCloudAndViewPoints(
      viewCenter, viewKeepout, viewWorkspace)
    rlAgent.PlotCloud(cloud)
    if showSteps: raw_input("Acquired point cloud.")

    #SaveCloud(cloud, viewPoints, viewPointIndices, "blocks1.mat")

    # call gpd
    descriptors = gpd.DetectGrasps(
      cloud, viewPoints, viewPointIndices, gpdNSamples, gpdScoreThresh, gpuId)

    if len(descriptors) > 0:
      # choose descriptor with max score
      bestScore = -float('inf')
      for descriptor in descriptors:
        if descriptor.score > bestScore:
          desc = descriptor
          bestScore = desc.score

      rlAgent.PlotDescriptors([desc])

      # check grasp and finish
      ant, antAndCf = rlEnv.TestGrasp(desc, rlAgent, objHandles)
    else:
      print("No grasps found!")
      ant = 0.0; perfect = 0.0

    # cleanup this scene
    print("Episode {}, antipodal={}, antipodal+collisionFree={}".format(episode, ant, antAndCf))
    antipodal.append(ant); perfect.append(antAndCf)
    rlEnv.RemoveObjectSet(objHandles)
    
    # Save results
    episodeTime.append(time.time()-episodeStartTime)
    saveData = {"gpuId":gpuId, "nObjects":nObjects, "objectFolder":objectFolder,
      "viewCenter":viewCenter, "viewKeepout":viewKeepout, "viewWorkspace":viewWorkspace,
      "nEpisodes":nEpisodes, "gpdScoreThresh":gpdScoreThresh, "gpdNSamples":gpdNSamples,
      "episodeTime":episodeTime, "antipodal":antipodal, "perfect":perfect}
    savemat(saveFileName, saveData)

def SaveCloud(cloud, viewPoints, viewPointIndices, fileName):
  '''Saves point cloud information for testing in Matlab.'''

  viewPointIndices = viewPointIndices + 1 # matlab is 1-indexed
  viewPoints = viewPoints.T
  cloud = cloud.T
  data = {"cloud":cloud, "viewPoints":viewPoints, "viewPointIndices":viewPointIndices}
  savemat(fileName, data)

if __name__ == "__main__":
  main()
  exit()

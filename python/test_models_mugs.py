#!/usr/bin/env python
'''Generates mesh files and point clouds for randomly generated rectangular blocks.'''

# python
import time
# scipy
from scipy.io import savemat
from numpy import array, mean
# self
import point_cloud
from rl_agent import RlAgent
from rl_environment import RlEnvironment

def main():
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================

  # system
  gpuId = 0

  # objects
  objectScale = [0.06, 0.12]
  nObjects = 1000
  directory = "/home/mgualti/Data/3DNet/Cat10_ModelDatabase/mug_upright/"

  # view
  viewCenter = array([0,0,0])
  viewKeepout = 0.60
  viewWorkspace = [(-1.0,1.0),(-1.0,1.0),(-1.0,1.0)]

  # visualization/saving
  showViewer = False
  showSteps = False
  plotImages = False

  # INITIALIZATION =================================================================================

  rlEnv = RlEnvironment(showViewer, removeTable=True)
  rlAgent = RlAgent(rlEnv, gpuId)

  # RUN TEST =======================================================================================

  for objIdx in xrange(nObjects):

    obj = rlEnv.Place3DNetObjectAtOrigin(directory, objectScale, "mug-{}".format(objIdx), True)
    cloud, normals = rlAgent.GetFullCloudAndNormals(viewCenter, viewKeepout, viewWorkspace, True)
    point_cloud.SaveMat("mug-{}.mat".format(objIdx), cloud, normals)

    rlAgent.PlotCloud(cloud)
    if plotImages:
      point_cloud.Plot(cloud, normals, 2)

    if showSteps:
      raw_input("Placed mug-{}.".format(objIdx))

    rlEnv.RemoveObjectSet([obj])

if __name__ == "__main__":
  main()
  exit()

#!/usr/bin/env python
'''Generates mesh files and point clouds for randomly generated rectangular blocks.'''

# python
import os
import fnmatch
# scipy
# self
from rl_environment import RlEnvironment

def main():
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================

  # objects
  directory = "/home/mgualti/Data/3DNet/Cat10_ModelDatabase/mug_upright/"

  # INITIALIZATION =================================================================================

  rlEnv = RlEnvironment(showViewer=True, enablePhysics=False, removeTable=True)

  # RUN TEST =======================================================================================


  fileNames = os.listdir(directory)
  fileNames = fnmatch.filter(fileNames, "*.ply")

  for fileName in fileNames:

    threeDNetName = fileName[:-4]

    rlEnv.env.Load(directory + "/" + fileName)
    body = rlEnv.env.GetKinBody(threeDNetName)

    raw_input(threeDNetName)

    rlEnv.env.Remove(body)

if __name__ == "__main__":
  main()
  exit()

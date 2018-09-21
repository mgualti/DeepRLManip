'''Provides an interface to the Matlab grasp detector.'''

# python
import os
# scipy
from numpy import array, ascontiguousarray, fromstring, reshape, rollaxis
# matlab
import matlab
import matlab.engine
# self
import hand_descriptor
from hand_descriptor import HandDescriptor

class GraspProxyMatlab:
  '''A class for interfacing with grasp detection.'''


  def __init__(self):
    '''Starts Matlab engine.'''

    self.matlabDir = self.caffeDir = os.getcwd() + "/matlab/"

    print("Starting Matlab...")
    self.eng = matlab.engine.start_matlab()

    # add all of the required directories to the MATLAB path
    self.eng.addpath("/home/mgualti/Programs/caffe/matlab")
    self.eng.addpath(self.matlabDir + "gpd2")
    self.eng.addpath(self.matlabDir)
    self.eng.parpool()

  def DetectGrasps(self, cloud, viewPoints, viewPointIndices, nSamples, scoreThresh, gpuId):
    '''Calls the DetectGrasps Matlab script.'''

    viewPointIndices = viewPointIndices + 1 # convert to Matlab 1-indexing

    mCloud = matlab.double(cloud.T.tolist())
    mViewPoints = matlab.double(viewPoints.T.tolist())
    mViewPointIndices = matlab.int32(viewPointIndices.tolist(), size=(len(viewPointIndices), 1))
    plotBitmap = matlab.logical([False, False, False])

    mGrasps = self.eng.DetectGrasps(
      mCloud, mViewPoints, mViewPointIndices, nSamples, scoreThresh, plotBitmap, gpuId)

    return self.UnpackGrasps(mGrasps)

  def UnpackGrasps(self, mGrasps):
    '''Extracts the list of grasps in Matlab format and returns a list in Python format.'''

    grasps = []
    for mGrasp in mGrasps:

      top = array(mGrasp["top"]).flatten()
      bottom = array(mGrasp["bottom"]).flatten()
      axis = array(mGrasp["axis"]).flatten()
      approach = array(mGrasp["approach"]).flatten()
      score = mGrasp["score"]

      # create grasp object
      T = hand_descriptor.PoseFromApproachAxisCenter(approach, axis, 0.5*bottom + 0.5*top)
      grasp = HandDescriptor(T)
      grasp.score = score
      grasps.append(grasp)

    return grasps
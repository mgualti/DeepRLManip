'''RL agent which uses the deictic state/action representation.'''

# python
from copy import copy
from time import time
# scipy
from numpy.linalg import norm
from numpy.random import choice, rand, randint
from numpy import any, argmax, argmin, array, exp, hstack, logical_and, isinf, repeat, reshape, \
  where, zeros
# self
import point_cloud
import hand_descriptor
from hand_descriptor import HandDescriptor
from rl_agent_hierarchical import RlAgentHierarchical

# AGENT ============================================================================================

class RlAgentLevel0(RlAgentHierarchical):

  def __init__(self, rlEnvironment, gpuId, hasHistory, nSamples, caffeDirPostfix=""):
    '''Initializes agent in the given environment.'''

    RlAgentHierarchical.__init__(self, rlEnvironment, gpuId, hasHistory, nSamples)

    # parameters
    self.level = 0

    # other internal variables
    self.SetInitialDescriptor(zeros(3))
    self.workspaceFilter = None

    # initialize caffe
    self.InitializeCaffe(caffeDirPostfix)

  def ComposeAction(self, prevDesc, baseCoord, imageCoord):
    '''Creates a new action and hand descriptor objects.'''

    action = [prevDesc.image, copy(imageCoord)]

    T = copy(prevDesc.T)
    T[0:3, 3] = baseCoord
    desc = HandDescriptor(T)

    return action, desc

  def SampleActions(self, prevDesc, cloudTree):
    '''Samples hand positions in both base frame and image coordinates.'''

    hX = prevDesc.handPoints
    if self.workspaceFilter is not None:
      mask = logical_and(hX[:, 2] >= self.workspaceFilter[0], hX[:, 2] <= self.workspaceFilter[1])
      if any(mask): hX = hX[mask, :]
    idx = randint(0, high=hX.shape[0], size=self.nSamples)
    hX = hX[idx, :] # sampling only points in the cloud

    bX = point_cloud.Transform(prevDesc.T, hX)

    dMax = max(prevDesc.imD, prevDesc.imW, prevDesc.imH)
    coordsXY = (hX[:, (0,1)] + (dMax/2.0)) * ((self.initDesc.imP-1) / dMax)
    coordsXZ = (hX[:, (0,2)] + (dMax/2.0)) * ((self.initDesc.imP-1) / dMax)
    coordsYZ = (hX[:, (1,2)] + (dMax/2.0)) * ((self.initDesc.imP-1) / dMax)
    iX = hstack((coordsXY, coordsXZ, coordsYZ))

    return bX, iX

  def SenseAndAct(self, hand, prevDesc, cloudTree, epsilon):
    '''Senses the current state, s, and determines the next action, a.'''

    prevDesc = self.initDesc

    # generate image for base frame descriptor
    prevDesc.GenerateDepthImage(cloudTree.data, cloudTree)
    s = self.emptyState if hand is None else [hand.image, self.emptyStateVector]

    # decide which location in the image to zoom into
    bX, iX = self.SampleActions(prevDesc, cloudTree)
    bestIdx, bestValue = self.SelectIndexEpsilonGreedy(s, prevDesc.image, iX, epsilon)

    # compose action
    a, desc = self.ComposeAction(prevDesc, bX[bestIdx], iX[bestIdx])

    return s, a, desc, bestValue

  def SetInitialDescriptor(self, center):
    '''Sets the center of the initial descriptor to the provided center.'''

    approach = array([0,0,-1]); axis = array([1,0,0])
    bTh = hand_descriptor.PoseFromApproachAxisCenter(approach, axis, center)
    self.initDesc = HandDescriptor(bTh)
    self.initDesc.imD = 4*self.initDesc.imD
    self.initDesc.imH = 4*self.initDesc.imH
    self.initDesc.imW = 4*self.initDesc.imW

  def SetSamplingWorkspace(self, approachMin, approachMax):
    '''Sets a filter in the base frame for which points to sample. All points are still included
       in the image.'''

    self.workspaceFilter = (approachMin, approachMax)
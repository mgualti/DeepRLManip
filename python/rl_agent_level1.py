'''RL agent which uses the deictic state/action representation.'''

# python
from copy import copy
# scipy
from numpy.linalg import norm
from numpy.random import choice, rand, randint, uniform
from numpy import absolute, argmax, argmin, array, exp, hstack, isinf, repeat, reshape, stack, \
  vstack, where, zeros
# self
import point_cloud
import hand_descriptor
from hand_descriptor import HandDescriptor
from rl_agent_hierarchical import RlAgentHierarchical

# AGENT ============================================================================================

class RlAgentLevel1(RlAgentHierarchical):

  def __init__(self, rlEnvironment, gpuId, hasHistory, nSamples, caffeDirPostfix=""):
    '''Initializes agent in the given environment.'''

    RlAgentHierarchical.__init__(self, rlEnvironment, gpuId, hasHistory, nSamples)

    # parameters
    self.level = 1

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

    # constrain to sample in center of image only
    hX = uniform(low=-1.0, high=1.0, size=(self.nSamples, 3))
    hX[:, 0] = hX[:, 0]*(prevDesc.imH/4.0)
    hX[:, 1] = hX[:, 1]*(prevDesc.imW/4.0)
    hX[:, 2] = hX[:, 2]*(prevDesc.imD/4.0)

    bX = point_cloud.Transform(prevDesc.T, hX)

    dMax = max(prevDesc.imD, prevDesc.imW, prevDesc.imH)
    coordsXY = (hX[:, (0,1)] + (dMax/2.0)) * ((prevDesc.imP-1) / dMax)
    coordsXZ = (hX[:, (0,2)] + (dMax/2.0)) * ((prevDesc.imP-1) / dMax)
    coordsYZ = (hX[:, (1,2)] + (dMax/2.0)) * ((prevDesc.imP-1) / dMax)
    iX = hstack((coordsXY, coordsXZ, coordsYZ))

    return bX, iX

  def SenseAndAct(self, hand, prevDesc, cloudTree, epsilon):
    '''Senses the current state, s, and determines the next action, a.'''

    # generate image for base frame descriptor
    prevDesc.GenerateDepthImage(cloudTree.data, cloudTree)
    s = self.emptyState if hand is None else [hand.image, self.emptyStateVector]

    # decide which location in the image to zoom into
    bX, iX = self.SampleActions(prevDesc, cloudTree)
    bestIdx, bestValue = self.SelectIndexEpsilonGreedy(s, prevDesc.image, iX, epsilon)

    # compose action
    a, desc = self.ComposeAction(prevDesc, bX[bestIdx], iX[bestIdx])

    return s, a, desc, bestValue
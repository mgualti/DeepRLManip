'''RL agent which uses the deictic state/action representation.'''

# python
from copy import copy
# scipy
from numpy.linalg import norm
from numpy.random import choice, rand, randint, uniform
from numpy import absolute, array, argmax, argmin, exp, hstack, isinf, linspace, repeat, reshape, \
  vstack, where, zeros
# self
import point_cloud
import hand_descriptor
from hand_descriptor import HandDescriptor
from rl_agent_hierarchical import RlAgentHierarchical

# AGENT ============================================================================================

class RlAgentLevel5(RlAgentHierarchical):

  def __init__(self, rlEnvironment, gpuId, hasHistory, nSamples, caffeDirPostfix=""):
    '''Initializes agent in the given environment.'''

    RlAgentHierarchical.__init__(self, rlEnvironment, gpuId, hasHistory, nSamples)

    # parameters
    self.level = 5

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

    # constrain to sample along axes only

    idx1 = int(self.nSamples/3)
    idx2 = int(2*idx1)
    idx3 = int(self.nSamples)

    hX = zeros((self.nSamples, 3))
    hX[0   :idx1, 0] = linspace(-prevDesc.imH, prevDesc.imH, idx1)
    hX[idx1:idx2, 1] = linspace(-prevDesc.imW, prevDesc.imW, idx1)
    hX[idx2:idx3, 2] = linspace(-prevDesc.imD, prevDesc.imD, self.nSamples-idx2)

    bX = point_cloud.Transform(prevDesc.T, hX)

    dMax = 2.0 * max(prevDesc.imD, prevDesc.imW, prevDesc.imH)
    coordsXY = (hX[:, (0,1)] + (dMax/2.0)) * ((prevDesc.imP-1) / dMax)
    coordsXZ = (hX[:, (0,2)] + (dMax/2.0)) * ((prevDesc.imP-1) / dMax)
    coordsYZ = (hX[:, (1,2)] + (dMax/2.0)) * ((prevDesc.imP-1) / dMax)
    iX = hstack((coordsXY, coordsXZ, coordsYZ))

    return bX, iX

  def SenseAndAct(self, hand, prevDesc, cloudTree, epsilon):
    '''Senses the current state, s, and determines the next action, a.'''

    # generate image for base frame descriptor
    newDesc = HandDescriptor(copy(prevDesc.T))
    newDesc.imH = 0.105; newDesc.imW = 0.105; newDesc.imD = 0.105
    newDesc.GenerateDepthImage(cloudTree.data, cloudTree)
    s = self.emptyState if hand is None else [hand.image, self.emptyStateVector]

    # decide which location in the image to zoom into
    bX, iX = self.SampleActions(newDesc, cloudTree)
    bestIdx, bestValue = self.SelectIndexEpsilonGreedy(s, newDesc.image, iX, epsilon)

    # compose action
    a, desc = self.ComposeAction(newDesc, bX[bestIdx], iX[bestIdx])

    return s, a, desc, bestValue
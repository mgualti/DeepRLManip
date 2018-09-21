'''RL agent which uses the deictic state/action representation.'''

# python
from copy import copy
# scipy
from numpy.linalg import inv
from numpy.random import choice, rand, randint, uniform
from numpy import absolute, argmax, argmin, array, concatenate, dot, exp, hstack, isinf, linspace, \
  ones, pi, repeat, stack, tile, vstack, where, zeros
# openrave
import openravepy
# ros
import tf
# self
import hand_descriptor
from hand_descriptor import HandDescriptor
from rl_agent_hierarchical import RlAgentHierarchical

# AGENT ============================================================================================

class RlAgentLevel3(RlAgentHierarchical):

  def __init__(self, rlEnvironment, gpuId, hasHistory, nSamples, caffeDirPostfix=""):
    '''Initializes agent in the given environment.'''

    RlAgentHierarchical.__init__(self, rlEnvironment, gpuId, hasHistory, nSamples)

    # parameters
    self.level = 3

    # initialize caffe
    self.InitializeCaffe(caffeDirPostfix)

  def ComposeAction(self, prevDesc, theta):
    '''Creates a new action and hand descriptor objects.'''

    action = [prevDesc.image, copy(theta)]

    R = openravepy.matrixFromAxisAngle(prevDesc.binormal, theta[1])[0:3,0:3]

    approach = dot(R, prevDesc.approach)
    axis = dot(R, prevDesc.axis)
    center = prevDesc.center

    T = hand_descriptor.PoseFromApproachAxisCenter(approach, axis, center)
    desc = HandDescriptor(T)

    return action, desc

  def SampleActions(self, prevDesc, cloudTree):
    '''Samples hand positions in both base frame and image coordinates.'''

    theta = zeros((self.nSamples, 6)) # last 3 columns are padded
    theta[:, 1] = linspace(-pi/2, pi/2, self.nSamples)

    return theta

  def SenseAndAct(self, hand, prevDesc, cloudTree, epsilon):
    '''Senses the current state, s, and determines the next action, a.'''

    # generate image for base frame descriptor
    prevDesc.GenerateDepthImage(cloudTree.data, cloudTree)
    s = self.emptyState if hand is None else [hand.image, self.emptyStateVector]

    # decide which location in the image to zoom into
    theta = self.SampleActions(prevDesc, cloudTree)
    bestIdx, bestValue = self.SelectIndexEpsilonGreedy(s, prevDesc.image, theta, epsilon)

    # compose action
    a, desc = self.ComposeAction(prevDesc, theta[bestIdx])

    return s, a, desc, bestValue
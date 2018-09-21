'''Reinforcement learning (RL) environment.'''

# python
from copy import copy
from time import sleep, time
# scipy
from scipy.spatial import cKDTree
from numpy.linalg import inv, norm
from numpy.random import rand, randint, randn
from numpy import array, arange, cos, dot, eye, hstack, pi, zeros
# openrave
import openravepy
# self
import point_cloud
import hand_descriptor
from hand_descriptor import HandDescriptor
from rl_environment import RlEnvironment

class RlEnvironmentGrasping(RlEnvironment):

  def __init__(self, showViewer=True, removeTable=False):
    '''Initializes openrave environment, etc.'''

    RlEnvironment.__init__(self, showViewer, removeTable)

  def IsAntipodalGrasp(self, descriptor, targetObject, maxAngleToFinger):
    '''Returns True if a grasp is near antipodal, based on the parameters.
    - Input descriptor: HandDescriptor object with pose of grasp.
    - Input targetObject: OpenRAVE object handle with cloud and normals entries.
    - Input maxAngleToFinger: Maximum angle between surfance normal and finger in degrees. Used
      10 degrees for blocks, 15 degrees for mugs, and 15 degrees for bottles.
    - Returns: True if antipodal grasp, False otherwise.
    '''

    # parameters
    contactWidth = 0.01
    maxAntipodalDist = 0.01
    maxAngleToFinger = cos(maxAngleToFinger*(pi/180))

    # put cloud into hand reference frame
    bTo = targetObject.GetTransform()
    bTh = descriptor.T
    hTo = dot(inv(bTh), bTo)
    X, N = point_cloud.Transform(hTo, targetObject.cloud, targetObject.normals)
    X, N = point_cloud.FilterWorkspace([(-descriptor.height/2, descriptor.height/2),
                                        (-descriptor.width/2, descriptor.width/2),
                                        (-descriptor.depth/2,descriptor.depth/2)], X, N)
    if X.shape[0] == 0:
      #print("No points in hand.")
      return False

    # find contact points
    leftPoint = min(X[:, 1]); rightPoint = max(X[:, 1])
    lX, lN = point_cloud.FilterWorkspace([(-1,1),(leftPoint,leftPoint+contactWidth),(-1,1)], X, N)
    rX, rN = point_cloud.FilterWorkspace([(-1,1),(rightPoint-contactWidth,rightPoint),(-1,1)], X, N)

    # find contact points normal to finger
    lX = lX[-lN[:, 1] >= maxAngleToFinger, :]
    rX = rX[ rN[:, 1] >= maxAngleToFinger, :]
    if lX.shape[0] == 0 or rX.shape[0] == 0:
      #print("No contact points normal to finger.")
      return False

    # are the closest two contact points nearly antipodal?
    leftTree = cKDTree(lX[:,(0, 2)])
    d, idxs = leftTree.query(rX[:, (0,2)])
    #if min(d) >= maxAntipodalDist:
    #  print("Contacts not antipodal.")
    return min(d) < maxAntipodalDist
  
  def TestGrasp(self, descriptor, rlAgent, objectHandles):
    '''Perform the action the robot selected and step the simulation forward one timestep.
    - Input descriptor: The descriptor the robot selected.
    - Input rlAgent: RLAgent object representing the robot.
    - Input objectHandles: Handles to all of the blocks currently in the scene.
    - Returns r: The grasping reward.
    '''

    # determine which object is closest to grasp
    minDist = float('inf'); targObj = None
    for obj in objectHandles:
      bTo = obj.GetTransform()
      dist = norm(bTo[0:3, 3] - descriptor.T[0:3, 3])
      if dist < minDist:
        targObj = obj
        minDist = dist

    # antipodal condition
    antipodal = self.IsAntipodalGrasp(descriptor, targObj, maxAngleToFinger=15)

    # collision condition
    rlAgent.MoveSensorToPose(descriptor.T)
    collisionFree = not self.env.CheckCollision(self.robot)

    # return result
    return float(antipodal), float(antipodal and collisionFree)
  
  def TransitionGraspHalfConditions(self, descriptor, rlAgent, objectHandles):
    '''Perform the action the robot selected and step the simulation forward one timestep.
    - Input descriptor: The descriptor the robot selected.
    - Input rlAgent: RLAgent object representing the robot.
    - Input objectHandles: Handles to all of the blocks currently in the scene.
    - Returns r: The grasping reward.
    '''

    # determine which object is closest to grasp
    minDist = float('inf'); targObj = None
    for obj in objectHandles:
      bTo = obj.GetTransform()
      dist = norm(bTo[0:3, 3] - descriptor.T[0:3, 3])
      if dist < minDist:
        targObj = obj
        minDist = dist

    # antipodal conditions
    halfAntipodal = self.IsAntipodalGrasp(descriptor, targObj, maxAngleToFinger=30)
    if not halfAntipodal: return 0.0
    antipodal = self.IsAntipodalGrasp(descriptor, targObj, maxAngleToFinger=15)

    # collision conditions
    T = copy(descriptor.T)
    T[0:3, 3] = T[0:3, 3] - 0.01*descriptor.approach
    rlAgent.MoveSensorToPose(T)
    backupCollisionFree = not self.env.CheckCollision(self.robot)
    rlAgent.MoveSensorToPose(descriptor.T)
    collisionFree = not self.env.CheckCollision(self.robot)
    halfCollisionFree = collisionFree or backupCollisionFree
    if not halfCollisionFree: return 0.0

    # compute reward
    conditions = [antipodal, collisionFree]
    nConditionsMet = sum(conditions)
    return (nConditionsMet+1.0) / 3.0

  def TransitionObjectTopGraspHalfConditions(self, descriptor, rlAgent, objectHandles):
    '''Perform the action the robot selected and step the simulation forward one timestep.
    - Input descriptor: The descriptor the robot selected.
    - Input rlAgent: RLAgent object representing the robot.
    - Input objectHandles: Handles to all of the blocks currently in the scene.
    - Returns r: The grasping reward.
    '''

    # determine which object is closest to grasp
    minDist = float('inf'); targObj = None
    for obj in objectHandles:
      bTo = obj.GetTransform()
      dist = norm(bTo[0:3, 3] - descriptor.T[0:3, 3])
      if dist < minDist:
        targObj = obj
        minDist = dist

    # object top grasp conditions
    bTo = targObj.GetTransform()
    cosAngleDiff = dot(-descriptor.approach, bTo[0:3, 2])
    halfTopGrasp = cosAngleDiff >= cos(130*(pi/180))
    if not halfTopGrasp: return 0.0
    topGrasp = cosAngleDiff >= cos(110*(pi/180))

    # antipodal conditions
    halfAntipodal = self.IsAntipodalGrasp(descriptor, targObj, maxAngleToFinger=30)
    if not halfAntipodal: return 0.0
    antipodal = self.IsAntipodalGrasp(descriptor, targObj, maxAngleToFinger=15)

    # collision conditions
    T = copy(descriptor.T)
    T[0:3, 3] = T[0:3, 3] - 0.01*descriptor.approach
    rlAgent.MoveSensorToPose(T)
    backupCollisionFree = not self.env.CheckCollision(self.robot)
    rlAgent.MoveSensorToPose(descriptor.T)
    collisionFree = not self.env.CheckCollision(self.robot)
    halfCollisionFree = collisionFree or backupCollisionFree
    if not halfCollisionFree: return 0.0

    # compute reward
    conditions = [topGrasp, antipodal, collisionFree]
    nConditionsMet = sum(conditions)
    return (nConditionsMet+1.0) / 4.0

  def TransitionTopGraspHalfConditions(self, descriptor, rlAgent, objectHandles):
    '''Perform the action the robot selected and step the simulation forward one timestep.
    - Input descriptor: The descriptor the robot selected.
    - Input rlAgent: RLAgent object representing the robot.
    - Input objectHandles: Handles to all of the blocks currently in the scene.
    - Returns r: The grasping reward.
    '''

    # determine which object is closest to grasp
    minDist = float('inf'); targObj = None
    for obj in objectHandles:
      bTo = obj.GetTransform()
      dist = norm(bTo[0:3, 3] - descriptor.T[0:3, 3])
      if dist < minDist:
        targObj = obj
        minDist = dist

    # object top grasp conditions
    desiredApproach = array([0, 0, -1.0])
    cosAngleDiff = dot(descriptor.approach, desiredApproach)
    halfTopGrasp = cosAngleDiff >= cos(65*(pi/180))
    if not halfTopGrasp: return 0.0
    topGrasp = cosAngleDiff >= cos(45*(pi/180))

    # antipodal conditions
    halfAntipodal = self.IsAntipodalGrasp(descriptor, targObj, maxAngleToFinger=20)
    if not halfAntipodal: return 0.0
    antipodal = self.IsAntipodalGrasp(descriptor, targObj, maxAngleToFinger=10)

    # collision conditions
    T = copy(descriptor.T)
    T[0:3, 3] = T[0:3, 3] - 0.01*descriptor.approach
    rlAgent.MoveSensorToPose(T)
    backupCollisionFree = not self.env.CheckCollision(self.robot)
    rlAgent.MoveSensorToPose(descriptor.T)
    collisionFree = not self.env.CheckCollision(self.robot)
    halfCollisionFree = collisionFree or backupCollisionFree
    if not halfCollisionFree: return 0.0

    # compute reward
    conditions = [topGrasp, antipodal, collisionFree]
    nConditionsMet = sum(conditions)
    return (nConditionsMet+1.0) / 4.0
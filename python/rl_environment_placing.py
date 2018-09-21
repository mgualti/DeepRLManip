'''Reinforcement learning (RL) environment.'''

# python
from copy import copy
from time import sleep, time
# scipy
from scipy.spatial import cKDTree
from numpy.linalg import inv, norm
from numpy.random import rand, randn, uniform
from numpy import arccos, argmin, array, arange, cos, dot, eye, hstack, mean, pi, power, repeat, \
  reshape, sqrt, sum, vstack, zeros
# openrave
import openravepy
# self
import point_cloud
import hand_descriptor
from hand_descriptor import HandDescriptor
from rl_environment_grasping import RlEnvironmentGrasping

class RlEnvironmentPlacing(RlEnvironmentGrasping):

  def __init__(self, showViewer=True, removeTable=False):
    '''Initializes openrave environment, etc.'''

    RlEnvironmentGrasping.__init__(self, showViewer, removeTable)

    # properties
    self.oTg = None
    self.holdingObject = None
    self.holdingPose = None

  def ClosestPointToRay(self, points, rayPoint, rayDirection):
    '''Finds the closest point in points to the ray specified by its origin and direction.'''

    rayPoint = repeat(reshape(rayPoint, (1,3)), points.shape[0], axis=0)
    rayPointToPoints = points - rayPoint
    pointsToRayPoint = -rayPointToPoints
    rayDirection = reshape(rayDirection, (3, 1))
    t0 = dot(rayPointToPoints, rayDirection)
    mask = (t0 < 0).flatten()
    t0 = repeat(reshape(t0, (t0.shape[0], 1)), 3, axis=1)
    rayDirection = repeat(reshape(rayDirection, (1, 3)), points.shape[0], axis=0)
    dist = sum(power(pointsToRayPoint + t0*rayDirection, 2), axis=1)
    dist[mask] = sum(power(pointsToRayPoint[mask, :], 2), axis=1)
    minDistIdx = argmin(dist)
    return points[minDistIdx], sqrt(dist[minDistIdx])

  def GraspObject(self, rlAgent, descriptor, objectHandles):
    '''Uses rlAgent to grasp the object and move it to a known holding pose.'''

    # determine which object is closest to grasp
    minDist = float('inf'); targObj = None
    for obj in objectHandles:
      bTo = obj.GetTransform()
      dist = norm(bTo[0:3, 3] - descriptor.T[0:3, 3])
      if dist < minDist:
        targObj = obj
        minDist = dist

    # move the object and hand together to the hand holding pose
    self.oTg = dot(inv(targObj.GetTransform()), descriptor.T)
    rlAgent.MoveHandToHoldingPose()
    T = rlAgent.MoveObjectToHandAtGrasp(descriptor.T, targObj)
    targObj.GetLinks()[0].SetStatic(True)
    self.holdingObject = targObj
    self.holdingPose = dot(T, descriptor.T)

    return

  def GraspObjectAndCenter(self, rlAgent, descriptor, objectHandles):
    '''Uses rlAgent to grasp the object and move it to a known holding pose.'''

    # determine which object is closest to grasp
    minDist = float('inf'); targObj = None
    for obj in objectHandles:
      bTo = obj.GetTransform()
      dist = norm(bTo[0:3, 3] - descriptor.T[0:3, 3])
      if dist < minDist:
        targObj = obj
        minDist = dist

    # center object in the hand -- as if fingers closed
    gTo = dot(inv(descriptor.T), targObj.GetTransform())
    gTo[1, 3] = 0
    bTo = dot(descriptor.T, gTo)
    targObj.SetTransform(bTo)

    # move the object and hand together to the hand holding pose
    self.oTg = dot(inv(targObj.GetTransform()), descriptor.T)
    rlAgent.MoveHandToHoldingPose()
    T = rlAgent.MoveObjectToHandAtGrasp(descriptor.T, targObj)
    targObj.GetLinks()[0].SetStatic(True)
    self.holdingObject = targObj
    self.holdingPose = dot(T, descriptor.T)

    return

  def TransitionPlaceOnObject(self, descriptor, rlAgent, objectHandles, showSteps):
    '''Perform the action the robot selected and step the simulation forward one timestep.'''

    # move the object and hand together to the place pose
    rlAgent.MoveSensorToPose(descriptor.T)
    rlAgent.MoveObjectToHandAtGrasp(self.holdingPose, self.holdingObject)

    # orientation
    bTo = self.holdingObject.GetTransform()
    gravity = array([0,0,-1])
    maxDist = -float('inf')
    for i in xrange(3):
      dist = abs(dot(bTo[0:3, i], gravity))
      if dist > maxDist:
        maxDist = dist
    halfGoodOrient = maxDist >= cos(30*(pi/180))
    if not halfGoodOrient:
      if showSteps: raw_input("Orientation of block is {} degrees.".format(arccos(maxDist)*(180/pi)))
      return 0.00
    goodOrient = maxDist >= cos(15*(pi/180))

    # above block
    topPoints = zeros((0, 3))
    for obj in objectHandles:
      if obj == self.holdingObject: continue
      objCloud = point_cloud.Transform(obj.GetTransform(), obj.cloud)
      mask = objCloud[:, 2] > max(objCloud[:, 2])-0.005
      topPoints = vstack((topPoints, objCloud[mask, :]))
    center = self.holdingObject.GetTransform()[0:3, 3]
    closestTopPoint, closestTopPointDist = self.ClosestPointToRay(
      topPoints, center, array([0,0,-1]))
    if closestTopPointDist > 0.003:
      if showSteps: raw_input("Not above block.")
      return 0.00

    # height
    objCloud = point_cloud.Transform(self.holdingObject.GetTransform(), self.holdingObject.cloud)
    bottomPoint, bottomPointDist = self.ClosestPointToRay(objCloud, center, array([0,0,-1]))
    height = bottomPoint[2] - closestTopPoint[2]
    halfGoodHeight = (height <= 0.03)
    goodHeight = (height <= 0.015)
    if not halfGoodHeight:
      if showSteps: raw_input("Height is {} cm".format(height*100))
      return 0.0

    # collision
    collisionFreeHand = not self.env.CheckCollision(self.robot)
    bTgTemp = copy(descriptor.T)
    bTgTemp[0:3, 3] = bTgTemp[0:3, 3] - 0.01*descriptor.approach
    rlAgent.MoveSensorToPose(bTgTemp)
    halfCollisionFreeHand = not self.env.CheckCollision(self.robot)
    rlAgent.MoveHandToHoldingPose()
    collisionFreeObject = not self.env.CheckCollision(self.holdingObject)
    bTo = self.holdingObject.GetTransform()
    bToTemp = copy(bTo)
    bToTemp[0:3, 3] = bTo[0:3, 3] + 0.02*array([0,0,1])
    self.holdingObject.SetTransform(bToTemp)
    halfCollisionFreeObject = not self.env.CheckCollision(self.holdingObject)
    self.holdingObject.SetTransform(bTo)
    rlAgent.MoveSensorToPose(descriptor.T)
    collisionFree = collisionFreeHand and collisionFreeObject
    halfCollisionFree = collisionFree or \
      (collisionFreeObject and halfCollisionFreeHand) or \
      (halfCollisionFreeObject and collisionFreeHand) or \
      (halfCollisionFreeObject and halfCollisionFreeHand)
    #print "cf", collisionFree, "hcf", halfCollisionFree, "cfo", collisionFreeObject, "cfh", \
    #  collisionFreeHand, "hcfo", halfCollisionFreeObject, "hcfh", halfCollisionFreeHand
    if not halfCollisionFree:
      if showSteps: raw_input("Not half collision free.")
      return 0.00

    # compute reward
    conditions = [goodOrient, goodHeight, collisionFree]
    nConditionsMet = sum(conditions)
    return (nConditionsMet+1.0) / 4.0

  def TransitionPlaceUprightOnSurfaceObject(self, descriptor, rlAgent, objectHandles,
    surfaceObjectHandles, showSteps):
    '''Perform the action the robot selected and step the simulation forward one timestep.'''

    # move the object and hand together to the place pose
    rlAgent.MoveSensorToPose(descriptor.T)
    rlAgent.MoveObjectToHandAtGrasp(self.holdingPose, self.holdingObject)

    # orientation
    bTo = self.holdingObject.GetTransform()
    gravity = array([0,0,-1])
    dist = dot(bTo[0:3, 2], -gravity)
    halfGoodOrient = dist >= cos(30*(pi/180))
    if not halfGoodOrient:
      if showSteps: raw_input("Orientation of block is {} degrees.".format(arccos(dist)*(180/pi)))
      return 0.00
    goodOrient = dist >= cos(15*(pi/180))

    # above surface
    topPoints = zeros((0, 3))
    for obj in surfaceObjectHandles:
      objCloud = point_cloud.Transform(obj.GetTransform(), obj.cloud)
      mask = objCloud[:, 2] > max(objCloud[:, 2])-0.005
      topPoints = vstack((topPoints, objCloud[mask, :]))
    center = self.holdingObject.GetTransform()[0:3, 3]
    closestTopPoint, closestTopPointDist = self.ClosestPointToRay(
      topPoints, center, array([0,0,-1]))
    if closestTopPointDist > 0.003:
      if showSteps: raw_input("Not above surface.")
      return 0.00

    # height
    objCloud = point_cloud.Transform(self.holdingObject.GetTransform(), self.holdingObject.cloud)
    bottomPoint, bottomPointDist = self.ClosestPointToRay(objCloud, center, gravity)
    height = bottomPoint[2] - closestTopPoint[2]
    halfGoodHeight = (height <= 0.04)
    goodHeight = (height <= 0.02)
    if not halfGoodHeight:
      if showSteps: raw_input("Height is {} cm".format(height*100))
      return 0.0

    # collision
    collisionFreeHand = not self.env.CheckCollision(self.robot)
    bTgTemp = copy(descriptor.T)
    bTgTemp[0:3, 3] = bTgTemp[0:3, 3] - 0.01*descriptor.approach
    rlAgent.MoveSensorToPose(bTgTemp)
    halfCollisionFreeHand = not self.env.CheckCollision(self.robot)
    rlAgent.MoveHandToHoldingPose()
    collisionFreeObject = not self.env.CheckCollision(self.holdingObject)
    bTo = self.holdingObject.GetTransform()
    bToTemp = copy(bTo)
    bToTemp[0:3, 3] = bTo[0:3, 3] - 0.02*gravity
    self.holdingObject.SetTransform(bToTemp)
    halfCollisionFreeObject = not self.env.CheckCollision(self.holdingObject)
    self.holdingObject.SetTransform(bTo)
    rlAgent.MoveSensorToPose(descriptor.T)
    collisionFree = collisionFreeHand and collisionFreeObject
    halfCollisionFree = collisionFree or \
      (collisionFreeObject and halfCollisionFreeHand) or \
      (halfCollisionFreeObject and collisionFreeHand) or \
      (halfCollisionFreeObject and halfCollisionFreeHand)
    #print "cf", collisionFree, "hcf", halfCollisionFree, "cfo", collisionFreeObject, "cfh", \
    #  collisionFreeHand, "hcfo", halfCollisionFreeObject, "hcfh", halfCollisionFreeHand
    if not halfCollisionFree:
      if showSteps: raw_input("Not half collision free.")
      return 0.00

    # compute reward
    conditions = [goodOrient, goodHeight, collisionFree]
    nConditionsMet = sum(conditions)
    return (nConditionsMet+1.0) / 4.0

  def TransitionPlaceUprightOnTable(self, descriptor, rlAgent, showSteps):
    '''Perform the action the robot selected and step the simulation forward one timestep.'''

    # move the object and hand together to the place pose
    rlAgent.MoveSensorToPose(descriptor.T)
    rlAgent.MoveObjectToHandAtGrasp(self.holdingPose, self.holdingObject)

    # upright
    objectPose = self.holdingObject.GetTransform()
    uprightDistance = dot(array([0,0,1]), objectPose[0:3, 2])
    halfUpright = uprightDistance >= cos(40*(pi/180))
    upright = uprightDistance >= cos(20*(pi/180))
    if not halfUpright: return 0.00

    # height
    objectCloud = point_cloud.Transform(objectPose, self.holdingObject.cloud)
    objectBottomHeight = min(objectCloud[:, 2])
    halfGoodHeight = objectBottomHeight >= -0.01 and objectBottomHeight <= 0.04
    goodHeight = objectBottomHeight >= 0.00 and objectBottomHeight <= 0.03
    if not halfGoodHeight: return 0.00

    # collision
    collisionFreeHand = not self.env.CheckCollision(self.robot)
    bTgTemp = copy(descriptor.T)
    bTgTemp[0:3, 3] = bTgTemp[0:3, 3] - 0.01*descriptor.approach
    rlAgent.MoveSensorToPose(bTgTemp)
    halfCollisionFreeHand = not self.env.CheckCollision(self.robot)
    rlAgent.MoveHandToHoldingPose()
    collisionFreeObject = not self.env.CheckCollision(self.holdingObject)
    bTo = self.holdingObject.GetTransform()
    bToTemp = copy(bTo)
    bToTemp[0:3, 3] = bTo[0:3, 3] + 0.01*bTo[0:3, 2]
    self.holdingObject.SetTransform(bToTemp)
    halfCollisionFreeObject = not self.env.CheckCollision(self.holdingObject)
    self.holdingObject.SetTransform(bTo)
    rlAgent.MoveSensorToPose(descriptor.T)
    collisionFree = collisionFreeHand and collisionFreeObject
    halfCollisionFree = collisionFree or \
      (collisionFreeObject and halfCollisionFreeHand) or \
      (halfCollisionFreeObject and collisionFreeHand) or \
      (halfCollisionFreeObject and halfCollisionFreeHand)
    #print "cf", collisionFree, "hcf", halfCollisionFree, "cfo", collisionFreeObject, "cfh", \
    #  collisionFreeHand, "hcfo", halfCollisionFreeObject, "hcfh", halfCollisionFreeHand
    if not halfCollisionFree: return 0.00

    # compute reward
    if goodHeight and upright and collisionFree:
      return 1.00

    if (halfGoodHeight and upright and collisionFree) or \
       (goodHeight and halfUpright and collisionFree) or \
       (goodHeight and upright and halfCollisionFree):
         return 0.75

    if (goodHeight and halfUpright and halfCollisionFree) or \
       (halfGoodHeight and upright and halfCollisionFree) or \
       (halfGoodHeight and halfUpright and collisionFree):
         return 0.50

    return 0.25
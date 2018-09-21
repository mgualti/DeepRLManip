'''Reinforcement learning (RL) agent and related utility functions.'''

# python
from time import time
# scipy
from scipy.spatial import cKDTree
from numpy.linalg import inv, norm
from numpy import array, cos, cross, dot, eye, pi, sin, tile, vstack, zeros
# openrave
import openravepy
# caffe
import caffe
# self
import point_cloud

# AGENT ============================================================================================

class RlAgent:

  def __init__(self, rlEnvironment, gpuId):
    '''Initializes agent in the given environment.'''

    # simple assignments
    self.rlEnv = rlEnvironment
    self.env = self.rlEnv.env
    self.robot = self.rlEnv.robot

    # get sensor from openrave
    self.sensor = self.env.GetSensors()[0]
    self.sTh = dot(inv(self.sensor.GetTransform()), self.robot.GetTransform())

    # initialize caffe
    if gpuId >= 0:
      caffe.set_mode_gpu()
      caffe.set_device(gpuId)
    else:
      caffe.set_mode_cpu()
    self.caffeFirstTrain = True

    # other internal variables
    self.plotCloudHandle = None
    self.plotDescriptorsHandle = None

  def GetCloud(self, workspace=None):
    '''Agent gets point cloud from its sensor from the current position.'''

    self.StartSensor()
    self.env.StepSimulation(0.001)

    data = self.sensor.GetSensorData(openravepy.Sensor.Type.Laser)
    cloud = data.ranges + data.positions[0]

    self.StopSensor()

    if workspace is not None:
      cloud = point_cloud.FilterWorkspace(workspace, cloud)

    return cloud

  def GetDualCloud(self, viewCenter, viewKeepout, workspace):
    '''Gets a point cloud combined form two views, 45 degrees above table and 90 degrees apart.
      - Returns cloud: nx3 combined point cloud.
      - Returns cloudTree: cKDTree object of cloud.
    '''

    poses = self.GetDualViewPoses(viewCenter, viewKeepout)

    cloud = []
    for pose in poses:
      self.MoveSensorToPose(pose)
      cloud.append(self.GetCloud(workspace))

    cloud = vstack(cloud)
    cloud = point_cloud.Voxelize(cloud, voxelSize=0.002)

    return cloud, cKDTree(cloud)

  def GetDualCloudAndNormals(self, viewCenter, viewKeepout, workspace):
    '''Same as above except surface normals are also computed and returned.
    - Returns cloud: nx3 combined point cloud.
    - Returns cloudTree: cKDTree object of cloud.
    - Returns normals: nx3 surface normals for cloud.
    '''

    poses = self.GetDualViewPoses(viewCenter, viewKeepout)

    cloud = []; viewPoints = []
    for pose in poses:
      self.MoveSensorToPose(pose)
      X = self.GetCloud(workspace)
      V = tile(pose[0:3,3], (X.shape[0], 1))
      cloud.append(X)
      viewPoints.append(V)

    cloud = vstack(cloud)
    viewPoints = vstack(viewPoints)
    normals = point_cloud.ComputeNormals(cloud, viewPoints, kNeighbors=30, rNeighbors=-1)
    #point_cloud.SaveMat("test.mat",cloud,normals)

    return cloud, cKDTree(cloud), normals

  def GetDualCloudAndViewPoints(self, viewCenter, viewKeepout, workspace):
    '''Same as above except returns view points and view point indices.
    - Returns cloud: nx3 combined point cloud.
    - Returns cloudTree: cKDTree object of cloud.
    - Returns viewPoints: mx3 list of view points from which the cloud was taken.
    - Returns viewPointIndices: Indices into the point cloud where each of the view points starts.
    '''

    poses = self.GetDualViewPoses(viewCenter, viewKeepout)

    cloud = []; viewPoints = []; viewPointIndices = zeros(len(poses), dtype='int')
    for i, pose in enumerate(poses):
      self.MoveSensorToPose(pose)
      cloud.append(self.GetCloud(workspace))
      viewPoints.append(pose[0:3,3])
      if i > 0: viewPointIndices[i] = viewPointIndices[i-1] + cloud[i-1].shape[0]

    cloud = vstack(cloud)
    viewPoints = vstack(viewPoints)

    return cloud, cKDTree(cloud), viewPoints, viewPointIndices

  def GetDualViewPoses(self, viewCenter, viewKeepout):
    '''Gets standard dual poses, 45 degress from table and 90 degrees apart.'''

    p1 = viewCenter + viewKeepout*array([0, -cos(45*(pi/180)), sin(45*(pi/180))])
    p2 = viewCenter + viewKeepout*array([0,  cos(45*(pi/180)), sin(45*(pi/180))])

    upChoice = array([1,0,0])
    viewPose1 = GeneratePoseGivenUp(p1, viewCenter, upChoice)
    viewPose2 = GeneratePoseGivenUp(p2, viewCenter, upChoice)

    return viewPose1, viewPose2

  def GetFullCloudAndNormals(self, viewCenter, viewKeepout, workspace, add45DegViews=False):
    '''Gets a full point cloud of the scene (6 views) and also computes normals.'''

    poses = self.GetFullViewPoses(viewCenter, viewKeepout, add45DegViews)

    cloud = []; viewPoints = []
    for pose in poses:
      self.MoveSensorToPose(pose)
      X = self.GetCloud(workspace)
      V = tile(pose[0:3,3], (X.shape[0], 1))
      cloud.append(X)
      viewPoints.append(V)

    cloud = vstack(cloud)
    viewPoints = vstack(viewPoints)
    normals = point_cloud.ComputeNormals(cloud, viewPoints, kNeighbors=30, rNeighbors=-1)

    return cloud, normals

  def GetFullViewPoses(self, viewCenter, viewKeepout, add45DegViews):
    '''Returns 6 poses, covering the full object. (No face has incidence more than 90 degrees.)'''

    viewPoints = []
    viewPoints.append(viewCenter + viewKeepout*array([ 0,  0,  1]))
    viewPoints.append(viewCenter + viewKeepout*array([ 0,  0, -1]))
    viewPoints.append(viewCenter + viewKeepout*array([ 0,  1,  0]))
    viewPoints.append(viewCenter + viewKeepout*array([ 0, -1,  0]))
    viewPoints.append(viewCenter + viewKeepout*array([ 1,  0,  0]))
    viewPoints.append(viewCenter + viewKeepout*array([-1,  0,  0]))

    if add45DegViews:
      viewPoints.append(viewCenter + viewKeepout*array([0, -cos(45*(pi/180)), sin(45*(pi/180))]))
      viewPoints.append(viewCenter + viewKeepout*array([0,  cos(45*(pi/180)), sin(45*(pi/180))]))
      viewPoints.append(viewCenter + viewKeepout*array([-cos(45*(pi/180)), 0, sin(45*(pi/180))]))
      viewPoints.append(viewCenter + viewKeepout*array([cos(45*(pi/180)), 0, sin(45*(pi/180))]))

    upChoice = array([0.9,0.1,0])
    upChoice = upChoice / norm(upChoice)

    viewPoses = []
    for point in viewPoints:
      viewPoses.append(GeneratePoseGivenUp(point, viewCenter, upChoice))

    return viewPoses

  def MoveHandToHoldingPose(self):
    '''Moves the hand to a special, pre-designated holding area.'''

    T = eye(4)
    T[0:3, 0] = array([-1, 0,  0])
    T[0:3, 1] = array([ 0, 1,  0])
    T[0:3, 2] = array([ 0, 0, -1])
    T[0:3, 3] = array([-1, 0,  0.30])

    self.MoveHandToPose(T)

  def MoveHandToPose(self, T):
    '''Moves the hand of the robot to the specified pose.'''

    self.robot.SetTransform(T)
    self.env.UpdatePublishedBodies()

  def MoveObjectToHandAtGrasp(self, bTg, objectHandle):
    '''Aligns the grasp on the object to the current hand position and moves the object there.
      - Input: The grasp in the base frame (4x4 homogeneous transform).
      - Input objectHandle: Handle to the object to move.
      - Retruns X: The transform applied to the object.
    '''

    bTo = objectHandle.GetTransform()
    bTs = self.sensor.GetTransform()

    X = dot(bTs, inv(bTg))

    with self.env:
      objectHandle.SetTransform(dot(X, bTo))

    return X

  def MoveSensorToPose(self, T):
    '''Moves the hand of the robot to the specified pose.'''

    with self.env:
      self.robot.SetTransform(dot(T, self.sTh))
      self.env.UpdatePublishedBodies()

  def PlotCloud(self, cloud):
    '''Plots a cloud in the environment.'''

    if not self.rlEnv.showViewer:
      return

    if self.plotCloudHandle is not None:
      self.UnplotCloud()

    self.plotCloudHandle = self.env.plot3(\
      points=cloud, pointsize=0.001, colors=zeros(cloud.shape), drawstyle=1)

  def PlotDescriptors(self, descriptors, graspColorRgb=[1,0,0]):
    '''Visualizes grasps in openrave viewer.'''

    if not self.rlEnv.showViewer:
      return

    if self.plotDescriptorsHandle is not None:
      self.UnplotDescriptors()

    if len(descriptors) == 0:
      return

    lineList = []; colorList = []
    for desc in descriptors:

      c = desc.bottom
      a = c - desc.depth*desc.approach
      l = c - 0.5*desc.width*desc.binormal
      r = c + 0.5*desc.width*desc.binormal
      lEnd = l + desc.depth*desc.approach
      rEnd = r + desc.depth*desc.approach

      lineList.append(c); lineList.append(a)
      lineList.append(l); lineList.append(r)
      lineList.append(l); lineList.append(lEnd)
      lineList.append(r); lineList.append(rEnd)

      for i in xrange(8): colorList.append(graspColorRgb)

    self.plotDescriptorsHandle = self.env.drawlinelist(\
      points=array(lineList), linewidth=3.0, colors=array(colorList))

  def StartSensor(self):
    '''Starts the sensor in openrave, displaying yellow haze.'''

    self.sensor.Configure(openravepy.Sensor.ConfigureCommand.PowerOn)
    self.sensor.Configure(openravepy.Sensor.ConfigureCommand.RenderDataOn)

  def StopSensor(self):
    '''Disables the sensor in openrave, removing the yellow haze.'''

    self.sensor.Configure(openravepy.Sensor.ConfigureCommand.PowerOff)
    self.sensor.Configure(openravepy.Sensor.ConfigureCommand.RenderDataOff)

  def UnplotCloud(self):
    '''Removes a cloud from the environment.'''

    if not self.rlEnv.showViewer:
      return

    if self.plotCloudHandle is not None:
      self.plotCloudHandle.Close()
      self.plotCloudHandle = None

  def UnplotDescriptors(self):
    '''Removes any descriptors drawn in the environment.'''

    if not self.rlEnv.showViewer:
      return

    if self.plotDescriptorsHandle is not None:
      self.plotDescriptorsHandle.Close()
      self.plotDescriptorsHandle = None

# UTILITIES ========================================================================================

def GeneratePoseGivenUp(sensorPosition, targetPosition, upAxis):
  '''Generates the sensor pose with the LOS pointing to a target position and the "up" close to a specified up.

  - Input sensorPosition: 3-element desired position of sensor placement.
  - Input targetPosition: 3-element position of object required to view.
  - Input upAxis: The direction the sensor up should be close to.
  - Returns T: 4x4 numpy array (transformation matrix) representing desired pose of end effector in the base frame.
  '''

  v = targetPosition - sensorPosition
  v = v / norm(v)

  u = upAxis - dot(upAxis, v) * v
  u = u / norm(u)

  t = cross(u, v)

  T = eye(4)
  T[0:3,0] = t
  T[0:3,1] = u
  T[0:3,2] = v
  T[0:3,3] = sensorPosition

  return T
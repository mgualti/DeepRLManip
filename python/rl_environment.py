'''Reinforcement learning (RL) environment.'''

# python
import os
import fnmatch
from time import sleep, time
# scipy
from scipy.io import loadmat
from numpy.random import choice, rand, randint, randn, uniform
from numpy import array, dot, eye, hstack, ones, pi, zeros
# openrave
import openravepy

class RlEnvironment:

  def __init__(self, showViewer=True, removeTable=False):
    '''Initializes openrave environment, etc.'''

    # Parameters
    self.projectDir = os.getcwd() + "/"

    # Create openrave environment
    self.env = openravepy.Environment()
    if showViewer: self.env.SetViewer('qtcoin')
    self.showViewer = showViewer
    self.env.Load(self.projectDir + "openrave/environment_2.xml")
    self.robot = self.env.GetRobots()[0]
    self.robot.SetDOFValues(array([0.0475]))

    # set collision checker options
    collisionChecker = openravepy.RaveCreateCollisionChecker(self.env, 'ode')
    self.env.SetCollisionChecker(collisionChecker)

    # don't want to be affected by gravity, since it is floating
    for link in self.robot.GetLinks():
      link.SetStatic(True)

    # set physics options
    #self.physicsEngine = openravepy.RaveCreatePhysicsEngine(self.env, "ode")
    #self.env.SetPhysicsEngine(self.physicsEngine)
    #self.env.GetPhysicsEngine().SetGravity([0,0,-9.8])
    self.env.StopSimulation()

    # table(s)
    tableObj = self.env.GetKinBody("table")
    netObj = self.env.GetKinBody("net")
    self.tablePosition = tableObj.GetTransform()[0:3,3]
    self.tableExtents = tableObj.ComputeAABB().extents()

    if removeTable:
      self.env.Remove(tableObj)
      self.env.Remove(netObj)

    # Internal state
    self.objectPoses = []

  def GetObjectPose(self, objectHandle):
    '''Returns the pose of an object.'''

    with self.env:
      return objectHandle.GetTransform()

  def MoveObjectToPose(self, objectHandle, T):
    '''Moves the object to the pose specified by the 4x4 matrix T.'''

    with self.env:
      objectHandle.SetTransform(T)

  def PlaceBlockAtOrigin(self, scaleMinMax, blockName, saveBlock=True):
    '''Places a single, random block at the origin of the world frame.'''

    # block creation parameters
    blockDensity = 1.0 / 0.05**3 # density = mass / volume
    blockColors = array(( \
      (1.0, 0.0, 0.0, 0.5), (0.0, 1.0, 0.0, 0.5), (0.0, 0.0, 1.0, 0.5), (0.0, 1.0, 1.0 ,0.5),
      (1.0, 0.0, 1.0, 0.5), (1.0, 1.0, 0.0, 0.5), (0.5, 1.0, 0.0, 0.5), (0.5, 0.0, 1.0, 0.5),
      (0.0, 0.5, 1.0, 0.5), (1.0, 0.5, 0.0, 0.5), (1.0, 0.0, 0.5, 0.5), (0.0, 1.0, 0.5, 0.5)  ))

    with self.env:

      # create and position block
      size = uniform(scaleMinMax[0]/2.0, scaleMinMax[1]/2.0, 3)
      mass = (2*size[0])*(2*size[1])*(2*size[2])*blockDensity
      body = openravepy.RaveCreateKinBody(self.env, "")
      body.InitFromBoxes(array([hstack((zeros(3), size))]), True)
      for g in body.GetLinks()[0].GetGeometries():
        g.SetDiffuseColor(blockColors[randint(len(blockColors))])
      body.size = size # body.ComputeAABB.extents() is innacurate, so use this instead
      body.SetName(blockName)
      body.GetLinks()[0].SetMass(mass)
      # https://en.wikipedia.org/wiki/List_of_moments_of_inertia
      inertialMoments = mass/12.0 * array([
        (2*size[1])**2+(2*size[2])**2,
        (2*size[0])**2+(2*size[2])**2,
        (2*size[0])**2+(2*size[1])**2])
      body.GetLinks()[0].SetPrincipalMomentsOfInertia(inertialMoments)
      body.SetTransform(eye(4))
      self.env.Add(body, True)
      body.GetLinks()[0].SetStatic(False)

      # it's also possible to save blocks as ply if ctmconv is installed
      if saveBlock:
        self.env.Save(blockName + ".dae", openravepy.Environment.SelectionOptions.Body, blockName)
        os.system("ctmconv " + blockName + ".dae " + blockName + ".ply")
        print("Saved " + blockName + ".")

      return body

  def PlaceCylinderAtOrigin(self, heightMinMax, radiusMinMax, name, saveBlock=True):
    '''Places a single, random cylinder at the origin of the world frame.'''

    # block creation parameters
    colors = array(( \
      (1.0, 0.0, 0.0, 0.5), (0.0, 1.0, 0.0, 0.5), (0.0, 0.0, 1.0, 0.5), (0.0, 1.0, 1.0 ,0.5),
      (1.0, 0.0, 1.0, 0.5), (1.0, 1.0, 0.0, 0.5), (0.5, 1.0, 0.0, 0.5), (0.5, 0.0, 1.0, 0.5),
      (0.0, 0.5, 1.0, 0.5), (1.0, 0.5, 0.0, 0.5), (1.0, 0.0, 0.5, 0.5), (0.0, 1.0, 0.5, 0.5)  ))

    with self.env:

      # create and position block
      height = uniform(heightMinMax[0], heightMinMax[1])
      radius = uniform(radiusMinMax[0], radiusMinMax[1])
      geomInfo = openravepy.KinBody.Link.GeometryInfo()
      geomInfo._type = openravepy.KinBody.Link.GeomType.Cylinder
      geomInfo._t[0,3] = radius
      geomInfo._vGeomData = [radius, height]
      geomInfo._bVisible = True
      geomInfo._fTransparency = 0.0
      geomInfo._vDiffuseColor = colors[randint(len(colors))]
      body = openravepy.RaveCreateKinBody(self.env, '')
      body.InitFromGeometries([geomInfo])
      body.SetName(name)
      self.env.Add(body, True)

      # it's also possible to save blocks as ply if ctmconv is installed
      if saveBlock:
        self.env.Save(name + ".dae", openravepy.Environment.SelectionOptions.Body, name)
        os.system("ctmconv " + name + ".dae " + name + ".ply")
        print("Saved " + name + ".")

      return body

  def Place3DNetObjectAtOrigin(self, directory, scaleMinMax, objectName, save=True):
    '''Places a single, random 3DNet object at the origin of the world frame.'''

    # objet parameters
    density = 1.0 / 0.05**3 # density = mass / volume
    colors = array(( \
      (1.0, 0.0, 0.0, 0.5), (0.0, 1.0, 0.0, 0.5), (0.0, 0.0, 1.0, 0.5), (0.0, 1.0, 1.0 ,0.5),
      (1.0, 0.0, 1.0, 0.5), (1.0, 1.0, 0.0, 0.5), (0.5, 1.0, 0.0, 0.5), (0.5, 0.0, 1.0, 0.5),
      (0.0, 0.5, 1.0, 0.5), (1.0, 0.5, 0.0, 0.5), (1.0, 0.0, 0.5, 0.5), (0.0, 1.0, 0.5, 0.5)  ))

    with self.env:

      # load objet
      fileNames = os.listdir(directory)
      fileNames = fnmatch.filter(fileNames, "*.ply")
      fileName = fileNames[randint(len(fileNames))]
      threeDNetName = fileName[:-4]
      scale = uniform(scaleMinMax[0], scaleMinMax[1])
      self.env.Load(directory + "/" + fileName, {'scalegeometry':str(scale)})
      body = self.env.GetKinBody(threeDNetName)

      # set object properties
      body.SetName(objectName)
      mass = scale*scale*scale*density
      body.GetLinks()[0].SetMass(mass)
      inertialMoments = mass/12.0 * array([
        (2*scale)**2+(2*scale)**2,
        (2*scale)**2+(2*scale)**2,
        (2*scale)**2+(2*scale)**2])
      body.GetLinks()[0].SetPrincipalMomentsOfInertia(inertialMoments)
      for g in body.GetLinks()[0].GetGeometries():
        g.SetDiffuseColor(colors[randint(len(colors))])
      body.SetTransform(eye(4))
      body.GetLinks()[0].SetStatic(False)

      # it's also possible to save as ply if ctmconv is installed
      if save:
        self.env.Save(objectName + ".dae", openravepy.Environment.SelectionOptions.Body, objectName)
        os.system("ctmconv " + objectName + ".dae " + objectName + ".ply")
        print("Saved " + objectName + ".")

      return body

  def PlaceObjects(self, nObjects, folderName, activeOrientOpts=[0,1,2,3,4,5]):
    '''Dumps a set of random objects on the table.'''

    # parameters
    positionMean = array([0.00, 0.00, 0.15])
    positionSigma = array([0.09, 0.09, 0.09])
    orientOptions = [\
      [(1,0,0),  pi/2, (0, 1,0), 1], \
      [(1,0,0), -pi/2, (0,-1,0), 1],\
      [(0,1,0),  pi/2, (-1,0,0), 0],\
      [(0,1,0), -pi/2, ( 1,0,0), 0],\
      [(0,0,1),     0, (0,0, 1), 2],\
      [(0,1,0),    pi, (0,0,-1), 2]]
    # [axis, angle, newAxis, newAxisIndex]
    maxPlaceAttempts = 10

    # limit orientation options
    newOrientOptions = []
    for idx in activeOrientOpts:
      newOrientOptions.append(orientOptions[idx])
    orientOptions = newOrientOptions

    # check directory
    fileNames = os.listdir(folderName)
    fileNames = fnmatch.filter(fileNames, "*.dae")
    fileIdxs = choice(len(fileNames), size=nObjects, replace=False)
    objectHandles = []

    with self.env:

      # load and position objects
      for i in xrange(nObjects):

        # choose a random object from the folder
        objectName = fileNames[fileIdxs[i]]

        # load object
        self.env.Load(folderName + "/" + objectName)
        shortObjectName = objectName[:-4]
        body = self.env.GetKinBody(shortObjectName)

        # load points and normals
        data = loadmat(folderName + "/" + shortObjectName + ".mat")
        body.cloud = data["cloud"]
        body.normals = data["normals"]

        for j in xrange(maxPlaceAttempts):

          # set position and orientation
          optionIndex = randint(len(orientOptions))
          orientOption= orientOptions[optionIndex]
          randomAngle = 2*pi * rand()

          R1 = openravepy.matrixFromAxisAngle(orientOption[0], orientOption[1])
          R2 = openravepy.matrixFromAxisAngle(orientOption[2], randomAngle)

          position = positionMean + positionSigma*randn(3)
          height = max(body.cloud[:, orientOption[3]]) - min(body.cloud[:, orientOption[3]])

          T = eye(4)
          T[0:3, 3] = position
          T[2, 3] = height / 2.0 + self.tableExtents[2]

          T = dot(T, dot(R1, R2))
          body.SetTransform(T)

          if not self.env.CheckCollision(body): break

        # add to environment
        objectHandles.append(body)

    return objectHandles

  def RemoveObjectSet(self, objectHandles):
    '''Removes all of the objects in the list objectHandles.'''

    with self.env:
      for objectHandle in objectHandles:
        self.env.Remove(objectHandle)

  def ResetObjectPoses(self, objectHandles):
    '''Replaces all objects to their remembered poses.'''

    with self.env:
      for i, obj in enumerate(objectHandles):
        self.objectPoses.append(obj.SetTransform(self.objectPoses[i]))

  def SetObjectPoses(self, objectHandles):
    '''Saves the pose of all objects in the scene to memory.'''

    with self.env:
      self.objectPoses = []
      for obj in objectHandles:
        self.objectPoses.append(obj.GetTransform())

  def StepSimulation(self, duration=10.0, step=0.001):
    '''Runs the simulator forward in time.'''

    #raw_input("Before Simulation")
    with self.env:
      nSteps = int(duration / step)
      for i in xrange(nSteps):
        self.env.StepSimulation(step)
    #raw_input("After Simulation")
'''Provides a class for representing a hand pose and a hand volume.'''

# python
from time import time
from copy import copy
# openrave
import openravepy
# scipy
from matplotlib import pyplot
from numpy.linalg import inv, norm
from numpy.random import rand, randn
from scipy.ndimage.morphology import grey_dilation
from numpy import arccos, arctan, arctan2, array, ascontiguousarray, concatenate, cross, eye, pi, \
  sqrt, stack, zeros
# self
import point_cloud
from c_extensions import SetImageToAverageValues, SetImageToMaxValues

class HandDescriptor():

  def __init__(self, T, image=None, function="", objectName=""):
    '''Creates a HandDescriptor object with everything needed.'''

    self.T = T
    self.image = image
    self.function = function
    self.objectName = objectName

    # hand size (used for drawing)
    self.depth = 0.075
    self.width = 0.085
    self.height = 0.01

    # image size (used for image descriptor)
    self.imP = 60
    self.imD = 0.09
    self.imW = 0.09
    self.imH = 0.09

    # hand axes
    self.axis = T[0:3, 0]
    self.binormal = T[0:3, 1]
    self.approach = T[0:3, 2]
    self.center = T[0:3, 3]
    self.bottom = self.center - 0.5*self.depth*self.approach
    self.top = self.center + 0.5*self.depth*self.approach

    # internal variables
    self.image = None
    self.handPoints = None
    self.handNormals = None

  def AddGaussianNoise(self, orientationSigma, positionSigma):
    '''Returns a new grasp with noise added to the current position and orientation.'''

    descriptor = self if rand() < 0.5 else self.Flip()

    q = openravepy.quatFromRotationMatrix(descriptor.T)
    q += orientationSigma * randn(4)
    q /= norm(q)
    T = openravepy.matrixFromQuat(q)

    c = descriptor.center + positionSigma * descriptor.center
    T[0:3,3] = c

    return HandDescriptor(\
      T, image=None, function=descriptor.function, objectName=descriptor.objectName)

  def ComputeHeightmap(self, X, vAxis):
    '''Projects the points X along vAxis creating an average height map from this.
      - Input X: nx3 numpy array of points.
      - Input vAxis: integer specifying (standard basis) axis to project along.
      - Returns: self.imP x self.imP image.
    '''

    if vAxis == 0:
      xyAxis = (1, 2)
      dxy = (self.imW, self.imD)
    elif vAxis == 1:
      xyAxis = (0, 2)
      dxy = (self.imH, self.imD)
    elif vAxis == 2:
      xyAxis = (0, 1)
      dxy = (self.imH, self.imW)
    else:
      raise Exception("Unknown value axis, {}.".format(vAxis))

    dMax = max(self.imD, self.imW, self.imH)
    im = zeros((self.imP, self.imP), dtype='float32')

    # get image coordinates of each point
    coordsX = (X[:, xyAxis[0]] + (dxy[0]/2.0)) * ((self.imP-1) / dxy[0])
    coordsY = (X[:, xyAxis[1]] + (dxy[1]/2.0)) * ((self.imP-1) / dxy[1])
    coords = stack((coordsX, coordsY), axis=1)
    coords[coords < 0] = 0
    coords[coords > (self.imP-1)] = (self.imP-1)

    # get height of each point
    values = 1 - ((X[:, vAxis] + (dMax / 2.0)) / dMax)

    # set image values at each coordinate
    coords = ascontiguousarray(coords, dtype='float32')
    values = ascontiguousarray(values, dtype='float32')
    SetImageToMaxValues(im, im.shape[0], im.shape[1], coords, values, values.shape[0])

    return im

  def ComputeHeightmapPolar(self, X, vAxis):
    '''Same as ComputeHeightmap except the image axes are [r,theta].'''

    if vAxis == 0:
      xyAxis = (1, 2)
    elif vAxis == 1:
      xyAxis = (0, 2)
    elif vAxis == 2:
      xyAxis = (0, 1)
    else:
      raise Exception("Unknown value axis, {}.".format(vAxis))

    dMax = max(self.imD, self.imW, self.imH)
    im = zeros((self.imP, self.imP), dtype='float32')

    # get image coordinates of each point
    r = (norm(X[:, xyAxis], axis=1) / (sqrt(2.0)*(dMax/2.0))) * (self.imP-1.0)
    theta = ((arctan2(X[:, xyAxis[1]], X[:, xyAxis[0]]) + pi) / (2.0*pi)) * (self.imP-1.0)
    coords = stack((r, theta), axis=1)
    coords[coords < 0] = 0
    coords[coords > (self.imP-1)] = (self.imP-1)

    # get height of each point
    values = 1 - ((X[:, vAxis] + (dMax / 2.0)) / dMax)

    # set image values at each coordinate
    coords = ascontiguousarray(coords, dtype='float32')
    values = ascontiguousarray(values, dtype='float32')
    SetImageToMaxValues(im, im.shape[0], im.shape[1], coords, values, values.shape[0])

    return im

  def ComputeHeightmapSpherical(self, X, vAxis):
    '''Same as ComputeHeightmap except X is first converted into spherical coordinates.'''

    if vAxis == 0:
      xyAxis = (1, 2)
    elif vAxis == 1:
      xyAxis = (0, 2)
    elif vAxis == 2:
      xyAxis = (0, 1)
    else:
      raise Exception("Unknown value axis, {}.".format(vAxis))

    im = zeros((self.imP, self.imP), dtype='float32')

    # convert X to (theta, phi, r)
    r = norm(X, axis=1)
    rPlusEps = copy(r)
    rPlusEps[r==0] = 1e-15 # avoid divide by zero errors

    theta = arccos(X[:,2] / rPlusEps)
    xPlusEps = X[:,0]
    xPlusEps[xPlusEps==0] = 1e-15 # avoid divide by zero errors
    phi = arctan(X[:,1] / xPlusEps)

    # convert (theta, phi, r) into units pixel
    rMax = norm([self.imD/2.0, self.imW/2.0, self.imH/2.0])
    r = (r / rMax) * (self.imP-1.0)
    theta = (theta / pi) * (self.imP-1.0)
    phi = ((phi+(pi/2.0)) / pi) * (self.imP-1.0)

    # catch rounding errors
    S = stack([r,theta,phi], axis=1)
    S[S>=(self.imP)] = self.imP-1.0
    S[S<0] = 0.0

    # get coords and image values
    coords = S[:, xyAxis]
    values = S[:, vAxis]
    values = values / (self.imP-1.0)

    # set image values at each coordinate
    coords = ascontiguousarray(coords, dtype='float32')
    values = ascontiguousarray(values, dtype='float32')
    SetImageToMaxValues(im, im.shape[0], im.shape[1], coords, values, values.shape[0])

    return im

  def ComputeNormalImage(self, X, N, vAxis):
    '''Projects the points X along vAxis and stores the normal x,y,z component here.
      - Input X: nx3 numpy array of points.
      - Input N: nx3 numpy array of normals (assumed to be normalized).
      - Input vAxis: integer specifying (standard basis) axis to project along.
      - Returns im: 3 x self.imP x self.imP image, where each channel is x,y,z normal components.
    '''

    if vAxis == 0:
      xyAxis = (1, 2)
    elif vAxis == 1:
      xyAxis = (0, 2)
    elif vAxis == 2:
      xyAxis = (0, 1)
    else:
      raise Exception("Unknown value axis, {}.".format(vAxis))

    dMax = max(self.imD, self.imW, self.imH)
    im1 = zeros((self.imP, self.imP), dtype='float32')
    im2 = zeros((self.imP, self.imP), dtype='float32')
    im3 = zeros((self.imP, self.imP), dtype='float32')

    # get image coordinates of each point
    coords = (X[:, xyAxis] + (dMax/2.0)) * ((self.imP-1) / dMax)
    coords[coords < 0] = 0
    coords[coords > (self.imP-1)] = (self.imP-1)

    # make between 0 and 1
    N = (N + 1.0) / 2.0

    # set image values at each coordinate
    coords = ascontiguousarray(coords, dtype='float32')
    N = ascontiguousarray(N, dtype='float32')
    SetImageToAverageValues(im1, self.imP, self.imP, coords, N[:, 0].flatten(), N.shape[0])
    SetImageToAverageValues(im2, self.imP, self.imP, coords, N[:, 1].flatten(), N.shape[0])
    SetImageToAverageValues(im3, self.imP, self.imP, coords, N[:, 2].flatten(), N.shape[0])

    return stack((im1, im2, im3), 0)

  def Flip(self):
    '''Creates a new descriptor flipped 180 degrees about the approach direction.'''

    T = copy(self.T)
    T[0:3, 0] = -T[0:3, 0]
    T[0:3, 1] = -T[0:3, 1]

    return HandDescriptor(T, None, self.function, self.objectName)

  def GetHandPoints(self, cloud, normals=None, cloudTree=None):
    '''Determines which points are in the hand descriptor.'''

    # Step 1: Find points within a ball containing the hand.

    if cloudTree is None:
      X = cloud
      N = normals
    else:
      searchRadius = norm(array([self.imD, self.imW, self.imH]) / 2.0)
      indices = cloudTree.query_ball_point(self.center, searchRadius)
      X = cloud[indices, :]
      if normals is not None:
        N = normals[indices, :]

    # Step 2: Transform points into grasp frame.
    # Step 3: Filter out points that are not in the hand.

    hTb = inv(self.T)
    workspace = [(-self.imH/2, self.imH/2), (-self.imW/2, self.imW/2), (-self.imD/2, self.imD/2)]

    if normals is None:
      X = point_cloud.Transform(hTb, X)
      X = point_cloud.FilterWorkspace(workspace, X)
    else:
      X, N = point_cloud.Transform(hTb, X, N)
      X, N = point_cloud.FilterWorkspace(hTb, X, N)

    if normals is None:
      self.handPoints = X
      return X
    self.handNormals = N
    return X, N

  def GenerateDepthImage(self, cloud, cloudTree):
    '''Creates a 3-channel depth image, where each channel is a different axis in the hand frame.'''

    X = self.GetHandPoints(cloud, None, cloudTree)

    # Project points along 3 axes onto images.
    im1 = self.ComputeHeightmap(X, 2)
    im2 = self.ComputeHeightmap(X, 1)
    im3 = self.ComputeHeightmap(X, 0)

    # Apply filter to images.
    im1 = grey_dilation(im1, size=3)
    im2 = grey_dilation(im2, size=3)
    im3 = grey_dilation(im3, size=3)

    # Combine images into one tensor.
    self.image = stack((im1, im2, im3), 0)

  def GenerateDepthImagePolar(self, cloud, cloudTree):
    '''Creates a 3-channel depth image in polar coordinates.'''

    X = self.GetHandPoints(cloud, None, cloudTree)

    # Project points along 3 axes onto images.
    im1 = self.ComputeHeightmapPolar(X, 2)
    im2 = self.ComputeHeightmapPolar(X, 1)
    im3 = self.ComputeHeightmapPolar(X, 0)

    # Apply filter to images.
    im1 = grey_dilation(im1, size=3)
    im2 = grey_dilation(im2, size=3)
    im3 = grey_dilation(im3, size=3)

    # Combine images into one tensor.
    self.image = stack((im1, im2, im3), 0)

  def GenerateDepthImageSpherical(self, cloud, cloudTree):
    '''Creates a 3-channel depth image in polar coordinates.'''

    X = self.GetHandPoints(cloud, None, cloudTree)

    # Project points along 3 axes onto images.
    im1 = self.ComputeHeightmapSpherical(X, 2)
    im2 = self.ComputeHeightmapSpherical(X, 1)
    im3 = self.ComputeHeightmapSpherical(X, 0)

    # Apply filter to images.
    im1 = grey_dilation(im1, size=3)
    im2 = grey_dilation(im2, size=3)
    im3 = grey_dilation(im3, size=3)

    # Combine images into one tensor.
    self.image = stack((im1, im2, im3), 0)

  def GenerateDepthNormalImage(self, cloud, normals, cloudTree):
    '''Creates a 3-channel depth image and a 9-channel normals image.'''

    X, N = self.GetHandPoints(cloud, normals, cloudTree)

    # Project points along 3 axes onto images.
    im1 = self.ComputeHeightmap(X, 2)
    im2 = self.ComputeHeightmap(X, 1)
    im3 = self.ComputeHeightmap(X, 0)
    im4 = self.ComputeNormalImage(X, N, 2)
    im5 = self.ComputeNormalImage(X, N, 1)
    im6 = self.ComputeNormalImage(X, N, 0)

    # Apply filter to images.
    im1 = grey_dilation(im1, size=(3,3))
    im2 = grey_dilation(im2, size=(3,3))
    im3 = grey_dilation(im3, size=(3,3))
    im4 = grey_dilation(im4, size=(1,3,3))
    im5 = grey_dilation(im5, size=(1,3,3))
    im6 = grey_dilation(im6, size=(1,3,3))

    # Combine images into one tensor.
    self.image = stack((im1, im2, im3), 0)
    self.image = concatenate((self.image, im4, im5, im6), 0)

  def GenerateFoveatedDepthImage(self, cloud, cloudTree):
    '''Sets self.image with close-up hand images and more distant hand images.'''

    descriptor2x = copy(self)
    descriptor2x.imD = 2*descriptor2x.imD
    descriptor2x.imW = 2*descriptor2x.imW
    descriptor2x.imH = 2*descriptor2x.imH
    descriptor2x.GenerateDepthImage(cloud, cloudTree)

    self.GenerateDepthImage(cloud, cloudTree)
    self.image = concatenate((self.image, descriptor2x.image), 0)

  def GenerateFoveatedDepthNormalImage(self, cloud, normals, cloudTree):
    '''Sets self.image with zoomed-in depth and normals and zoomed-out depth and normals.'''

    descriptor2x = copy(self)
    descriptor2x.imD = 2*descriptor2x.imD
    descriptor2x.imW = 2*descriptor2x.imW
    descriptor2x.imH = 2*descriptor2x.imH

    self.GenerateDepthNormalImage(cloud, normals, cloudTree)
    descriptor2x.GenerateDepthNormalImage(cloud, normals, cloudTree)

    self.image = concatenate((self.image, descriptor2x.image), 0)

  def IsBelowTable(self, tablePosition):
    '''Checks if any part of the hand is below the table.'''

    a = self.bottom[2] - self.depth * self.approach[2] # arm
    b = self.bottom[2] - (self.width/2.0) * self.binormal[2]
    c = self.bottom[2] + (self.width/2.0) * self.binormal[2]
    d = b + self.depth * self.approach[2]
    e = c + self.depth * self.approach[2]

    return a < tablePosition[2] or \
           b < tablePosition[2] or \
           c < tablePosition[2] or \
           d < tablePosition[2] or \
           e < tablePosition[2]

  def InCollision(self, cloud, cloudTree):
    '''Checks whether each finger and hand back are near points in the cloud.'''

    halfFingerWidth = self.height / 2.0
    collisionBottom = self.bottom - halfFingerWidth*self.approach

    # checks fingers
    startL = self.bottom + ((self.width/2.0)+halfFingerWidth)*self.binormal
    startR = self.bottom - ((self.width/2.0)+halfFingerWidth)*self.binormal
    nSteps = int(round(self.depth/halfFingerWidth))
    for i in xrange(nSteps):
      queryPoint = startL + (nSteps-i-1)*halfFingerWidth*self.approach
      idxs = cloudTree.query_ball_point(queryPoint, halfFingerWidth)
      if len(idxs) > 0: return True
      queryPoint = startR + (nSteps-i-1)*halfFingerWidth*self.approach
      idxs = cloudTree.query_ball_point(queryPoint, halfFingerWidth)
      if len(idxs) > 0: return True

    # checks hand back
    start = collisionBottom - ((self.width/2.0)+halfFingerWidth)*self.binormal
    nSteps = int(round(self.width/halfFingerWidth+3))
    for i in xrange(nSteps):
      queryPoint = start + i*halfFingerWidth*self.binormal
      idxs = cloudTree.query_ball_point(queryPoint, halfFingerWidth)
      if len(idxs) > 0: return True

    # checks arm
    #start = collisionBottom
    #nSteps = int(round(self.depth/halfFingerWidth-1))
    #for i in xrange(nSteps):
    #  queryPoint = start - i*halfFingerWidth*self.approach
    #  idxs = cloudTree.query_ball_point(queryPoint, halfFingerWidth)
    #  if len(idxs) > 0: return True

    return False

  def PlotImage(self):
    '''Plots the image descriptor for this grasp.'''

    if self.image is None:
      return

    nChannels = self.image.shape[0]
    nDepthImages = int(nChannels/3)

    pyplot.figure()

    for i in xrange(nChannels):
      pyplot.subplot(nDepthImages, 3, i+1)
      pyplot.imshow(self.image[i,:,:], cmap='gray')

    pyplot.show(block=True)

# UTILITIES ========================================================================================

def PoseFromApproachAxisCenter(approach, axis, center):
  '''Given grasp approach and axis unit vectors, and center, get homogeneous transform for grasp.'''

  T = eye(4)
  T[0:3, 0] = axis
  T[0:3, 1] = cross(approach, axis)
  T[0:3, 2] = approach
  T[0:3, 3] = center

  return T
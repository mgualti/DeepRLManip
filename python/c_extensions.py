'''Interface to C functions in the extensions folder.'''

import os # python
import ctypes # python
from numpy.ctypeslib import ndpointer # scipy

cextensions = ctypes.cdll.LoadLibrary(os.getcwd() + "/extensions/extensions.so")

SetImageToAverageValues = cextensions.SetImageToAverageValues
SetImageToAverageValues.restype = None
SetImageToAverageValues.argtypes = [\
  ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
  ctypes.c_int,
  ctypes.c_int,
  ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
  ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
  ctypes.c_int]

SetImageToMaxValues = cextensions.SetImageToMaxValues
SetImageToMaxValues.restype = None
SetImageToMaxValues.argtypes = [\
  ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
  ctypes.c_int,
  ctypes.c_int,
  ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
  ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
  ctypes.c_int]
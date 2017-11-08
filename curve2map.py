# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 09:54:17 2015

@author: jmilli
"""


import numpy as np
from scipy.interpolate import interp1d

def create2dMap(values,inputRadii=None,maxRadius=None):
    """
    This function takes a 1D radial distribution in input and builds a 2map 
    """
    nbValues=len(values)
    if inputRadii==None:
        inputRadii=np.arange(0,nbValues)
        maxRadius=nbValues
    else:
        if maxRadius==None:
            raise ValueError('You must provide a maximum radius')
    imageAxis = np.arange(-maxRadius/2,maxRadius/2)
    x,y = np.meshgrid(imageAxis,imageAxis)
    distmap = abs(x+1j*y)
#    map2d = np.ndarray(distmap.shape)
    radiusOK = np.isfinite(values)
    func = interp1d(inputRadii[radiusOK],values[radiusOK],kind='cubic',
                            bounds_error=False,fill_value=np.nan) 
    map2d = func(distmap)
    return map2d,distmap                             
                            
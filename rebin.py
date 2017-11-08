# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 12:47:09 2015

@author: jmilli
"""
import numpy as np

def rebin2d(a, shape):
    """
    Function that rebins a 2d array.
    http://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array
    Input:
    - a: 2d numpy array to rebin
    - shape: tuple with the new size of the array. The new size must be a multiple of the 
    		 original size.
    """
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

def rebin3d(cube, shape2d):
    """
    Function that apply a 2d rebin to each 2d slice of a 3d array
    """
    nframes=cube.shape[0]
    binnedCube=np.ndarray((nframes,shape2d[0],shape2d[1]))
    for z in range(nframes):
        binnedCube[z,:,:]=rebin2d(cube[z,:,:],shape2d)        
    return binnedCube

# unit test
def unitTest():
    """
    Function to test the rebin2d and rebin3d functions implemented above
    """
    x1=np.arange(0,10)
    y1=np.arange(10,20)
    x,y=np.meshgrid(x1,y1)
    
    print(x)
    print(y)
    
    xr=rebin2d(x,(5,10))
    print(xr)
    
    yr=rebin2d(y,(5,10))
    print(yr)
    
    xrr=rebin2d(x,(5,5))
    print(xrr)
    
    cube3d=np.ndarray((2,10,10))
    cube3d[0,:,:]=x
    cube3d[1,:,:]=y
    
    print(cube3d[0,:,:])
    print(cube3d[1,:,:])
    cube3dr = rebin3d(cube3d,(5,10))
    print(cube3dr[0,:,:])
    print(cube3dr[1,:,:])

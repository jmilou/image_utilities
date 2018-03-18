# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 08:41:37 2015

@author: jmilli
"""

import numpy as np
import cv2

def frame_rotate(array, angle, interpolation='bicubic', cy=None, cx=None):
    """ Rotates a frame.
    
    Parameters
    ----------
    array : array_like 
        Input frame, 2d array.
    angle : float
        Rotation angle in degrees. The image is rotated clockwise if the angle is >0
    interpolation : {'bicubic', 'bilinear', 'nearneig'}, optional
        'nneighbor' stands for nearest-neighbor interpolation,
        'bilinear' stands for bilinear interpolation,
        'bicubic' for interpolation over 4x4 pixel neighborhood.
        The 'bicubic' is the default. The 'nearneig' is the fastest method and
        the 'bicubic' the slowest of the three. The 'nearneig' is the poorer
        option for interpolation of noisy astronomical images.
    cy, cx : float, optional
        Coordinates X,Y  of the point with respect to which the rotation will be 
        performed. By default the rotation is done with respect to the center 
        of the frame; central pixel if frame has odd size.
        
    Returns
    -------
    array_out : array_like
        Resulting frame.
        
    """
    if not array.ndim == 2:
        raise TypeError('Input array is not a frame or 2d array.')
    array = np.float32(array)	
    y, x = array.shape
    
    if not cy and not cx:  cy, cx = y//2, x//2
    
    if interpolation == 'bilinear':
        intp = cv2.INTER_LINEAR
    elif interpolation == 'bicubic':
        intp= cv2.INTER_CUBIC
    elif interpolation == 'nearneig':
        intp = cv2.INTER_NEAREST
    else:
        raise TypeError('Interpolation method not recognized.')
    
    M = cv2.getRotationMatrix2D((cx,cy), angle, 1)
    array_out = cv2.warpAffine(array.astype(np.float32), M, (x, y), flags=intp)
             
    return array_out

if __name__=='__main__':  
    import pyds9 
    ds9=pyds9.DS9()
    test_image = np.random.rand(99,99)
    test_image[49,:]=3
    test_image[:,49]=3
    ds9.set_np2arr(test_image)
    test_image_r = frame_rotate(test_image, 90)
    ds9.set_np2arr(test_image_r)
    print(np.sum(test_image),np.sum(test_image_r))

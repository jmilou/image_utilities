#! /usr/bin/env python

"""
Module containing functions for cubes frame registration.
"""

from __future__ import division

__author__ = 'C. Gomez @ ULg'
__all__ = ['frame_shift',
           'frame_center_radon',
           'cube_recenter_radon',
           'cube_recenter_dft_upsampling',
           'cube_recenter_gauss2d_fit',
           'cube_recenter_com',
           'cube_center_fframe']

import numpy as np
import cv2
import photutils
import pywt
import itertools as itt
import pyprind
from scipy.ndimage.interpolation import shift
from skimage.transform import radon
from multiprocessing import Pool, cpu_count
from image_registration import chi2_shift
from matplotlib import pyplot as plt
from ..conf import timeInit, timing, eval_func_tuple
from ..var import (get_square, frame_center, wavelet_denoise, get_annulus, 
                        pp_subplots)


def frame_shift(array, shift_y, shift_x, lib='opencv', interpolation='bicubic'):
    """ Shifts an 2d array by shift_y, shift_x. Boundaries are filled with zeros. 

    Parameters
    ----------
    array : array_like
        Input 2d array.
    shift_y, shift_x: float
        Shifts in x and y directions.
    lib : {'opencv', 'ndimage'}, string optional 
        Whether to use opencv or ndimage library.
    interpolation : {'bicubic', 'bilinear', 'nearneig'}, optional
        'nneighbor' stands for nearest-neighbor interpolation,
        'bilinear' stands for bilinear interpolation,
        'bicubic' for interpolation over 4x4 pixel neighborhood.
        The 'bicubic' is the default. The 'nearneig' is the fastest method and
        the 'bicubic' the slowest of the three. The 'nearneig' is the poorer
        option for interpolation of noisy astronomical images.
    
    Returns
    -------
    array_shifted : array_like
        Shifted 2d array.
        
    """
    if not array.ndim == 2:
        raise TypeError ('Input array is not a frame or 2d array')
    
    image = array.copy()
    
    if lib=='ndimage':
        if interpolation == 'bilinear':
            intp = 1
        elif interpolation == 'bicubic':
            intp= 3
        elif interpolation == 'nearneig':
            intp = 0
        else:
            raise TypeError('Interpolation method not recognized.')
        
        array_shifted = shift(image, (shift_y, shift_x), order=intp)
    
    elif lib=='opencv':
        if interpolation == 'bilinear':
            intp = cv2.INTER_LINEAR
        elif interpolation == 'bicubic':
            intp= cv2.INTER_CUBIC
        elif interpolation == 'nearneig':
            intp = cv2.INTER_NEAREST
        else:
            raise TypeError('Interpolation method not recognized.')
        
        image = np.float32(image)
        y, x = image.shape
        M = np.float32([[1,0,shift_x],[0,1,shift_y]])
        array_shifted = cv2.warpAffine(image, M, (x,y), flags=intp)
        
    else:
        raise ValueError('Lib not recognized, try opencv or ndimage')
    
    return array_shifted


def _radon_costf(frame, cent, radint, coords):
    """ Radon cost function used in frame_center_radon().
    """
    frame_shifted = frame_shift(frame, coords[0], coords[1])
    frame_shifted_ann = get_annulus(frame_shifted, radint, cent-radint)
#    theta = np.linspace(start=0., stop=360., num=frame_shifted_ann.shape[0], 
#                        endpoint=False)

    theta1 = np.linspace(start=35., stop=55., num=frame_shifted_ann.shape[0]/2)
    theta2 = np.linspace(start=125., stop=145., num=frame_shifted_ann.shape[0]/2)
    theta=np.concatenate((theta1,theta2))

    sinogram = radon(frame_shifted_ann, theta=theta, circle=True)
    costf = np.sum(np.abs(sinogram[cent,:]))
    return costf


def frame_center_radon(array, cropsize=101, hsize=0.4, step=0.01, wavelet=False,
                       threshold=1000, mask_center=None, nproc=None, 
                       verbose=True, plot=True, debug=False):
    """ Finding the center of a broadband (co-added) frame with speckles and 
    satellite spots elongated towards the star (center). 
    
    The input frame might be processed with a wavelet filter to enhance the 
    presence of the satellite spots or speckles. The type of wavelet used has
    been tuned empirically, but probably is a per case situation that needs more 
    careful attention. By default the frame is not filtered. 
    
    The radon transform comes from scikit-image package. Takes a few seconds to
    compute one radon transform with good resolution. The whole idea of this
    algorithm is based on Pueyo et al. 2014 paper: 
    http://arxiv.org/abs/1409.6388 
    
    Parameters
    ----------
    array : array_like
        Input 2d array or image.
    cropsize : int, optional
        Size in pixels of the cropped central area of the input array that will
        be used. It should be large enough to contain the satellite spots.
    hsize : float, optional
        Size of the box for the grid search. The frame is shifted to each 
        direction from the center in a hsize length with a given step.
    step : float, optional
        The step of the coordinates change.
    wavelet : {False, True}, bool optional
        Whether to perform a wavelet filtering of the input frame or not.
    threshold : int, optional
        Value for thresholding the wavelet coefficients.
    mask_center : None or int, optional
        If None the central area of the frame is kept. If int a centered zero 
        mask will be applied to the frame. By default the center isn't masked.
    nproc : int, optional
        Number of processes for parallel computing. If None the number of 
        processes will be set to (cpu_count()/2). 
    verbose : {True, False}, bool optional
        Whether to print to stdout some messages and info.
    plot : {True, False}, bool optional
        Whether to plot the radon cost function. 
    debug : {False, True}, bool optional
        Whether to print and plot intermediate info.
    
    Returns
    -------
    optimy, optimx : float
        Values of the Y, X coordinates of the center of the frame based on the
        radon optimization.
    
    """
    from .cosmetics import frame_crop
    
    if not array.ndim==2:
        raise TypeError('Input array is not a frame or 2d array')

    if verbose:  start_time = timeInit()
    frame = array.copy()
    frame = frame_crop(frame, cropsize, verbose=False)
    if wavelet: 
        frame = wavelet_denoise(frame, pywt.Wavelet('bior2.2'), threshold, 100)
        if frame.shape[0] > cropsize:  
            frame = frame[:-1,:-1]
            
    listyx = np.linspace(start=-hsize, stop=hsize, num=2*hsize/step+1, 
                         endpoint=True)
    if not mask_center:
        radint = 0
    else:
        if not isinstance(mask_center, int):
            raise(TypeError('Mask_center must be either None or an integer'))
        radint = mask_center
    
    coords = [(y,x) for y in listyx for x in listyx]
    cent, _ = frame_center(frame)
                
    if debug:
        frame_test = get_annulus(frame, radint, cent-radint)
        theta = np.linspace(start=0., stop=360., num=cent*2, endpoint=False)
        sinogram = radon(frame_test, theta=theta, circle=True)
        pp_subplots(frame_test, sinogram)
        print(np.sum(np.abs(sinogram[cent,:])))
        
    if not nproc:   # Hyper-threading "duplicates" the cores -> cpu_count/2
        nproc = int(cpu_count()/2) 
    pool = Pool(processes=nproc)  
    res = pool.map(eval_func_tuple,itt.izip(itt.repeat(_radon_costf),              
                                            itt.repeat(frame), itt.repeat(cent),
                                            itt.repeat(radint), coords)) 
    costf = np.array(res)
    pool.close()
        
    if verbose:  
        msg = 'Done {} radon transform calls distributed in {} processes'
        print(msg.format(len(coords), nproc))
    
    if plot:          
        cost_bound = costf.reshape(listyx.shape[0], listyx.shape[0])
        plt.contour(cost_bound, cmap='CMRmap', origin='lower', lw=1, hold='on')
        plt.imshow(cost_bound, cmap='CMRmap', origin='lower', 
                   interpolation='nearest')
        plt.colorbar()
        plt.grid('off')
        plt.show()
    
    argm = np.argmax(costf)
    optimy, optimx = coords[argm]
    
    if debug:  print('Cost function max: {}'.format(costf.max()))
    
    if verbose: 
        msg = 'Finished grid search radon optimization. Y={:.5f}, X={:.5f}'
        print(msg.format(optimy, optimx))
        timing(start_time)
    return optimy, optimx
    
      
def cube_recenter_radon(array, full_output=False, verbose=True, **kwargs):
    """ Recenters a cube looping through its frames and calling the 
    frame_center_radon() function. 
    
    Parameters
    ----------
    array : array_like
        Input 3d array or cube.
    full_output : {False, True}, bool optional
        If True the recentered cube is returned along with the y and x shifts.
    verbose : {True, False}, bool optional
        Whether to print timing and intermediate information to stdout.
    
    Optional parameters (keywords and values) can be passed to the     
    frame_center_radon function. 
    
    Returns
    -------
    array_rec : array_like
        Recentered cube.
    If full_output is True:
    y, x : 1d array of floats
        Shifts in y and x.
     
    """
    if not array.ndim == 3:
        raise TypeError('Input array is not a cube or 3d array')

    if verbose:  start_time = timeInit()

    n_frames = array.shape[0]
    x = np.zeros((n_frames))
    y = np.zeros((n_frames))
    array_rec = array.copy()
    
    bar = pyprind.ProgBar(n_frames, stream=1, title='Looping through frames')
    for i in xrange(n_frames):
        y[i], x[i] = frame_center_radon(array[i], verbose=False, plot=False, 
                                        **kwargs)
        array_rec[i] = frame_shift(array[i], y[i], x[i])
        bar.update()
        
    if verbose:  timing(start_time)

    if full_output:
        return array_rec, y, x
    else:
        return array_rec
                

def cube_recenter_dft_upsampling(array, subimage=False, ref_y=None, ref_x=None,
                                 fwhm=4, full_output=False, verbose=True,
                                 save_shifts=False, debug=False):                          
    """ Recenters a cube of frames using the DFT upsampling method as 
    proposed in Guizar et al. 2008 (see Notes) plus a chi^2, for determinig
    automatically the upsampling factor, as implemented in the package 
    'image_registration' (see Notes).
    
    The algorithm (DFT upsampling) obtains an initial estimate of the 
    cross-correlation peak by an FFT and then refines the shift estimation by 
    upsampling the DFT only in a small neighborhood of that estimate by means 
    of a matrix-multiply DFT.
    
    Parameters
    ----------
    array : array_like
        Input cube.
    subimage : {False, True}, bool optional
        Whether to use a subimage instead of the full frame.
    ref_y, ref_x : int
        Coordinates of the center of the subimage.    
    fwhm : float
        FWHM size in pixels.
    full_output : {False, True}, bool optional
        Whether to return 2 1d arrays of shifts along with the recentered cube 
        or not.
    verbose : {True, False}, bool optional
        Whether to print to stdout the timing or not.
    save_shifts : {False, True}, bool optional
        Whether to save the shifts to a file in disk.
    debug : {False, True}, bool optional
        Whether to print to stdout the shifts or not. 
    
    Returns
    -------
    array_recentered : array_like
        The recentered cube.
    If full_output is True:
    y, x : array_like
        1d arrays with the shifts in y and x.     
    
    Notes
    -----
    Package documentation for "Image Registration Methods for Astronomy":
    https://github.com/keflavich/image_registration
    http://image-registration.rtfd.org
    
    Guizar-Sicairos et al. "Efficient subpixel image registration algorithms," 
    Opt. Lett. 33, 156-158 (2008). 
    The algorithm registers two images (2-D rigid translation) within a fraction 
    of a pixel specified by the user. 
    Instead of computing a zero-padded FFT (fast Fourier transform), this code 
    uses selective upsampling by a matrix-multiply DFT (discrete FT) to 
    dramatically reduce computation time and memory without sacrificing 
    accuracy. With this procedure all the image points are used to compute the 
    upsampled cross-correlation in a very small neighborhood around its peak. 
    
    """
    if not array.ndim == 3:
        raise TypeError('Input array is not a cube or 3d array')
    
    if verbose:  start_time = timeInit()
    
    n_frames = array.shape[0]
    x = np.zeros((n_frames))
    y = np.zeros((n_frames))
    array_rec = array.copy()
    if subimage: 
        size = int(fwhm*3)
        sub_image_1 = get_square(array_rec[0], size=size, y=ref_y, x=ref_x)
        
    for i in xrange(1, n_frames):
        if subimage:
            size = int(fwhm*3)
            sub_image = get_square(array[i], size=size, y=ref_y, x=ref_x)
            dx, dy, edx, edy = chi2_shift(sub_image_1, sub_image, 
                                      upsample_factor='auto')
        else:
            dx, dy, edx, edy = chi2_shift(array_rec[0], array[i], 
                                      upsample_factor='auto')
        x[i] = -dx
        y[i] = -dy
        if debug:  print y[i], x[i]
        array_rec[i] = frame_shift(array[i], y[i], x[i])
        
    if verbose:  timing(start_time)
        
    if save_shifts: 
        np.savetxt('recent_dft_shifts.txt', np.transpose([y, x]), fmt='%f')
    if full_output:
        return array_rec, y, x
    else:
        return array_rec
  

def cube_recenter_gauss2d_fit(array, pos_y, pos_x, fwhm=4, full_output=False, 
                              verbose=True, save_shifts=False, debug=False):
    """ Recenters the frames of a cube wrt the 1st one. The shifts are found
    fitting a 2d gaussian to a subimage centered at (pos_x, pos_y). This 
    assumes the frames don't have too large shifts (>5px). The frames are
    shifted using the function frame_shift() (bicubic interpolation).
    
    Parameters
    ----------
    array : array_like
        Input cube.
    pos_y, pos_x : int
        Coordinates of the center of the subimage.    
    fwhm : float
        FWHM size in pixels.
    full_output : {False, True}, bool optional
        Whether to return 2 1d arrays of shifts along with the recentered cube 
        or not.
    verbose : {True, False}, bool optional
        Whether to print to stdout the timing or not.
    save_shifts : {False, True}, bool optional
        Whether to save the shifts to a file in disk.
    debug : {False, True}, bool optional
        Whether to print to stdout the shifts or not. 
        
    Returns
    -------
    array_recentered : array_like
        The recentered cube.
    If full_output is True:
    y, x : array_like
        1d arrays with the shifts in y and x. 
    
    """
    if not array.ndim == 3:
        raise TypeError('Input array is not a cube or 3d array')
    
    if verbose:  start_time = timeInit()
    
    n_frames = array.shape[0]
    x = np.zeros((n_frames))
    y = np.zeros((n_frames))
    array_recentered = np.zeros_like(array)
    
    size = int(fwhm*3)
    sub_image, y1, x1 = get_square(array[0], size=size, y=pos_y, x=pos_x,
                                   position=True)
    sub_image = sub_image.byteswap().newbyteorder()
    x_first, y_first = photutils.morphology.centroid_2dg(sub_image)             # centroid_2dg returns (x,y)
    y_first = y1 + y_first
    x_first = x1 + x_first                                                      # coord of source in first frame
    
    array_recentered[0] = array[0]
    for i in xrange(1, n_frames):
        sub_image, y1, x1 = get_square(array[i], size=13, y=pos_y, x=pos_x,
                                       position=True)
        sub_image = sub_image.byteswap().newbyteorder()
        x_i, y_i = photutils.morphology.centroid_2dg(sub_image)                 
        y_i = y1 + y_i
        x_i = x1 + x_i
        y[i] = y_first-y_i                                                      
        x[i] = x_first-x_i
        if debug:  print y[i], x[i]
        array_recentered[i] = frame_shift(array[i], y[i], x[i])

    if verbose:  timing(start_time)

    if save_shifts: 
        np.savetxt('recent_gauss_shifts.txt', np.transpose([y, x]), fmt='%f')     
    if full_output:
        return array_recentered, y, x
    else:
        return array_recentered


def cube_recenter_com(array, pos_y, pos_x, fwhm=4, full_output=False,
                      verbose=True, save_shifts=False, debug=False):
    """ Recenters the frames of a cube wrt the 1st one. The shifts are found
    getting the centroid of a subimage (center: (pos_x, pos_y)) as its center 
    of mass determined from the image moments. This assumes the frames don't 
    have too large shifts (>5px). The frames are shifted using the function 
    frame_shift() (bicubic interpolation).
    
    Parameters
    ----------
    array : array_like
        Input cube.
    pos_y, pos_x : int
        Coordinates of the center of the subimage.    
    fwhm : float
        FWHM size in pixels.
    full_output : {False, True}, bool optional
        Whether to return 2 1d arrays of shifts along with the recentered cube 
        or not.
    verbose : {True, False}, bool optional
        Whether to print to stdout the timing or not.
    save_shifts : {False, True}, bool optional
        Whether to save the shifts to a file in disk.
    debug : {False, True}, bool optional
        Whether to print to stdout the shifts or not. 
        
    Returns
    -------
    array_recentered : array_like
        The recentered cube.
    If full_output is True:
    y, x : array_like
        1d arrays with the shifts in y and x.
    """
    if not array.ndim == 3:
        raise TypeError('Input array is not a cube or 3d array')
    
    if verbose:  start_time = timeInit()
    
    n_frames = array.shape[0]
    x = np.zeros((n_frames))
    y = np.zeros((n_frames))
    array_recentered = np.zeros_like(array)
    
    size = int(fwhm*3)
    sub_image, y1, x1 = get_square(array[0], size=size, y=pos_y, x=pos_x,
                                      position=True)
    sub_image = sub_image.byteswap().newbyteorder()
    x_first, y_first = photutils.morphology.centroid_com(sub_image)             # centroid_2dg returns (x,y)
    y_first = y1 + y_first
    x_first = x1 + x_first                                                      # coord of source in first frame
    
    array_recentered[0] = array[0]
    for i in xrange(1, n_frames):
        sub_image, y1, x1 = get_square(array[i], size=13, y=pos_y, x=pos_x,
                                       position=True)
        sub_image = sub_image.byteswap().newbyteorder()
        x_i, y_i = photutils.morphology.centroid_com(sub_image)                 
        y_i = y1 + y_i
        x_i = x1 + x_i
        y[i] = y_first-y_i                                                      
        x[i] = x_first-x_i
        if debug:  print y[i], x[i]
        array_recentered[i] = frame_shift(array[i], y[i], x[i])
    
    if verbose:  timing(start_time)
    
    if save_shifts: 
        np.savetxt('recent_com_shifts.txt', np.transpose([y, x]), fmt='%f')     
    if full_output:
        return array_recentered, y, x
    else:
        return array_recentered


def cube_center_fframe(array, ceny, cenx, fwhm=4):
    """ Centers ONLY the first frame of a cube (approx. integer shifts). The
    shifts are found by fitting a 2d gaussian to a subimage centered at (cenx,
    ceny) given coordinates. This cube can be used later with the above 
    functions that will recenter the rest of the frames of the cube wrt the 1st.  
    
    Parameters
    ----------
    array : array_like
        Input cube.
    ceny, cenx : int
        Coordinates of the center of the subimage.
    fwhm : float
        FWHM size in pixels.
    
    Returns
    -------
    array_out : array_like
        Output cube where 1st frame has been shifted to the center.
    
    """
    if not array.ndim == 3:
        raise TypeError('Input array is not a cube or 3d array')
    size = int(fwhm*3)
    sub_image, y1, x1 = get_square(array[0], size=size, y=ceny, x=cenx,
                                   position=True)
    x_fit, y_fit = photutils.morphology.centroid_2dg(sub_image) 
    y_fit = y1 + int(round(y_fit))
    x_fit = x1 + int(round(x_fit))
    cy, cx = frame_center(array[0])
    ynew = cy - y_fit
    xnew = cx - x_fit
    print 'Y_shift = {:}, X_shift = {:}'.format(ynew, xnew)
    array_out = array.copy()
    array_out[0] = frame_shift(array[0], ynew, xnew)
    
    return array_out 
    


        
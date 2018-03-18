#! /usr/bin/env python

"""
2d fitting.
"""
import numpy as np
import os
import pandas as pd
from scipy.optimize import leastsq
from astropy.modeling import models
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import gaussian_sigma_to_fwhm, gaussian_fwhm_to_sigma, sigma_clipped_stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
from scipy.ndimage import fourier_shift
from scipy.ndimage import shift
from multiprocessing import Pool, cpu_count
import itertools as itt
import pyprind
import warnings

try:
    import cv2
    no_opencv = False
except ImportError:
    msg = "Opencv python bindings are missing."
    warnings.warn(msg, ImportWarning)
    no_opencv = True


def fit_2dgaussian(array, crop=False, cent=None, cropsize=15, fwhmx=4, fwhmy=4, 
                   theta=0, threshold=False, sigfactor=6, full_output=False, 
                   plot=True,verbose=True):
    """ Fitting a 2D Gaussian to the 2D distribution of the data with photutils.
    
    Parameters
    ----------
    array : array_like
        Input frame with a single PSF.
    crop : {False, True}, optional
        If True an square sub image will be cropped.
    cent : tuple of int, optional
        X,Y integer position of source in the array for extracting the subimage. 
        If None the center of the frame is used for cropping the subframe (the 
        PSF is assumed to be ~ at the center of the frame). 
    cropsize : int, optional
        Size of the subimage.
    fwhmx, fwhmy : float, optional
        Initial values for the standard deviation of the fitted Gaussian, in px.
    theta : float, optional
        Angle of inclination of the 2d Gaussian counting from the positive X
        axis.
    threshold : {False, True}, optional
        If True the background pixels will be replaced by small random Gaussian 
        noise.
    sigfactor : int, optional
        The background pixels will be thresholded before fitting a 2d Gaussian
        to the data using sigma clipped statistics. All values smaller than
        (MEDIAN + sigfactor*STDDEV) will be replaced by small random Gaussian 
        noise. 
    full_output : {False, True}, optional
        If False it returns just the centroid, if True also returns the 
        FWHM in X and Y (in pixels), the amplitude and the rotation angle.
    plot : {True, False}, optional
        If True, the function prints out parameters of the fit 
    plot : {True, False}, optional
        If True, the function plots the data and residuals with contours
        
    Returns
    -------
    mean_y : float
        Source centroid y position on input array from fitting. 
    mean_x : float
        Source centroid x position on input array from fitting.
        
    If *full_output* is True it returns:
    mean_y, mean_x : floats
        Centroid. 
    fwhm_y : float
        FHWM in Y in pixels. 
    fwhm_x : float
        FHWM in X in pixels.
    amplitude : float
        Amplitude of the Gaussian.
    theta : float
        Rotation angle.
    
    """
    if not array.ndim == 2:
        raise TypeError('Input array is not a frame or 2d array')
    
    if crop:
        if cent is None:
            ceny, cenx = array.shape[1]//2 , array.shape[0]//2
        else:
            cenx, ceny = cent
        
        imside = array.shape[0]
        psf_subimage, suby, subx = get_square(array, min(cropsize, imside), 
                                              ceny, cenx, position=True)  
    else:
        psf_subimage = array.copy()  
    
    if threshold:
        _, clipmed, clipstd = sigma_clipped_stats(psf_subimage, sigma=2)
        indi = np.where(psf_subimage<=clipmed+sigfactor*clipstd)
        subimnoise = np.random.randn(psf_subimage.shape[0], psf_subimage.shape[1])*clipstd#*50
        psf_subimage[indi] = subimnoise[indi]
    
    yme, xme = np.where(psf_subimage==psf_subimage.max())
    # Creating the 2D Gaussian model
    gauss = models.Gaussian2D(amplitude=psf_subimage.max(), x_mean=xme, 
                              y_mean=yme, x_stddev=fwhmx*gaussian_fwhm_to_sigma, 
                              y_stddev=fwhmy*gaussian_fwhm_to_sigma, theta=theta)
    # Levenberg-Marquardt algorithm
    fitter = LevMarLSQFitter()                  
    y, x = np.indices(psf_subimage.shape)
    fit = fitter(gauss, x, y, psf_subimage, maxiter=1000, acc=1e-08)

    if crop:
        mean_y = fit.y_mean.value + suby
        mean_x = fit.x_mean.value + subx
    else:
        mean_y = fit.y_mean.value
        mean_x = fit.x_mean.value 
    fwhm_y = fit.y_stddev.value*gaussian_sigma_to_fwhm
    fwhm_x = fit.x_stddev.value*gaussian_sigma_to_fwhm 
    amplitude = fit.amplitude.value
    theta = fit.theta.value
    
    if plot:
        plot_image_fit_residuals(psf_subimage,psf_subimage-fit(x, y),save=None)
    if verbose:
        print('FWHM_y =', fwhm_y)
        print('FWHM_x =', fwhm_x)
        print()
        print('centroid y =', mean_y)
        print('centroid x =', mean_x)
        print('centroid y subim =', fit.y_mean.value)
        print('centroid x subim =', fit.x_mean.value)
        print() 
        print('peak =', amplitude)
        print('theta =', theta)
    
    if full_output:
        return pd.DataFrame({'centroid_y': mean_y, 'centroid_x': mean_x,
                             'fwhm_y': fwhm_y, 'fwhm_x': fwhm_x,
                             'amplitude': amplitude, 'theta': theta})
    else:
        return mean_y, mean_x

def plot_image_fit_residuals(image,model,save=None):
    """
    Plots a figure with 2 images: the image, and the residuals after subtracting the model.
    On top of the image, the contours of the model are shown
    Parameters
    ----------
    image : array_like
        Input array    
    model : array_like
        Model array
    save: str, opt
        filename where to save the figure (should be a valid path and file name such
        as ~/jmilli/test.pdf)
    """
    x_vect = np.arange(0,image.shape[1])        
    y_vect = np.arange(0,image.shape[0])
    x_array,y_array = np.meshgrid(x_vect,y_vect)
    plt.close(1)
    fig =  plt.figure(1, figsize=(7.5,3))
    gs = gridspec.GridSpec(1,3, height_ratios=[1], width_ratios=[1,1,0.06])
    gs.update(left=0.1, right=0.9, bottom=0.1, top=0.93, wspace=0.2, hspace=0.03)
    ax1 = plt.subplot(gs[0,0]) # Area for the first plot
    ax2 = plt.subplot(gs[0,1]) # Area for the second plot
    ax3 = plt.subplot(gs[0,2]) # Area for the second plot
    im = ax1.imshow(image,cmap='CMRmap',origin='lower', interpolation='nearest',\
        extent=[np.min(x_array),np.max(x_array),np.min(y_array),np.max(y_array)],vmin=np.nanmin(image),vmax=np.nanmax(image))
    ax1.set_xlabel('X in px')
    ax1.set_ylabel('Y in px')
    ax1.contour(x_array,y_array,model,3,colors='w')
    ax1.grid(True,c='w')
    ax2.imshow(image-model,cmap='CMRmap',origin='lower', interpolation='nearest',\
        extent=[np.min(x_array),np.max(x_array),np.min(y_array),np.max(y_array)],vmin=np.nanmin(image),vmax=np.nanmax(image))
    ax2.set_xlabel('X in px')
    ax2.grid(True,c='w')
    fig.colorbar(im, cax=ax3)
    if save is not None:
        if isinstance(save,str) and os.path.exists(os.path.dirname(save)):
            fig.savefig(save)
        else:
            print('The filename {0} is not valid. Enter a valid filename such as "/Users/jmilli/test.pdf"'.format(save))
            
        
def fit_2dmoffat(array, yy, xx, full_output=False,fwhm=4):
    """Fits a star/planet with a 2D circular Moffat PSF.
    
    Parameters
    ----------
    array : array_like
        Subimage with a single point source, approximately at the center. 
    yy : int
        Y integer position of the first pixel (0,0) of the subimage in the 
        whole image.
    xx : int
        X integer position of the first pixel (0,0) of the subimage in the 
        whole image.
    full_output: bool, opt
        Whether to return floor, height, mean_y, mean_x, fwhm, beta, or just 
        mean_y, mean_x
    fwhm: float, opt
        First estimate of the fwhm
    
    Returns
    -------
    floor : float
        Level of the sky background (fit result).
    height : float
        PSF amplitude (fit result).
    mean_x : float
        Source centroid x position on the full image from fitting.
    mean_y : float
        Source centroid y position on the full image from fitting. 
    fwhm : float
        Gaussian PSF full width half maximum from fitting (in pixels).
    beta : float
        "beta" parameter of the moffat function.
    """
    maxi = array.max() # find starting values
    floor = np.ma.median(array.flatten())
    height = maxi - floor
    if height==0.0: # if star is saturated it could be that 
        floor = np.mean(array.flatten())  # median value is 32767 or 65535 --> height=0
        height = maxi - floor

    mean_y = (np.shape(array)[0]-1)/2
    mean_x = (np.shape(array)[1]-1)/2

    fwhm = np.sqrt(np.sum((array>floor+height/2.).flatten()))

    beta = 4
    
    p0 = floor, height, mean_y, mean_x, fwhm, beta

    def moffat(floor, height, mean_y, mean_x, fwhm, beta): # def Moffat function
        alpha = 0.5*fwhm/np.sqrt(2.**(1./beta)-1.)    
        return lambda y,x: floor + height/((1.+(((x-mean_x)**2+(y-mean_y)**2)/\
                                                alpha**2.))**beta)

    def err(p,data):
        return np.ravel(moffat(*p)(*np.indices(data.shape))-data)
    
    p = leastsq(err, p0, args=(array), maxfev=1000)
    p = p[0]
    
    # results
    floor = p[0]                                
    height = p[1]
    mean_y = p[2] + yy
    mean_x = p[3] + xx
    fwhm = np.abs(p[4])
    beta = p[5]
    
    if full_output:
        return floor, height, mean_y, mean_x, fwhm, beta
    else:
        return mean_y, mean_x

def get_square(array, size, y, x, position=False):                 
    """ Returns an square subframe. 
    
    Parameters
    ----------
    array : array_like
        Input frame.
    size : int
        Size of the subframe.
    y : int
        Y coordinate of the center of the subframe (obtained with the function
        ``frame_center``).
    x : int
        X coordinate of the center of the subframe (obtained with the function
        ``frame_center``).
    position : bool optional
        If set to True return also the coordinates of the bottom-left vertex.
        
    Returns
    -------
    array_view : array_like
        Sub array.
        
    """
    if not array.ndim == 2:
        raise TypeError('Input array is not a frame or 2d array.')
    
    # wing is added to the sides of the subframe center
    if size%2 != 0:
        wing = int(np.floor(size / 2.))
    else:
        wing = (size / 2.) - 0.5
    # +1 because python doesn't include the endpoint when slicing
    array_view = array[int(y-wing):int(y+wing+1),
                       int(x-wing):int(x+wing+1)].copy()
    
    if position:
        return array_view, y-wing, x-wing
    else:
        return array_view

def get_square_robust(array, size, y, x, position=False, 
                      out_borders='reduced_square', return_wings=False,
                      strict=False):                 
    """ 
    Returns a square subframe from a larger array robustly (different options in
    case the requested subframe outpasses the borders of the larger array.
    
    Parameters
    ----------
    array : array_like
        Input frame.
    size : int, ideally odd
        Size of the subframe to be returned.
    y, x : int
        Coordinates of the center of the subframe.
    position : bool, {False, True}, optional
        If set to True, returns also the coordinates of the left bottom vertex.
    out_borders: string {'reduced_square','rectangular', 'whatever'}, optional
        Option that set what to do if the provided size is such that the 
        sub-array exceeds the borders of the array:
            - 'reduced_square' (default) -> returns a smaller square sub-array: 
            the biggest that fits within the borders of the array (warning msg)
            - 'rectangular' -> returns a cropped sub-array with only the part 
            that fits within the borders of the array; thus a rectangle (warning
            msg)
            - 'whatever' -> returns a square sub-array of the requested size, 
            but filled with zeros where it outpasses the borders of the array 
            (warning msg)
    return_wings: bool, {False,True}, optional
        If True, the function only returns the size of the sub square
        (this can be used to test that there will not be any size reduction of 
        the requested square beforehand)
    strict: bool, {False, True}, optional
        Set to True when you want an error to be raised if the size is not an 
        odd number. Else, the subsquare will be computed even if size is an even
        number. In the later case, the center is placed in such a way that 
        frame_center function of the sub_array would give the input center 
        (at pixel = half dimension minus 1).
        
    Returns
    -------
    default:
    array_view : array_like
        Sub array of the requested dimensions (or smaller depending on its 
        location in the original array and the selected out_borders option)

    if position is set to True and return_wing to False: 
    array_view, y_coord, x_coord: array_like, int, int 
        y_coord and x_coord are the indices of the left bottom vertex

    if return_wing is set to True: 
    wing: int
        the semi-size of the square in agreement with the out_borders option
    """
    if not array.ndim == 2:
        raise TypeError('Input array is not a frame or 2d array.')
    
    n_y = array.shape[0]
    n_x = array.shape[1]

    if strict:
        if size%2==0: 
            raise ValueError('The given size of the sub-square should be odd.')
        wing_bef = int((size-1)/2)
        wing_aft = wing_bef
    else:
        if size%2==0:
            wing_bef = (size/2)-1
            wing_aft = size/2
        else:
            wing_bef = int((size-1)/2)
            wing_aft = wing_bef

    #Consider the case of the sub-array exceeding the array
    if (y-wing_bef < 0 or y+wing_aft+1 >= n_y or x-wing_bef < 0 or 
        x+wing_aft+1 >= n_x):
        if out_borders=='reduced_square':
            wing_bef = min(y,x,n_y-1-y,n_x-1-x)
            wing_aft = wing_bef
            msg = "!!! WARNING: The size of the square sub-array was reduced"+\
                  " to fit within the borders of the array. Now, wings = "+\
                  str(wing_bef)+"px x "+str(wing_aft)+ "px !!!"
            print(msg)
        elif out_borders=='rectangular':
            wing_y = min(y,n_y-1-y)
            wing_x = min(x,n_x-1-x)
            y_init = y-wing_y
            y_fin = y+wing_y+1
            x_init = x-wing_x
            x_fin = x+wing_x+1
            array_view = array[int(y_init):int(y_fin), int(x_init):int(x_fin)]
            msg = "!!! WARNING: The square sub-array was changed to a "+\
                  "rectangular sub-array to fit within the borders of the "+\
                  "array. Now, [y_init,yfin]= ["+ str(y_init)+", "+ str(y_fin)+\
                  "] and [x_init,x_fin] = ["+ str(x_init)+", "+ str(x_fin)+ "]."
            print(msg)
            if position:
                return array_view, y_init, x_init
            else:
                return array_view
        else:
            msg = "!!! WARNING: The square sub-array was not changed but it"+\
                  " exceeds the borders of the array."
            print(msg)

    if return_wings:
        return wing_bef,wing_aft
    
    else:
        # wing is added to the sides of the subframe center. Note the +1 when 
        # closing the interval (python doesn't include the endpoint)
        array_view = array[int(y-wing_bef):int(y+wing_aft+1),
                           int(x-wing_bef):int(x+wing_aft+1)].copy()
        if position:
            return array_view, y-wing_bef, x-wing_bef
        else:
            return array_view

def _centroid_2dg_frame(cube, frnum, size, pos_y, pos_x, verbose=True,plot=False, \
                        fwhm=4,threshold=False):
    """ Finds the centroid by using a 2d gaussian fitting in one frame from a 
    cube. To be called from within cube_recenter_gauss2d_fit().
    """
    sub_image, y1, x1 = get_square_robust(cube[frnum], size=size, y=pos_y, 
                                          x=pos_x, position=True)
    y_i, x_i = fit_2dgaussian(sub_image, crop=False, fwhmx=fwhm, fwhmy=fwhm, 
                              threshold=threshold, sigfactor=1, verbose=True,plot=False)
    y_i = y1 + y_i
    x_i = x1 + x_i
    return y_i, x_i

def cube_recenter_gauss2d_fit(array, xy, fwhm=4, subi_size=5, nproc=1,
                              imlib='opencv', interpolation='lanczos4',
                              full_output=False, verbose=True, save_shifts=False,
                              offset=None, negative=False, debug=False,
                              threshold=False):
    """ Recenters the frames of a cube. The shifts are found by fitting a 2d 
    gaussian to a subimage centered at (pos_x, pos_y). This assumes the frames 
    don't have too large shifts (>5px). The frames are shifted using the 
    function frame_shift() (bicubic interpolation).
    
    Parameters
    ----------
    array : array_like
        Input cube.
    xy : tuple of int
        Coordinates of the center of the subimage.    
    fwhm : float or array_like
        FWHM size in pixels, either one value (float) that will be the same for
        the whole cube, or an array of floats with the same dimension as the 
        0th dim of array, containing the fwhm for each channel (e.g. in the case
        of an ifs cube, where the fwhm varies with wavelength)
    subi_size : int, optional
        Size of the square subimage sides in terms of FWHM.
    nproc : int or None, optional
        Number of processes (>1) for parallel computing. If 1 then it runs in 
        serial. If None the number of processes will be set to (cpu_count()/2).
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    full_output : {False, True}, bool optional
        Whether to return 2 1d arrays of shifts along with the recentered cube 
        or not.
    verbose : {True, False}, bool optional
        Whether to print to stdout the timing or not.
    save_shifts : {False, True}, bool optional
        Whether to save the shifts to a file in disk.
    offset : tuple of floats, optional
        If None the region of the frames used for the 2d Gaussian fit is shifted
        to the center of the images (2d arrays). If a tuple is given it serves
        as the offset of the fitted area wrt the center of the 2d arrays.
    negative : {False, True}, optional
        If True a negative 2d Gaussian fit is performed.
    debug : {False, True}, bool optional
        If True the details of the fitting are shown. This might produce an
        extremely long output and therefore is limited to <20 frames.
        
    Returns
    -------
    array_recentered : array_like
        The recentered cube. Frames have now odd size.
    If full_output is True:
    y, x : array_like
        1d arrays with the shifts in y and x. 
    
    """    
    if not array.ndim == 3:
        raise TypeError('Input array is not a cube or 3d array')
    
    n_frames = array.shape[0]
    if isinstance(fwhm,int) or isinstance(fwhm,float):  
        fwhm_tmp = fwhm 
        fwhm = np.zeros(n_frames)
        fwhm[:] = fwhm_tmp
    subfr_sz = subi_size*fwhm
    subfr_sz = subfr_sz.astype(int)
        
    if debug and array.shape[0]>20:
        msg = 'Debug with a big array will produce a very long output. '
        msg += 'Try with less than 20 frames in debug mode.'
        raise RuntimeWarning(msg)
    
    pos_x, pos_y = xy
    
    if not isinstance(pos_x,int) or not isinstance(pos_y,int):
        raise TypeError('pos_x and pos_y should be ints')

    # TODO: verify correct handling of even/odd cases
    # If frame size is even we drop a row and a column
    if array.shape[1]%2==0:
        array = array[:,1:,:].copy()
    if array.shape[2]%2==0:
        array = array[:,:,1:].copy()
        
    cy, cx = array.shape[1]//2,array.shape[2]//2
    array_recentered = np.empty_like(array)  
    
    if not nproc:   # Hyper-threading "duplicates" the cores -> cpu_count/2
        nproc = (cpu_count()/2) 
    if nproc==1:
        res = []
        bar = pyprind.ProgBar(n_frames, stream=1, 
                              title='2d Gauss-fitting, looping through frames')
        for i in range(n_frames):
            res.append(_centroid_2dg_frame(array, i, subfr_sz[i], 
                                           pos_y, pos_x, negative, debug, fwhm[i], 
                                           threshold))
            bar.update()
        res = np.array(res)
    elif nproc>1:
        pool = Pool(processes=int(nproc))  
        res = pool.map(eval_func_tuple, zip(itt.repeat(_centroid_2dg_frame),
                                itt.repeat(array), range(n_frames), subfr_sz,
                                itt.repeat(pos_y), itt.repeat(pos_x),
                                itt.repeat(negative), itt.repeat(debug), fwhm,
                                itt.repeat(threshold)))
        res = np.array(res)
        pool.close()
    y = cy - res[:, 0]
    x = cx - res[:, 1]
    #return x, y
    
    if offset is not None:
        offx, offy = offset
        y -= offy
        x -= offx
    
    bar2 = pyprind.ProgBar(n_frames, stream=1, title='Shifting the frames')
    for i in range(n_frames):
        if debug:
            print("\nShifts in X and Y")
            print(x[i], y[i])
        array_recentered[i] = frame_shift(array[i], y[i], x[i], imlib=imlib,
                                          interpolation=interpolation)
        bar2.update()
        
    if save_shifts: 
        np.savetxt('recent_gauss_shifts.txt', np.transpose([y, x]), fmt='%f')
    if full_output:
        return array_recentered, y, x
    else:
        return array_recentered

def frame_shift(array, shift_y, shift_x, imlib='opencv',
                interpolation='lanczos4'):
    """ Shifts a 2D array by shift_y, shift_x. Boundaries are filled with zeros.

    Parameters
    ----------
    array : array_like
        Input 2d array.
    shift_y, shift_x: float
        Shifts in x and y directions.
    imlib : {'opencv', 'ndimage-fourier', 'ndimage-interp'}, string optional
        Library or method used for performing the image shift.
        'ndimage-fourier', does a fourier shift operation and preserves better
        the pixel values (therefore the flux and photometry). Interpolation
        based shift ('opencv' and 'ndimage-interp') is faster than the fourier
        shift. 'opencv' is recommended when speed is critical.
    interpolation : {'bicubic', 'bilinear', 'nearneig'}, optional
        Only used in case of imlib is set to 'opencv' or 'ndimage-interp', where
        the images are shifted via interpolation.
        For 'ndimage-interp' library: 'nearneig', bilinear', 'bicuadratic',
        'bicubic', 'biquartic', 'biquintic'. The 'nearneig' interpolation is
        the fastest and the 'biquintic' the slowest. The 'nearneig' is the
        poorer option for interpolation of noisy astronomical images.
        For 'opencv' library: 'nearneig', 'bilinear', 'bicubic', 'lanczos4'.
        The 'nearneig' interpolation is the fastest and the 'lanczos4' the
        slowest and accurate. 'lanczos4' is the default.
    
    Returns
    -------
    array_shifted : array_like
        Shifted 2d array.

    """
    if not array.ndim == 2:
        raise TypeError('Input array is not a frame or 2d array')
    
    image = array.copy()

    if imlib == 'ndimage-fourier':
        shift_val = (shift_y, shift_x)
        array_shifted = fourier_shift(np.fft.fftn(image), shift_val)
        array_shifted = np.fft.ifftn(array_shifted)
        array_shifted = array_shifted.real

    elif imlib == 'ndimage-interp':
        if interpolation == 'nearneig':
            order = 0
        elif interpolation == 'bilinear':
            order = 1
        elif interpolation == 'bicuadratic':
            order = 2
        elif interpolation == 'bicubic':
            order = 3
        elif interpolation == 'biquartic':
            order = 4
        elif interpolation == 'biquintic':
            order = 5
        else:
            raise TypeError('Scipy.ndimage interpolation method not recognized.')
        
        array_shifted = shift(image, (shift_y, shift_x), order=order)
    
    elif imlib == 'opencv':
        if no_opencv:
            msg = 'Opencv python bindings cannot be imported. Install opencv or '
            msg += 'set imlib to ndimage-fourier or ndimage-interp'
            raise RuntimeError(msg)

        if interpolation == 'bilinear':
            intp = cv2.INTER_LINEAR
        elif interpolation == 'bicubic':
            intp= cv2.INTER_CUBIC
        elif interpolation == 'nearneig':
            intp = cv2.INTER_NEAREST
        elif interpolation == 'lanczos4':
            intp = cv2.INTER_LANCZOS4
        else:
            raise TypeError('Opencv interpolation method not recognized.')
        
        image = np.float32(image)
        y, x = image.shape
        M = np.float32([[1,0,shift_x],[0,1,shift_y]])
        array_shifted = cv2.warpAffine(image, M, (x,y), flags=intp)

    else:
        raise ValueError('Image transformation library not recognized.')
    
    return array_shifted

def eval_func_tuple(f_args):
    """ Takes a tuple of a function and args, evaluates and returns result"""
    return f_args[0](*f_args[1:])                       

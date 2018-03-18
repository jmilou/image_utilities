# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 20:49:37 2015
Modif on 20170207 to replace cv2 by scipy

@author: jmilli
"""
import numpy as np
from scipy import signal,ndimage
from scipy.ndimage import median_filter

def correctBadPixelInFrame(data,threshold=1,verbose=True,details=False):
    """
    This function finds the hot and dead pixels in a 2D dataset. 
    Input:
        - data: the 2d array to correct
        - threshold: (optional) number of standard deviations used to cutoff 
                     the bad pixels.By default 1.
        - verbose:  (optional) to display the number of bad pixels
        - details: (optional) to return a dictionary containing 
                    the dead pixels, hot pixels and the combined dead and hot pixels
    Output:
        - the cleaned data, with bad pixels interpolated using the median neighbours
        - a binary mask of the bad pixels (if details is False), or a dictionary
        containing 3 binary masks (if details is True)
    """
    ny,nx  = data.shape
    nb_pixels = nx*ny
    size=3
    fixed_image = data.astype(np.float32)
#    blurred = cv2.medianBlur(fixed_image, size)
    blurred = signal.medfilt(fixed_image, kernel_size=size)
    difference = data - blurred
    std_diff = np.nanstd(difference)
    med_diff = np.nanmedian(difference)
    threshold_high = med_diff+threshold*std_diff
    threshold_low = med_diff-threshold*std_diff

#    sobel = cv2.Sobel(fixed_image, -1,1,1) 
    dx = ndimage.sobel(fixed_image, 0)  # horizontal derivative
    dy = ndimage.sobel(fixed_image, 1)  # vertical derivative
    sobel = np.hypot(dx, dy)  # magnitude
    sobel *= 255.0 / np.max(sobel)  # normalize (Q&D)

    low_gradient_pixels = np.abs(sobel)/np.std(sobel) < threshold    
    dead_pixels = (difference < threshold_low) & low_gradient_pixels
    high_pixels = difference > threshold_high # these are hot pixels and 
                                                # also some good pixels where 
                                                # the gradient is strong !
    hot_pixels = high_pixels & low_gradient_pixels
    bad_pixels = hot_pixels | dead_pixels
    list_bad_pixels = np.nonzero(bad_pixels)
    if verbose:
        nb_dead_pixels = np.sum(dead_pixels)    
        nb_hot_pixels = np.sum(hot_pixels)
        nb_bad_pixels = np.sum(bad_pixels)     
        print('There are {0:6d}  dead pixels or {1:6.2f}%'.format(nb_dead_pixels,\
                                        float(nb_dead_pixels)/nb_pixels*100.))
        print('There are {0:6d}   hot pixels or {1:6.2f}%'.format(nb_hot_pixels,\
                                            float(nb_hot_pixels)/nb_pixels*100.))
        print('Total:    {0:6d}   bad pixels or {1:6.2f}%'.format(nb_bad_pixels,\
                                            float(nb_bad_pixels)/nb_pixels*100.))
    for y,x in zip(list_bad_pixels[0],list_bad_pixels[1]):
        fixed_image[y,x]=blurred[y,x]
    if details:
        bad_pixels={'dead_pixels':dead_pixels,'hot_pixels':hot_pixels,\
                    'bad_pixels':bad_pixels}
    return fixed_image,bad_pixels

def correctBadPixelInCube(cube,threshold=1,verbose=False):
    """
    This function finds the hot and dead pixels in a 3D dataset (cube of images). 
    Input:
        - data: the 3d cube to correct
        - threshold: (optional) number of standard deviations used to cutoff 
                     the bad pixels.By default 1.
        - verbose:  (optional) to display the number of bad pixels
    Output:
        - a binary mask of the bad pixels (if details is False), or a dictionary
        containing 3 binary masks (if details is True)
        - the data, with bad pixels interpolated using the median neighbours
    """
    if not cube.ndim == 3:
        raise TypeError('\nThe array is not a cube.')
    nbFrames = cube.shape[0]
    cleanedCube = np.copy(cube)
    for i in range(nbFrames):
        cleanedImage,bad_pixels = correctBadPixelInFrame(cube[i,:,:],threshold,verbose)                
        cleanedCube[i,:,:] = cleanedImage
    return cleanedCube
    
def correctZimpolBadPixelInFrame(array, cx,cy,size=5,radius=60,threshold=20,verbose=True):
    """ Corrects the bad pixels in a way adapted for Zimpol frames. The bad pixel is
     replaced by the median of the adjacent pixels.

     Parameters
     ----------
     array : array_like
         Input 2d array.
     cx :  star X location to protect the region close to the star from bac pixel correction
     cy : star Y location to protect the region close to the star from bac pixel correction
     size : odd int, optional
         The size the box (size x size) of adjacent pixels for the median filter.
     radius : int, optional
         Radius of the circular aperture (at the center of the frames) for the
         protection mask.
     verbose : {True, False}, bool optional
         If True additional information will be printed.

     Return
     ------
     frame : array_like
         Frame with bad pixels corrected.
     """
    import cv2
    if not array.ndim == 2:
        raise TypeError('Array is not a 2d array or single frame')
    if size % 2 == 0:
        raise TypeError('Size of the median blur kernel must be an odd integer')
    frame = np.array(array,dtype=np.float32)
    kernel = np.ones((5,5),np.float32)
    kernel[1:4,2]=0
    kernel[2,1:4]=0
    kernel /= np.sum(kernel)
    frame_filtered = cv2.filter2D(frame,-1,kernel)
    bpm = (frame-frame_filtered)>threshold
    if radius>0:
        bpm[cy-radius//2:cy+radius//2,cx-radius//2:cx+radius//2]=False
    bpm[0:2,:]=False
    bpm[-2:,:]=False
    bpm[:,0:2]=False
    bpm[:,-2:]=False
    
    nbBadPixels = np.sum(bpm*1.)
    percentageBadPixels = nbBadPixels/(frame.shape[0]*frame.shape[1])
    if verbose:
        print('{0:.0f} bad pixels or {1:.3f}%'.format(nbBadPixels,100*percentageBadPixels))
    if percentageBadPixels>0.05:
        print('Warning: high number of bad pixels: {0:.0f} or {1:.3f}%'.format(nbBadPixels,100*percentageBadPixels))
    frame_bp_corrected = np.copy(frame)
    frame_bp_corrected[np.where(bpm)] = frame_filtered[np.where(bpm)]
    smoothed = median_filter(frame_bp_corrected, size, mode='nearest')
    frame[np.where(bpm)] = smoothed[np.where(bpm)]
    return frame

def correctZimpolBadPixelInCube(cube, cx,cy,size=5,radius=60,threshold=20,verbose=True):
    """
    This function finds the hot pixels in a 3D dataset (cube of images). 
    Input:
        - data: the 2d array to correct
    Output:
        - the data, with bad pixels interpolated using the median neighbours
    """
    if not cube.ndim == 3:
        raise TypeError('\nThe array is not a cube.')
    nbFrames = cube.shape[0]
    cleanedCube = np.copy(cube)
    for i in range(nbFrames):
        cleanedImage = correctZimpolBadPixelInFrame(cube[i,:,:],cx,cy,size=size,radius=radius,threshold=threshold,verbose=verbose)                
        cleanedCube[i,:,:] = cleanedImage
    return cleanedCube


if __name__ == "__main__":
    from astropy.io import fits
    import os
    pathTarget = '/Users/jmilli/Desktop/test_rebin'
    pathRaw = os.path.join(pathTarget,'raw')
    pathOut = os.path.join(pathTarget,'pipeline')        
    fileNames = 'SPHER.2017-07-14T05:45:20.523_left.fits'
    cube = fits.getdata(os.path.join(pathOut,fileNames))
    cube_corrected = correctBadPixelInCube(cube,threshold=3,verbose=True)
    fits.writeto(os.path.join(pathOut,fileNames).replace('.fits','_bp_corrected.fits'),cube_corrected,overwrite=True)


## test
#from astropy.io import fits
#import numpy as np
#import vip
#ds9=vip.fits.vipDS9()
#
#testFile = '/Volumes/MILOU_1TB_2/HD114082/HD114082_zimpol/raw_2017-02-17/nonsaturated/raw_posang00/SPHER.2017-02-17T07:07:45.022.fits'
#
#hdu = fits.open(testFile)
#hdu.info()
#cubeCallas = hdu[1].data
#cubeCallas.shape
#ds9.display(cubeCallas)
#
#
#t = correctZimpolBadPixelInCube(cubeCallas, 495,484,size=5, verbose=True)
#ds9.display(cubeCallas,t)
#
#img = cubeCallas[0,:,:]
#ds9.display(img)
#
##img_corrected = correctBadPixelInFrame(img,details=False)
##
##ds9.display(img,img_corrected,img-img_corrected)
##
##
##img_corrected2 = vip.preproc.badpixremoval.cube_fix_badpix_clump(img, 488, 492, 3., sig=4.0, protect_psf=True,\
##            verbose=True, half_res_y=False, min_thr=None, mid_thr=None, max_nit=5, full_output=False)
##ds9.display(img,img_corrected2,img-img_corrected2)
##
##bias_hdu = fits.open('/Volumes/MILOU_1TB_2/HD114082/HD114082_zimpol/raw_2017-02-17/calib/SPHER.2017-02-22T18:05:10.738.fits')
##cubeBias = bias_hdu[1].data
##bias = np.median(cubeBias,axis=0)
##
##flat_hdu = fits.open('/Volumes/MILOU_1TB_2/HD114082/HD114082_zimpol/raw_2017-02-17/calib/SPHER.2017-02-22T18:15:42.670.fits')
##cubeFlat = flat_hdu[1].data
##flat = np.median(cubeFlat,axis=0)
##
##ds9.display(cubeFlat-bias,img-img_corrected2)
##
##
##img_corrected3 = vip.preproc.badpixremoval.frame_fix_badpix_isolated(img, bpm_mask=None, sigma_clip=5, num_neig=5,\
##                              size=5, protect_mask=False, radius=30, verbose=True)
##ds9.display(img,img_corrected3,img-img_corrected3)
##
##kernel = np.ones((5,5),np.float32)
##kernel[1:4,2]=0
##kernel[2,1:4]=0
##kernel /= np.sum(kernel)
##ds9.display(kernel)
##
##import cv2
##img = np.array(img,dtype=np.float)
##img_filtered = cv2.filter2D(img,-1,kernel)
##ds9.display(img-img_filtered)
##test = img-img_filtered
##
##bpm = test > 100
##print(np.sum(bpm))
##ds9.display(bpm)
##
##img2 = np.copy(img)
##img2[bpm]=img_filtered[bpm]
##ds9.display(img,img2)
#
#t = correctZimpolBadPixelInFrame(img, 495,484,size=5, verbose=True)
#ds9.display(img,t)

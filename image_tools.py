# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 15:33:27 2016

@author: jmilli
"""

import numpy as np
from scipy.ndimage import fourier_shift,interpolation
import matplotlib.pyplot as plt

def distance_array(shape,centerx=None,centery=None,verbose=True,fullOutput=False):
    """
    Creates a 2d array with the distance from the center of the frame.
    Input: 
        - shape: a tuple indicating the desired shape of the output array, e.g. (100,100)
                The 1st dim refers to the y dimension and the 2nd to the x dimension
        - centerx: the center of the frame from which to compute the distance from
                    by default shape[1]/2 (integer division). Accepts numerical value
        - centery: same for the y dimension
        - verbose: to print a warning for even dimensions
        - fullOutput: if True returns the distance array and in addition the 2d array
                    of x values and y values in 2nd and 3rd ouptuts.
    """
    if len(shape) != 2 :
        raise ValueError('The shape must be a tuple of 2 elements for the y and x dimension!')
    if centerx == None:
        centerx = shape[1]//2
        if np.mod(shape[1],2) == 0 and verbose:
            print('The X dimension is even ({0:d}), the center is assumed to be in {1:d}. Use the option centerx={2:.1f} if the center is between 4 pixels'.format(shape[1],centerx,shape[1]/2.-0.5))
        if np.mod(shape[1],2) == 1 and verbose:
            print('The X dimension is odd ({0:d}), the center is assumed to be in {1:d}'.format(shape[1],centerx))
    if centery == None:
        centery = shape[0]//2
        if np.mod(shape[0],2) == 0 and verbose:
            print('The Y dimension is even ({0:d}), the center is assumed to be in {1:d}. Use the option centery={2:.1f} if the center is between 4 pixels'.format(shape[0],centery,shape[0]/2.-0.5))
        if np.mod(shape[0],2) == 1 and verbose:
            print('The Y dimension is odd ({0:d}), the center is assumed to be in {1:d}'.format(shape[0],centery))
    x_array = np.arange(shape[1])-centerx
    y_array = np.arange(shape[0])-centery
    xx_array,yy_array=np.meshgrid(x_array,y_array)
    dist_center = np.abs(xx_array+1j*yy_array)
    if fullOutput:
        return dist_center,xx_array,yy_array
    return dist_center

def angle_array(shape,centerx=None,centery=None,verbose=True,fullOutput=False):
    """
    Creates a 2d array with the angle in rad (0 is North, then the array is positive going East
    and becomes negative after we pass South).
    Input: 
        - shape: a tuple indicating the desired shape of the output array, e.g. (100,100)
                The 1st dim refers to the y dimension and the 2nd to the x dimension
        - centerx: the center of the frame from which to compute the distance from
                    by default shape[1]/2 (integer division). Accepts numerical value
        - centery: same for the y dimension
        - verbose: to print a warning for even dimensions
        - fullOutput: if True returns the angle array and in addition the 2d array
                    of x values and y values in 2nd and 3rd ouptuts.
    """
    if len(shape) != 2 :
        raise ValueError('The shape must be a tuple of 2 elements for the y and x dimension!')
    if centerx == None:
        centerx = shape[1]//2
        if np.mod(shape[1],2) == 0 and verbose:
            print('The X dimension is even ({0:d}), the center is assumed to be in {1:d}'.format(shape[1],centerx))
    if centery == None:
        centery = shape[0]//2
        if np.mod(shape[0],2) == 0 and verbose:
            print('The Y dimension is even ({0:d}), the center is assumed to be in {1:d}'.format(shape[0],centery))
    x_array = np.arange(shape[1])-centerx
    y_array = np.arange(shape[0])-centery
    xx_array,yy_array=np.meshgrid(x_array,y_array)
    theta = -np.arctan2(xx_array,yy_array)
    if fullOutput:
        return theta,xx_array,yy_array
    return theta

def spider_mask(shape,centerx=None,centery=None,rmin=0,rmax=None,\
                NE=True,NW=True,hwidth=4,verbose=True,fullOutput=False):
    """
    Create a binary mask for the diffraction patter of the spiders
    Input:
        - shape: a tuple indicating the desired shape of the output array, e.g. (100,100)
                The 1st dim refers to the y dimension and the 2nd to the x dimension
        - centerx: the center of the frame from which to compute the distance from
                    by default shape[1]/2 (integer division). Accepts numerical value
        - centery: same for the y dimension
        - rmin: the minimum radius where the mask starts
        - rmax: the maximum radius where the mask starts
        - NE (respectively NW): whether the mask includes the North-East spider
            (respectively North-West)
        - hwidth: half-width of the spider in pixel
        - verbose: to print a warning for even dimensions        
        - fullOutput: if True returns the mask for the NE and SW spiders.
    """
    if len(shape) != 2 :
        raise ValueError('The shape must be a tuple of 2 elements for the y and x dimension!')
    if centerx == None:
        centerx = shape[1]//2
        if np.mod(shape[1],2) == 0 and verbose:
            print('The X dimension is even ({0:d}), the center is assumed to be in {1:d}. Use the option centerx={2:.1f} if the center is between 4 pixels'.format(shape[1],centerx,shape[1]/2.-0.5))
        if np.mod(shape[1],2) == 1 and verbose:
            print('The X dimension is odd ({0:d}), the center is assumed to be in {1:d}'.format(shape[1],centerx))
    if centery == None:
        centery = shape[0]//2
        if np.mod(shape[0],2) == 0 and verbose:
            print('The Y dimension is even ({0:d}), the center is assumed to be in {1:d}. Use the option centery={2:.1f} if the center is between 4 pixels'.format(shape[0],centery,shape[0]/2.-0.5))
        if np.mod(shape[0],2) == 1 and verbose:
            print('The Y dimension is odd ({0:d}), the center is assumed to be in {1:d}'.format(shape[0],centery))
    x_array = np.arange(shape[1])-centerx
    y_array = np.arange(shape[0])-centery
    xx_array,yy_array=np.meshgrid(x_array,y_array)
    dist_center = np.abs(xx_array+1j*yy_array)
    # the normal vector to the spider NE has coordinates (Y,X) = (sin(50),-cos(50))
    spider_pa_rad = np.deg2rad(50.)
    normal_vect_spider_NE_xy = np.array((np.cos(spider_pa_rad),np.sin(spider_pa_rad)))
    dist_to_spider_NE = np.abs(xx_array*normal_vect_spider_NE_xy[0]+yy_array*normal_vect_spider_NE_xy[1])
    mask_NE = np.logical_and(dist_to_spider_NE<hwidth,dist_center>=rmin)
    if rmax is not None:
        mask_NE = np.logical_and(mask_NE,dist_center<=rmax)
    normal_vect_spider_NW_xy = np.array((-np.cos(spider_pa_rad),np.sin(spider_pa_rad)))
    dist_to_spider_NW = np.abs(xx_array*normal_vect_spider_NW_xy[0]+yy_array*normal_vect_spider_NW_xy[1])
    mask_NW = np.logical_and(dist_to_spider_NW<hwidth,dist_center>=rmin)
    if rmax is not None:
        mask_NW = np.logical_and(mask_NW,dist_center<=rmax)
    if fullOutput:
        return mask_NE,mask_NW
    else:
        return np.logical_or(mask_NE,mask_NW)

def spider_mask_any_pa(shape,centerx=None,centery=None,rmin=0,rmax=None,\
                pa=40,hwidth=4,verbose=True):
    """
    Create a binary mask for the diffraction patter of the spiders
    Input:
        - shape: a tuple indicating the desired shape of the output array, e.g. (100,100)
                The 1st dim refers to the y dimension and the 2nd to the x dimension
        - centerx: the center of the frame from which to compute the distance from
                    by default shape[1]/2 (integer division). Accepts numerical value
        - centery: same for the y dimension
        - rmin: the minimum radius where the mask starts
        - rmax: the maximum radius where the mask starts
        - pa: the position angle of the spider. By default it is -40º
        - verbose: to print a warning for even dimensions        
    """
    if len(shape) != 2 :
        raise ValueError('The shape must be a tuple of 2 elements for the y and x dimension!')
    if centerx == None:
        centerx = shape[1]//2
        if np.mod(shape[1],2) == 0 and verbose:
            print('The X dimension is even ({0:d}), the center is assumed to be in {1:d}. Use the option centerx={2:.1f} if the center is between 4 pixels'.format(shape[1],centerx,shape[1]/2.-0.5))
        if np.mod(shape[1],2) == 1 and verbose:
            print('The X dimension is odd ({0:d}), the center is assumed to be in {1:d}'.format(shape[1],centerx))
    if centery == None:
        centery = shape[0]//2
        if np.mod(shape[0],2) == 0 and verbose:
            print('The Y dimension is even ({0:d}), the center is assumed to be in {1:d}. Use the option centery={2:.1f} if the center is between 4 pixels'.format(shape[0],centery,shape[0]/2.-0.5))
        if np.mod(shape[0],2) == 1 and verbose:
            print('The Y dimension is odd ({0:d}), the center is assumed to be in {1:d}'.format(shape[0],centery))
    x_array = np.arange(shape[1])-centerx
    y_array = np.arange(shape[0])-centery
    xx_array,yy_array=np.meshgrid(x_array,y_array)
    dist_center = np.abs(xx_array+1j*yy_array)
    # the normal vector to the spider NE has coordinates (Y,X) = (sin(50),-cos(50))
    spider_pa_rad = np.deg2rad(pa)
    normal_vect_spider_NE_xy = np.array((np.cos(spider_pa_rad),np.sin(spider_pa_rad)))
    dist_to_spider_NE = np.abs(xx_array*normal_vect_spider_NE_xy[0]+yy_array*normal_vect_spider_NE_xy[1])
    mask_NE = np.logical_and(dist_to_spider_NE<hwidth,dist_center>=rmin)
    if rmax is not None:
        mask_NE = np.logical_and(mask_NE,dist_center<=rmax)

    spider_NW_pa_rad = np.deg2rad(pa+80)
    normal_vect_spider_NW_xy = np.array((np.cos(spider_NW_pa_rad),np.sin(spider_NW_pa_rad)))
    dist_to_spider_NW = np.abs(xx_array*normal_vect_spider_NW_xy[0]+yy_array*normal_vect_spider_NW_xy[1])
    mask_NW = np.logical_and(dist_to_spider_NW<hwidth,dist_center>=rmin)
    if rmax is not None:
        mask_NW = np.logical_and(mask_NW,dist_center<=rmax)
    return np.logical_or(mask_NE,mask_NW)

def shift_image_nofft(image,dx,dy,verbose=True):
    """ 
    Shifts an 2d array by dx, dy.  
    If dx is positive, the image is shifted to the right
    If dy is positive, the image is shifted up.
    """
    if not image.ndim == 2:
        raise TypeError ('Input array is not a frame or 2d array')
    if isinstance(dx, (list, tuple, np.ndarray)) or isinstance(dy, (list, tuple, np.ndarray)):
            if len(dx)>1 or len(dy)>1:
                raise TypeError ('Input shift dx or dy is not a one-element list')
            else:
                dx = dx[0]
                dy = dy[0]
    image = np.array(image,dtype=np.float)
    if np.any(~np.isfinite(image)):
        if verbose:
            print('Warning {0:d} pixels with nan values were set to 0'.format(np.sum(~np.isfinite(image))))        
        image[~np.isfinite(image)]=0
    image_shifted = interpolation.shift(image, (dy,dx),order=3,prefilter=True)
    return image_shifted

def shift_cube_nofft(cube,dx,dy,verbose=True):
    """ 
    Shifts a cube by dx, dy. dx and dy can be either 2 lists or 2 floats if the 
    shift is the same for all frames.    
    If dx is positive, the image is shifted to the right
    If dy is positive, the image is shifted up.
    """
    if cube.ndim == 2:
        if isinstance(dx, (list, tuple, np.ndarray)) or isinstance(dy, (list, tuple, np.ndarray)):
            return shift_image(cube,dx,dy)
    elif not cube.ndim == 3:
        raise TypeError ('Input array is not a cube')
    nb_frames = cube.shape[0]
    if isinstance(dx, (float,int,complex)):
        dx = np.ones(nb_frames)*dx
    if isinstance(dy, (float,int,complex)):
        dy = np.ones(nb_frames)*dy
    if len(dx)!= nb_frames or len(dy)!=nb_frames:
        raise TypeError ('Input list of shifts has not the same length as the number of frames')
    cube_shifted = np.zeros_like(cube)
    if np.any(~np.isfinite(cube)):
        if verbose:
            print('Warning {0:3.1f}% pixels with nan values were set to 0'.format(\
                  np.sum(~np.isfinite(cube))*100./(cube.shape[0]*cube.shape[1]*cube.shape[2])))
    for i in range(nb_frames):
        cube_shifted[i,:,:] = shift_image_nofft(cube[i,:,:],dx[i],dy[i],verbose=False)
    return cube_shifted


def shift_image(image,dx,dy,verbose=True):
    """ 
    Shifts an 2d array by dx, dy.  
    If dx is positive, the image is shifted to the right
    If dy is positive, the image is shifted up.
    """
    if not image.ndim == 2:
        raise TypeError ('Input array is not a frame or 2d array')
    if isinstance(dx, (list, tuple, np.ndarray)) or isinstance(dy, (list, tuple, np.ndarray)):
            if len(dx)>1 or len(dy)>1:
                raise TypeError ('Input shift dx or dy is not a one-element list')
            else:
                dx = dx[0]
                dy = dy[0]
    image = np.array(image,dtype=np.float)
    if np.any(~np.isfinite(image)):
        if verbose:
            print('Warning {0:d} pixels with nan values were set to 0'.format(np.sum(~np.isfinite(image))))        
        image[~np.isfinite(image)]=0
    image_shifted = fourier_shift(np.fft.fftn(image), (dy,dx))
    image_shifted = np.fft.ifftn(image_shifted)
    image_shifted = image_shifted.real
    return image_shifted

def shift_cube(cube,dx,dy,verbose=True):
    """ 
    Shifts a cube by dx, dy. dx and dy can be either 2 lists or 2 floats if the 
    shift is the same for all frames.    
    If dx is positive, the image is shifted to the right
    If dy is positive, the image is shifted up.
    """
    if cube.ndim == 2:
        if isinstance(dx, (list, tuple, np.ndarray)) or isinstance(dy, (list, tuple, np.ndarray)):
            return shift_image(cube,dx,dy)
    elif not cube.ndim == 3:
        raise TypeError ('Input array is not a cube')
    nb_frames = cube.shape[0]
    if isinstance(dx, (float,int,complex)):
        dx = np.ones(nb_frames)*dx
    if isinstance(dy, (float,int,complex)):
        dy = np.ones(nb_frames)*dy
    if len(dx)!= nb_frames or len(dy)!=nb_frames:
        raise TypeError ('Input list of shifts has not the same length as the number of frames')
    cube_shifted = np.zeros_like(cube)
    if np.any(~np.isfinite(cube)):
        if verbose:
            print('Warning {0:3.1f}% pixels with nan values were set to 0'.format(\
                  np.sum(~np.isfinite(cube))*100./(cube.shape[0]*cube.shape[1]*cube.shape[2])))
    for i in range(nb_frames):
        cube_shifted[i,:,:] = shift_image(cube[i,:,:],dx[i],dy[i],verbose=False)
    return cube_shifted

##test of the implementation of the shift
#import vip
#ds9 = vip.fits.vipDS9()
#test = np.random.randn(2,20,40)
#print(np.any(~np.isfinite(test)))
#test[0,1,1]=np.nan
#test[0,5,:]=np.nan
#print(np.any(~np.isfinite(test)))
#test[:,10,:]=2
#test[:,:,10]=2
#ds9.display(test)
#test_shifted = shift_cube(test,3.,3.1,verbose=True)
#ds9.display(test,test_shifted)

def bin_array(array,binned_indices,func=np.mean):
    """
    Given a 1D-array and an integer array (binned_indices) containing the 
    new indices of the elements to be binned together, returns the binned 1D array.
    For example:
        array = [0,1,2,3,4,5,6,7,8]
        binned_indices = [0,0,0,1,1,2,-1,2]
        then binned_array = [mean([0,1,2,4]),mean([4,5]),mean([6,7,8])] if method='mean'
    Input:
        - array: a 1D array
        - binned_indices: a 1D integer array of same length as array containing 
                the new indices of the elements of the original array to be binned 
                together. Any negative index is considered as a bad daa and ignored
        - func: np.mean by default (can be changed to np.median or any other function
    Output:
        - binner_array: the rebinned array of length max(binned_indices)
    """
    nb = len(array)
    array = np.asarray(array)
    if len(binned_indices)!=nb:
        raise IndexError('The input array "binned_indices" must have the same length as "array"')
    unique_entries,unique_indices,unique_counts=np.unique(binned_indices,return_index=True,return_counts=True)
    good_indices = unique_entries>=0
    unique_entries = unique_entries[good_indices]
    unique_indices = unique_indices[good_indices]
    unique_counts = unique_counts[good_indices]
    nb_rebinned = np.max(unique_entries)+1
    # unique_entries should be a list of integers from 0 to nb_rebinned.
    if not np.all(unique_entries==np.arange(nb_rebinned)):
        raise IndexError('The input array "binned_indices" must contain all integers from 0 to len(binned_array)-1')
    binned_array = np.ndarray((nb_rebinned),dtype=float)
    for new_index in unique_entries:
        if unique_counts[new_index]==1:
            binned_array[new_index] = unique_entries[new_index]
        else:
            test_array = np.ndarray((nb),dtype=int)
            test_array.fill(new_index)
            id_to_bin = (binned_indices==test_array)
            binned_array[new_index] = func(array[id_to_bin])
    return binned_array            

#array = [0,1,2,3,4,5,6,7,8]
#binned_indices = [-2,0,0,1,1,2,2,2,1]
#binned_array =bin_array(array,binned_indices)
#print(binned_array)
#then binned_array = [mean([0,1,2,4]),mean([4,5]),mean([6,7,8])] if method='mean'


def bin_cube(cube,binned_indices,func=np.mean):
    """
    Given a cube of frames and an integer array (binned_indices) containing the 
    new indices of the frames to be binned together, returns the binned cube.
    Input:
        - cube: a 3D array
        - binned_indices: a 1D integer array of same length as array containing 
                the new indices of the elements of the original array to be binned 
                together. Any negative index is considered as a bad daa and ignored
        - func: np.mean by default (can be changed to np.median or any other function
                    accepting the option axis=0
    Output:
        - binned_cube: the rebinned cube of length max(binned_indices) along axis 0
    """
    nb = cube.shape[0]
    cube = np.asarray(cube)
    if len(binned_indices)!=nb:
        raise IndexError('The input array "binned_indices" must be of length cube.shape[0]')
    unique_entries,unique_indices,unique_counts=np.unique(binned_indices,return_index=True,return_counts=True)
    good_indices = unique_entries>=0
    unique_entries = unique_entries[good_indices]
    unique_indices = unique_indices[good_indices]
    unique_counts = unique_counts[good_indices]
    nb_rebinned = np.max(unique_entries)+1
    # unique_entries should be a list of integers from 0 to nb_rebinned.
    if not np.all(unique_entries==np.arange(nb_rebinned)):
        raise IndexError('The input array "binned_indices" must contain all integers from 0 to len(binned_array)-1')
    binned_cube = np.ndarray((nb_rebinned,cube.shape[1],cube.shape[2]),dtype=float)
    for new_index in unique_entries:
        if unique_counts[new_index]==1:
            binned_cube[new_index,:,:] = cube[new_index,:,:]
        else:
            test_array = np.ndarray((nb),dtype=int)
            test_array.fill(new_index)
            id_to_bin = (binned_indices==test_array)
            binned_cube[new_index,:,:] = func(cube[id_to_bin,:,:],axis=0)
    return binned_cube            

#import vip
#ds9 = vip.fits.vipDS9()
#cube = np.random.randn(9,40,40)
#cube[3,:,:] = 10
#binned_indices = [3,0,0,1,1,2,2,2,1]
#binned_cube =bin_cube(cube,binned_indices)
#ds9.display(cube,binned_cube)

def make_binned_indices(score,rebin,extrareject=0,plot=True):
    """
    Given a score (highest is better), we make a array of indices to rebin the data 
    """
    nb=len(score)
    nb_reject = np.mod(nb,rebin)
    nb_rebinned = nb//rebin
    if nb_reject==0:
        print('You can use the whole array or reject k*{1:d} elements'.format(nb_reject,rebin))
    else:
        print('You can reject {0:d} elements or any {0:d}+k*{1:d} elements'.format(nb_reject,rebin))
    sorted_score = sorted(score)
    rebinned_sorted_score = np.ndarray((nb_rebinned))
    for i in np.arange(nb_reject,nb,rebin):
        rebinned_sorted_score[i//rebin] = np.mean(sorted_score[i:i+rebin])
    if plot:
        plt.close(1)
        plt.figure(1)
        plt.plot(sorted_score,'b-',label='original score')
        plt.plot(np.arange(nb_reject,nb,rebin)+rebin*0.5,rebinned_sorted_score,'g-',label='rebinned score')
        for k in range(5):
            plt.plot(np.zeros(2)+nb_reject+k*rebin,[np.min(score),np.max(score)],'r-')
        plt.ylabel('Score')
        plt.xlabel('Sorted index')
    print('You decided to reject {0:d} frames. Use the keyword extrareject=0,1 or 2 if you want exclude {1:d},{2:d} or {3:d} frames'.format(nb_reject+extrareject*rebin,nb_reject,nb_reject+rebin,nb_reject+2*rebin))
    threshold = sorted_score[nb_reject+extrareject*rebin]
    id_to_keep = score>=threshold
    if np.all(id_to_keep):
        return np.arange(nb)//rebin
    else:
        binned_indices = np.ndarray((nb),dtype=int)
        indices_good = 0
        for i,s in enumerate(score):
            if s>=threshold:
                binned_indices[i]=indices_good//rebin
                indices_good = indices_good+1
            else:
                binned_indices[i]=-1
    return binned_indices

def subtract_median(image,row=True,column=False):
    """
    Computes the median in each row/column/depth and subtracts it. 
    This can be useful to get rid of some electronic
    noise
    Input:
        - image: a 2D image or a 3D cube
        - row: if True removes the median value of each row
        - column: if True removes the median value of each column
    Output:
        - the filtered image
    """
    if image.ndim==2:
        if row:
            filtered_image = image - np.nanmedian(image,axis=1, keepdims=True)
        if column:
            filtered_image = image - np.nanmedian(image,axis=0, keepdims=True)      
    elif image.ndim==3:
        if row:
            filtered_image = image - np.nanmedian(image,axis=2, keepdims=True)
        if column:
            filtered_image = image - np.nanmedian(image,axis=1, keepdims=True)      
    else:
        raise TypeError('The input array "image" must be a 2D or 3D numpy array')
    return filtered_image
            
if __name__ == "__main__":
    score = np.random.randn(40)
    binned_indices = make_binned_indices(score,7,plot=True)
    print(binned_indices)
    binned_array =bin_array(score,binned_indices)
    print(binned_array)


    cube = np.random.rand(2,10,10) 
    cube[0,3,:] += 2
    filtered_cube = subtract_median(cube,row=True,column=False)
    plt.figure(0)
    plt.imshow(cube[0,:,:])
    plt.figure(1)
    plt.imshow(filtered_cube[0,:,:])

    dist_spider = spider_mask((100,100),hwidth=0.5,rmax=50,rmin=20)
    plt.figure(0)
    plt.imshow(dist_spider,origin='lower')
    

    mask_spider_test = spider_mask_any_pa((400,400),centerx=200,centery=200,rmin=100,rmax=200,\
                pa=-40,hwidth=4,verbose=True)

#    image = np.random.rand(10,10) 
#    image[3,:] += 2
#    filtered_image = subtract_median(image,row=False,column=True)
#    plt.figure(0)
#    plt.imshow(image)
#    plt.figure(1)
#    plt.imshow(filtered_image)

#cube[3,:,:] = 10
#binned_indices = [3,0,0,1,1,2,2,2,1]
#binned_cube =bin_cube(cube,binned_indices)
#ds9.display(cube,binned_cube)

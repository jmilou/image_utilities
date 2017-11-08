# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 12:47:09 2015

@author: jmilli
"""
import numpy as np
import matplotlib.pyplot as plt

def measureFWHM(array,offset=0.,fulloutput=False,verbose=True,plot=True):
    """
    Function that takes in input an array and measure the FWHM (linear interpolation)
    It assumes by default a 0 offset.
    """
    if array.ndim != 1:
        raise TypeError('The input array is not 1D')
    len_array = len(array)
    max_array = np.max(array)
    argmax_array = np.argmax(array)
    threshold = max_array-(max_array-offset)/2.    
    i=argmax_array
    while(i<len_array and array[i]>threshold):
        i +=1
    hwhm_after = (array[i-1]-threshold)/(array[i-1]-array[i]) + (i-1-argmax_array)
    if verbose:
        print('HWHM after = {0:6.1f}'.format(hwhm_after))
    i=argmax_array
    while (i>0 and array[i]>threshold):
        i -=1
    hwhm_before = 1-(threshold-array[i])/(array[i+1]-array[i])+argmax_array-(i+1)
    fwhm = hwhm_before+hwhm_after
    if verbose:
        print('HWHM before = {0:6.1f}'.format(hwhm_before))
        print('FWHM = {0:6.1f}'.format(fwhm))        
    if plot:
        plt.plot(array,color='blue')
        plt.plot([argmax_array],[max_array],'og')
        plt.plot(argmax_array+np.array([-hwhm_before,0,hwhm_after]),np.ones(3)*threshold,'or')
    if fulloutput:
        return hwhm_before,hwhm_after,fwhm
    return fwhm

def getFWHMuncertainty(array,error,offset=0.,nbRealizations=long(10000)):
    """
    Computes the uncertainty on the FWHM by doing many measurements (10 000 by default)
    after introducing gaussian noise of 0 mean and dispersion error 
    Returns the standard deviation of the measurements
    """
    fwhm_array =np.ndarray((nbRealizations))
    for k in range(nbRealizations):
        array_with_noise = array+error*np.random.randn(len(array))
        fwhm_array[k] = measureFWHM(array_with_noise,offset=offset,verbose=False,plot=False,fulloutput=False)
    mean_fwhm = np.mean(fwhm_array)
    med_fwhm = np.median(fwhm_array)
    std_fwhm = np.std(fwhm_array)
    print('Mean    = {0:6.2f}'.format(mean_fwhm))
    print('Median  = {0:6.2f}'.format(med_fwhm))
    print('RMS     = {0:6.2f}'.format(std_fwhm))
    return std_fwhm

def sig2fwhm(sigma):
    """
    Converts a sigma from a gaussian into a FWHM
    """
    return 2*sigma*np.sqrt(2*np.log(2))

def fwhm2sig(fwhm):
    """
    Converts a FWHM from a gaussian into a sigma
    """
    return fwhm/(2*np.sqrt(2*np.log(2)))

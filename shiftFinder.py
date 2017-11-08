#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 14:38:45 2017

@author: jmilli
"""
from astropy.io import fits ,ascii
#import os,sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
#import glob
#import vip
#ds9=vip.fits.vipDS9()
from astropy.modeling.functional_models import Gaussian2D
import mpfit 
from fwhm import fwhm2sig,sig2fwhm
from scipy.ndimage import gaussian_filter, median_filter  
import numpy as np

class ShiftFinder():
    """ Object to find the star in an image or a shift between 2 images, that implements 
    different techniques of peak fitting or correlation.
    
    Attributes:
        - image: the input image
        - nx: the x dimension of the image
        - ny: the y dimension of the image
        - mask: binary mask for the region of interest
        - sky_mask: binary mask for the sky region
        - sky_median: median value fo the sky
        - sky_rms: standard deviation of the sky
        - subimage_half_size_x , subimage_half_size_y : integer 
                for the half size of the subimage to crop
                before analyzing the subimage. The subimage is square with a side of 
                2*subimage_half_size+1 (odd dimension)
        - sub_image
        - self.guess_x, self.guess_y: the guessed position of the star, also used to crop the image
    Methods:
        -
    """ 

    def __init__(self,image,crop=None,mask=None,guess_xy=None,sky=None,threshold=None):
        """
        Constructor of the class. 
        Preprocess the image by cropping it (if asked) around a guessed center
        and subtracting a sky offset (if asked) and masking the pixels unwanted 
        for the registration (if asked).
        Input:
            - image: the full frame image to analyze
            - crop: (optional) integer, to specify the half-size of the subimage if the image 
                    has to be cropped before fitting the 2D gaussian
            - mask: (optional) a binary mask to specify the pixels where the star is 
                    located. If None (default), the whole image is used
                    True is for good pixels and False for bad pixels
            - sky: (optional) a binary mask to specify the pixels to be used to measure the
                    background of the image, to be subtracted before doing                     
                    the fit. Pixels with True are used for the sky evaluation/
            - threshold (optional) a threshold to discard pixels above that value from the fit
        Output:
            - nothing
        """            
        if not image.ndim == 2:
            raise TypeError('The input image is not a 2d array.')
        self.image = np.copy(image)
        self.ny,self.nx = image.shape

        # We set the mask for the valid pixels for the fit
        if mask is not None:
            if (mask.ndim != 2) or mask.shape[1] != self.nx or mask.shape[0] != self.ny:
                raise TypeError('The binary mask image has the wrong shape')

        if mask is None:
            self.mask_fit = np.ones_like(self.image,dtype=bool)
        else:
            self.mask_fit = mask
        if threshold is not None: # we add the pixels above the threshold in the mask
            self.mask_fit[self.image>threshold] = False

        # We set the mask for the valid pixles to evaluate the sky background
        if sky is not None:
            if (sky.ndim != 2) or sky.shape[1] != self.nx or sky.shape[0] != self.ny:
                raise TypeError('The sky mask image has the wrong shape')
        self.mask_sky = sky

        # We set the guessed star center value
        if guess_xy is not None:
            if len(guess_xy) != 2:
                raise TypeError('The guess_xy parameter must be a 2-element list')
            elif guess_xy[1]<=0 or guess_xy[1]>=self.ny:
                raise TypeError('The guessed Y center (guess_xy[1]) must be between 0 and {1:d}'.format(self.ny))
            elif guess_xy[0]<=0 or guess_xy[0]>=self.nx:
                raise TypeError('The guessed X center (guess_xy[0]) must be between 0 and {1:d}'.format(self.nx))
            else:
                self.guess_x,self.guess_y = guess_xy
        else:
            self.guess_x,self.guess_y = image.shape[1]//2,image.shape[0]//2

        if crop is not None:
            self.subimage_half_size_x = crop
            self.subimage_half_size_y = crop
        else:
            self.subimage_half_size_x = image.shape[1]//2
            self.subimage_half_size_y = image.shape[0]//2         
        self.preprocess_image()
        
    def preprocess_image(self):
        """
        Prepare the image for the analsis: crop it if asked, subtract the background
        """
        if self.mask_sky is not None:
            sky_median = np.median(self.image[self.mask_sky])
            sky_std = np.nanstd(self.image[self.mask_sky])
            if np.isfinite(sky_median) and np.isfinite(sky_std):
#                print('Sky = {0:3.1e} RMS = {1:3.1e}'.format(sky_median,sky_std))
                self.sky_median = sky_median
                self.sky_rms = sky_std
            else:
                raise TypeError('The evaluation of the sky yielded nan for the median or std.')
        else:
            self.sky_median = 0
            self.sky_rms = 1
#        if self.mask_fit is not None:
        self.image[~self.mask_fit] = np.nan
        
        self.subimage = self.image[self.guess_y-self.subimage_half_size_y:self.guess_y+self.subimage_half_size_y,\
                                       self.guess_x-self.subimage_half_size_x:self.guess_x+self.subimage_half_size_x]-self.sky_median
                         
    def gauss2D_fit_erf(self,p,fjac=None, x=None,y=None, z=None,err=None):
        '''
        Computes the residuals to be minimized by mpfit, given a model and data.
        '''
        model = Gaussian2D(p[0],p[1],p[2],p[3],p[4],np.radians(p[5]))(x,y)    
        status = 0
        return ([status, ((z-model)/err).ravel()])
                   
    def fit_gaussian(self,plot=True,verbose=False,save=None,**kwargs):
        """
        Perform a fit of a 2D gaussian. 
        Input:
            - plot: (optional) bool. If True, makes a plot of the image with 
            the contours of the gaussian
            - verbose: (optional) bool. If True, prints the verbose of mpdfit
            - additional optional keywords can be 'amp', 'centerx', 'centery', 
                'sigmax','sigmay','fwhm' or 'theta' to set the value of the 
                first guess of the fit. theta must be between 0 and 90
            - save: (optional) string with the name to save a pdf of the fit (only
                    valid if plot=True)
                    and a ds9 reg file (still to be implemented)
        Output:
            - fit_result: a dictionary with the parameters of the best fit. 
            The entries are 'AMP' 'X' 'FWHMX' 'Y' 'FWHMY' 'FWHM' 'THETA' 'ell' 
            - fit_error: a dictionary with the parameters of the error on the previous parameters (same entries)
            - chi2: value of the chi square
            - chi2_reduced: value of the reduced chi squared
        """

        # We first set a default guess              
        filtered_image = gaussian_filter(self.subimage,2)
#        if save is not None:
#            fits.writeto(save+'_initial.fits',self.subimage,clobber=True)
        argmax = np.nanargmax(filtered_image) 
        ymax,xmax = np.unravel_index(argmax,self.subimage.shape)
        amp= np.nanmax(self.subimage)
        centerx=xmax-self.subimage_half_size_x # the x center is in the range -subimage_half_size_x .. subimage_half_size_x
        centery=ymax-self.subimage_half_size_y # the y center is in the range -subimage_half_size_y .. subimage_half_size_y
        guess_dico = {'amp':amp,'centerx':centerx,'centery':centery,'sigx':2.,'sigy':2.,'theta':0.} 
        for k,v in kwargs.items():
            if k in guess_dico.keys():
                guess_dico[k]=v  
            elif k=='fwhm':
                guess_dico['sigx'] = fwhm2sig(v)
                guess_dico['sigy'] = fwhm2sig(v)
            elif k=='fwhmx':
                guess_dico['sigx'] = fwhm2sig(v)
            elif k=='fwhmy':
                guess_dico['sigy'] = fwhm2sig(v)
            else:
                raise TypeError('Keyword {0:s} not understood'.format(k))

        # We also set default boundaries
        parinfo =[{'fixed':0, 'limited':[1,1], 'limits':[0.,2*np.max([guess_dico['amp'],amp])]}, # Force the amplitude to be >0
                       {'fixed':0, 'limited':[1,1], 'limits':[-self.subimage_half_size_x+1,self.subimage_half_size_x-2]}, # We restrain the center to be 1px 
                       {'fixed':0, 'limited':[1,1], 'limits':[-self.subimage_half_size_y+1,self.subimage_half_size_y-2]}, # away from the edge
                       {'fixed':0, 'limited':[1,1], 'limits':[0.5,np.max([10.,guess_dico['sigx']])]}, # sigma_x between 0.5 and 10px
                       {'fixed':0, 'limited':[1,1], 'limits':[0.5,np.max([10.,guess_dico['sigy']])]}, # sigma_y between 0.5 and 10px
                       {'fixed':0, 'limited':[1,1], 'limits':[0,180.]}] # We limit theta beween 0 and 90 deg        

        x_vect = np.arange(-self.subimage_half_size_x,self.subimage_half_size_x)
        y_vect = np.arange(-self.subimage_half_size_y,self.subimage_half_size_y)
        x_array,y_array = np.meshgrid(x_vect,y_vect)
        fa = {'x': x_array, 'y': y_array, 'z':self.subimage, 'err':np.ones_like(self.subimage)*self.sky_rms}
        guess = [guess_dico['amp'],guess_dico['centerx'],guess_dico['centery'],guess_dico['sigx'],guess_dico['sigy'],guess_dico['theta']]                        
        m = mpfit.mpfit(self.gauss2D_fit_erf, guess, functkw=fa, parinfo=parinfo, quiet=(not verbose)*1)  
        if m.status == 0:
            print('Fit failed. Try to help the minimizer by providing a better first guess')
            if plot:
                plt.close(1)
                fig =  plt.figure(1, figsize=(4.5,3))
                gs = gridspec.GridSpec(1,2, height_ratios=[1], width_ratios=[1,0.06])
                gs.update(left=0.1, right=0.9, bottom=0.1, top=0.93, wspace=0.2, hspace=0.03)
                ax1 = plt.subplot(gs[0,0]) # Area for the first plot
                ax3 = plt.subplot(gs[0,1]) # Area for the second plot
                im = ax1.imshow(self.subimage,cmap='Greys',origin='lower', interpolation='nearest',extent=[self.guess_x-self.subimage_half_size_x,self.guess_x+self.subimage_half_size_x-1,self.guess_y-self.subimage_half_size_y,self.guess_y+self.subimage_half_size_y-1],vmin=np.nanmin(self.subimage),vmax=np.nanmax(self.subimage))
                ax1.set_xlabel('X in px')
                ax1.set_ylabel('Y in px')
                ax1.grid(True,c='w')
                ax1.text(0.95, 0.01, 'Fit failed',verticalalignment='bottom', horizontalalignment='right',\
                       transform=ax1.transAxes,color='red', fontsize=15)
                fig.colorbar(im, cax=ax3)
                if save is not None:
                    fig.savefig(save+'.pdf')
            null_dico = {'AMP':0,'X':0,'Y':0,'FWHMX':0,'FWHMY':0,'FWHM':0,'THETA':0,'ell':0}
            return null_dico,null_dico,0.,0.
        residuals = self.gauss2D_fit_erf(m.params,x=x_array,y=y_array,z=self.subimage,err=np.ones_like(self.subimage)*self.sky_rms)[1].reshape(self.subimage.shape)
        chi2 = np.sum(residuals**2)
        chi2_reduced = chi2 / m.dof
        sig = np.array([m.params[3],m.params[4]])
        sig_error=np.array([m.perror[3],m.perror[4]])
        error_ell = 4/(np.sum(sig)**2)*np.sqrt(np.sum((sig*sig_error)**2))
        fwhm = sig2fwhm(sig)
        fwhm_error = sig2fwhm(sig_error)
        fit_result = {'AMP':m.params[0],'X':m.params[1]+self.guess_x,'FWHMX':fwhm[0],\
                      'Y':m.params[2]+self.guess_y,'FWHMY':fwhm[1],'FWHM':np.mean(fwhm),'THETA':m.params[5],'ell':(sig[1]-sig[0])/np.mean(sig)}
        fit_error = { 'AMP':m.perror[0],'X':m.perror[1],'Y':m.perror[2],'FWHMX':fwhm_error[0],\
                     'FWHMY':fwhm_error[1],'FWHM':np.mean(fwhm_error),'THETA': m.perror[5],\
                      'ell':error_ell}
        print('X={0:4.2f}+/-{1:4.2f} Y={2:4.2f}+/-{3:4.2f} FWHM={4:3.2f}+/-{5:4.2f} ell={6:4.2f}+/-{7:4.2f}'.format(fit_result['X'],\
              fit_error['X'],fit_result['Y'],fit_error['Y'],fit_result['FWHM'],fit_error['FWHM'],fit_result['ell'],fit_error['ell'],))
        print('AMP={0:4.2e}+/-{1:3.2e} theta={2:3.1f}+/-{3:3.1f}deg SKY={4:4.2f}+/-{5:4.2f}'.format(fit_result['AMP'],\
              fit_error['AMP'],fit_result['THETA'],fit_error['THETA'],self.sky_median,self.sky_rms))
        print('DOF={0:d} CHI2={1:.1f} CHI2_r={2:.1f}'.format(m.dof,chi2,chi2_reduced))
        if plot:
            plt.close(1)
            fig =  plt.figure(1, figsize=(7.5,3))
            gs = gridspec.GridSpec(1,3, height_ratios=[1], width_ratios=[1,1,0.06])
            gs.update(left=0.1, right=0.9, bottom=0.1, top=0.93, wspace=0.2, hspace=0.03)
            ax1 = plt.subplot(gs[0,0]) # Area for the first plot
            ax2 = plt.subplot(gs[0,1]) # Area for the second plot
            ax3 = plt.subplot(gs[0,2]) # Area for the second plot
            im = ax1.imshow(self.subimage,cmap='CMRmap',origin='lower', interpolation='nearest',extent=[self.guess_x-self.subimage_half_size_x,self.guess_x+self.subimage_half_size_x-1,self.guess_y-self.subimage_half_size_y,self.guess_y+self.subimage_half_size_y-1],vmin=np.nanmin(self.subimage),vmax=np.nanmax(self.subimage))
            ax1.set_xlabel('X in px')
            ax1.set_ylabel('Y in px')
            ax1.contour(x_array+self.guess_x,y_array+self.guess_y,self.sky_median+Gaussian2D(m.params[0],m.params[1],m.params[2],m.params[3],m.params[4],np.radians(m.params[5]))(x_array,y_array),3,colors='w')
            ax1.grid(True,c='w')
            im2 = ax2.imshow(residuals,cmap='CMRmap',origin='lower', interpolation='nearest',extent=[self.guess_x-self.subimage_half_size_x,self.guess_x+self.subimage_half_size_x-1,self.guess_y-self.subimage_half_size_y,self.guess_y+self.subimage_half_size_y-1],vmin=np.nanmin(self.subimage),vmax=np.nanmax(self.subimage))#, extent=[np.min(x_array),np.max(x_array),np.min(y_array),np.max(y_array)])
            ax2.set_xlabel('X in px')
            ax2.grid(True,c='w')
            fig.colorbar(im, cax=ax3)
            if save is not None:
                fig.savefig(save+'.pdf')
        return fit_result,fit_error,chi2,chi2_reduced
        
if __name__ == '__main__':
#    import vip
#    ds9=vip.fits.vipDS9()
    cube = fits.getdata('/Users/jmilli/Documents/SPHERE/Sparta/2017-03-19/sparta_DTTS_cube_2017-03-19.fits')
    img1 = cube[6000,:,:]
    mask_sky = np.zeros_like(img1,dtype=bool)
    mask_sky[0:10,0:10]=True
#    ds9.display(img1)    
#    mask_sky = np.ones(img1.shape,dtype=bool)
#    shift_finder = ShiftFinder(img1,crop=10,guess_xy=[20,18])
    shift_finder = ShiftFinder(img1,sky=mask_sky)
    fit_result,fit_error,chi2,chi2_reduced = shift_finder.fit_gaussian(verbose=False,amp=100)
    
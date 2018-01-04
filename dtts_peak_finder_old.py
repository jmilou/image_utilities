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
import photutils

class Dtts_peak_finder_old():
    """ Object that analyses the DTTS images and finds the ones with a star, 
    and automatically detect if the LWE is there or not.
    
    Attributes:
        - image: the input image
    Methods:
        -
    """ 

    def __init__(self,cube,background='auto'):
        """
        Constructor of the class. 
        Input:
            - cube: the DTTS cube to analyze
            - background: 'auto' for automatic background detection and subtraction
                        or 'False' for no background subtraction
        Output:
            - nothing
        """            
        if not cube.ndim == 3 or cube.shape[1]!=32 or cube.shape[2]!=32:
            raise TypeError('The input is not a 32x32 cube.')
        
        self.nframes,self.ny,self.nx = cube.shape
        self.mask = np.zeros((32,32),dtype=bool)
        self.mask[7:-7,7:-7]=True
        self.mask[:,0]=True
        self.mask[0,:]=True
        px_scale = 11.5e-3 #in arcsec (from DTTS documentation: f/D=40.38 and pix=18micron)
        self.theoretical_fwhm = np.rad2deg(1.53e-6/8.)*3600/px_scale #(from DTTS documentation)
        self.theoretical_sig = fwhm2sig(self.theoretical_fwhm)

        x_vect = np.arange(0,self.nx)        
        y_vect = np.arange(0,self.ny)
        self.x_array,self.y_array = np.meshgrid(x_vect,y_vect)
        self.cube = np.copy(cube)
        self.residuals = np.zeros_like(cube)
        threshold_bck = 6. #threshold for the automatic background detection
        max_cube = np.max(cube,axis=(1,2))
        if background=='auto':
            nbck = np.sum(max_cube<threshold_bck)
            print('Automatic selection of {0:d} frames as backgrounds'.format(nbck))
        elif background=='False':
            nbck = 0
            print('Background selection was de-activated')
        else:
            print('Background subtraction method not understood: {0}. It should be auto or False'.format(background))

        if nbck>0:
            plt.figure(0)
            plt.semilogy(max_cube,label='star')
            plt.semilogy(np.arange(self.nframes)[max_cube<threshold_bck],max_cube[max_cube<threshold_bck],'or',label='no star')
            plt.legend(frameon=False)

            # we define the master background as the mean of all backgrounds
            bck_cube = self.cube[max_cube<threshold_bck,:,:]
            print('Max background value: {0:.1f} ADU'.format(np.max(bck_cube)))    
            self.bck = np.mean(bck_cube,axis=0)
        else:
            print('No background subtraction')
            self.bck = np.zeros((self.ny,self.nx))
        # Threshold for star detection. Anything below is discarded.
        threshold_star = 15.

        self.fit_result = {'AMP':np.ones(self.nframes)*np.nan,'X':np.ones(self.nframes)*np.nan,'FWHMX':np.ones(self.nframes)*np.nan,\
                      'Y':np.ones(self.nframes)*np.nan,'FWHMY':np.ones(self.nframes)*np.nan,'FWHM':np.ones(self.nframes)*np.nan,\
                      'THETA':np.ones(self.nframes)*np.nan,'ell':np.ones(self.nframes)*np.nan,'CHI2':np.ones(self.nframes)*np.nan,\
                      'CHI2_r':np.ones(self.nframes)*np.nan,'annulus_rms':np.ones(self.nframes)*np.nan,\
                      'circle_rms':np.ones(self.nframes)*np.nan,\
                      'T_rms':np.ones(self.nframes)*np.nan,'B_rms':np.ones(self.nframes)*np.nan,\
                      'L_rms':np.ones(self.nframes)*np.nan,'R_rms':np.ones(self.nframes)*np.nan,'asymmetry':np.ones(self.nframes)*np.nan}
        self.fit_error = { 'AMP':np.ones(self.nframes)*np.nan,'X':np.ones(self.nframes)*np.nan,'Y':np.ones(self.nframes)*np.nan,'FWHMX':np.ones(self.nframes)*np.nan,\
                     'FWHMY':np.ones(self.nframes)*np.nan,'FWHM':np.ones(self.nframes)*np.nan,'THETA': np.ones(self.nframes)*np.nan,\
                      'ell':np.ones(self.nframes)*np.nan}

        self.good_frames, = np.where(max_cube>=threshold_star)
                         
    def gauss2D_fit_erf(self,p,fjac=None, x=None,y=None, z=None,err=None):
        '''
        Computes the residuals to be minimized by mpfit, given a model and data.
        '''
        model = Gaussian2D(p[0],p[1],p[2],p[3],p[4],np.radians(p[5]))(x,y)    
        status = 0
        return ([status, ((z-model)/err).ravel()])
                   
    def fit_gaussian(self,plot=True,verbose=False,save=None):
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
        for i in self.good_frames:
            if verbose:
                print('Processing image {0:d}'.format(i))
            current_image = self.cube[i,:,:]-self.bck
            current_ma = np.ma.masked_array(current_image-self.bck,mask=self.mask)

            sky_med = np.median(current_ma)
            sky_rms = np.std(current_ma)
            if sky_med>5:
                print('Warning, the sky level is high: {0:5.1f} ADU'.format(sky_med))
            if sky_rms>5:
                print('Warning, the background noise is high: {0:5.1f} ADU'.format(sky_rms))

            # We first set a default guess              
            filtered_image = gaussian_filter(current_image,2)
            argmax = np.argmax(filtered_image) 
            ymax,xmax = np.unravel_index(argmax,current_image.shape)
            amp= np.max(current_image)
            guess_dico = {'amp':amp,'centerx':xmax,'centery':ymax,'sigx':self.theoretical_sig,'sigy':self.theoretical_sig,'theta':0.} 
            # We also set default boundaries
            parinfo =[{'fixed':0, 'limited':[1,1], 'limits':[0.,2*amp]}, # Force the amplitude to be >0
                      {'fixed':0, 'limited':[1,1], 'limits':[7,self.nx-7]}, # We restrain the center to be 1px 
                      {'fixed':0, 'limited':[1,1], 'limits':[7,self.ny-7]}, # away from the edge
                      {'fixed':0, 'limited':[1,1], 'limits':[self.theoretical_sig,1.4*self.theoretical_sig]}, # sigma_x between 0.5 and 10px
                      {'fixed':0, 'limited':[1,1], 'limits':[self.theoretical_sig,1.4*self.theoretical_sig]}, # sigma_y between 0.5 and 10px
                      {'fixed':0, 'limited':[1,1], 'limits':[0,180.]}] # We limit theta beween 0 and 90 deg        
    
            fa = {'x': self.x_array, 'y': self.y_array, 'z':current_image, 'err':np.ones_like(current_image)*sky_rms}
            guess = [guess_dico['amp'],guess_dico['centerx'],guess_dico['centery'],guess_dico['sigx'],guess_dico['sigy'],guess_dico['theta']]                        
            m = mpfit.mpfit(self.gauss2D_fit_erf, guess, functkw=fa, parinfo=parinfo,quiet=1)# quiet=(not verbose)*1)  
            if m.status == 0:
                print('Fit failed for frame {0:d}. Try to help the minimizer by providing a better first guess'.format(i))
            else:
                residuals = self.gauss2D_fit_erf(m.params,x=self.x_array,y=self.y_array,z=current_image,err=np.ones_like(current_image)*sky_rms)[1].reshape(current_image.shape)
                self.residuals[i,:,:] = residuals #+self.bck #+sky_med
                self.fit_result['CHI2'][i] = np.sum(residuals**2)
                self.fit_result['CHI2_r'][i] = self.fit_result['CHI2'][i] / m.dof
                self.fit_result['AMP'][i] = m.params[0]
                self.fit_result['X'][i] = m.params[1]
                self.fit_result['Y'][i] = m.params[2]
                sig = np.array([m.params[3],m.params[4]])
                sig_error=np.array([m.perror[3],m.perror[4]])
                error_ell = 4/(np.sum(sig)**2)*np.sqrt(np.sum((sig*sig_error)**2))
                fwhm = sig2fwhm(sig)
                fwhm_error = sig2fwhm(sig_error)
                self.fit_result['FWHMX'][i] = fwhm[0]
                self.fit_result['FWHMY'][i] = fwhm[1]
                self.fit_result['FWHM'][i] = np.mean(fwhm)
                self.fit_result['THETA'][i] = m.params[5]
                self.fit_result['ell'][i] = (sig[1]-sig[0])/np.mean(sig)
                
                self.fit_error['AMP'][i] = m.perror[0]
                self.fit_error['X'][i] = m.perror[1]
                self.fit_error['Y'][i] = m.perror[2]
                self.fit_error['FWHMX'][i] = fwhm_error[0]
                self.fit_error['FWHMY'][i] = fwhm_error[1]
                self.fit_error['FWHM'][i] = np.mean(fwhm_error)
                self.fit_error['THETA'][i] = m.perror[5]
                self.fit_error['ell'][i] = error_ell

                annulus_aperture = photutils.CircularAnnulus((self.fit_result['X'][i],self.fit_result['Y'][i]), \
                                                     r_in=self.theoretical_fwhm,\
                                                     r_out=2*self.theoretical_fwhm)
                phot_table_annulus = photutils.aperture_photometry(residuals**2, \
                                    annulus_aperture,error=np.ones_like(current_image)*sky_rms)
                self.fit_result['annulus_rms'][i] = np.sqrt(phot_table_annulus['aperture_sum'][0])/phot_table_annulus['aperture_sum_err'][0]

                circular_aperture = photutils.CircularAperture((self.fit_result['X'][i],self.fit_result['Y'][i]), \
                                                     r=2.5*self.theoretical_fwhm)
                phot_table_circle = photutils.aperture_photometry(residuals**2, \
                                    circular_aperture,error=np.ones_like(current_image)*sky_rms)
                self.fit_result['circle_rms'][i] = np.sqrt(phot_table_circle['aperture_sum'][0])/phot_table_circle['aperture_sum_err'][0]

                mask_aperture = np.zeros_like(current_image,dtype=bool)
                mask_R = np.copy(mask_aperture) #right
                mask_R[self.x_array>self.fit_result['X'][i]]=True
                mask_L = np.copy(mask_aperture) #left
                mask_L[self.x_array<self.fit_result['X'][i]]=True
                mask_T = np.copy(mask_aperture) #top 
                mask_T[self.y_array>self.fit_result['Y'][i]]=True
                mask_B = np.copy(mask_aperture) # bottom
                mask_B[self.y_array<self.fit_result['Y'][i]]=True
                phot_table_T = photutils.aperture_photometry(residuals**2, annulus_aperture,\
                                error=np.ones_like(current_image)*sky_rms,mask=mask_T)
                self.fit_result['T_rms'][i] = np.sqrt(phot_table_T['aperture_sum'][0])/phot_table_T['aperture_sum_err'][0]
                phot_table_B = photutils.aperture_photometry(residuals**2, annulus_aperture,\
                                error=np.ones_like(current_image)*sky_rms,mask=mask_B)
                self.fit_result['B_rms'][i] = np.sqrt(phot_table_B['aperture_sum'][0])/phot_table_B['aperture_sum_err'][0]
                phot_table_L = photutils.aperture_photometry(residuals**2, annulus_aperture,\
                                error=np.ones_like(current_image)*sky_rms,mask=mask_L)
                self.fit_result['L_rms'][i] = np.sqrt(phot_table_L['aperture_sum'][0])/phot_table_L['aperture_sum_err'][0]
                phot_table_R = photutils.aperture_photometry(residuals**2, annulus_aperture,\
                                error=np.ones_like(current_image)*sky_rms,mask=mask_R)
                self.fit_result['R_rms'][i] = np.sqrt(phot_table_R['aperture_sum'][0])/phot_table_R['aperture_sum_err'][0]
                self.fit_result['asymmetry'][i] = np.max([np.abs(self.fit_result['T_rms'][i]-self.fit_result['B_rms'][i]),np.abs(self.fit_result['L_rms'][i]-self.fit_result['T_rms'][i])])

                if verbose:
                    print('X={0:4.2f}+/-{1:4.2f} Y={2:4.2f}+/-{3:4.2f} FWHM={4:3.2f}+/-{5:4.2f} ell={6:4.2f}+/-{7:4.2f}'.format(self.fit_result['X'][i],\
                          self.fit_error['X'][i],self.fit_result['Y'][i],self.fit_error['Y'][i],self.fit_result['FWHM'][i],self.fit_error['FWHM'][i],self.fit_result['ell'][i],self.fit_error['ell'][i],))
                    print('AMP={0:4.2e}+/-{1:3.2e} theta={2:3.1f}+/-{3:3.1f}deg SKY={4:4.2f}+/-{5:4.2f}'.format(self.fit_result['AMP'][i],\
                          self.fit_error['AMP'][i],self.fit_result['THETA'][i],self.fit_error['THETA'][i],sky_med,sky_rms))
                    print('DOF={0:d} CHI2={1:.1f} CHI2_r={2:.1f}'.format(m.dof,self.fit_result['CHI2'][i],self.fit_result['CHI2_r'][i]))
                if plot:
                    plt.close(1)
                    fig =  plt.figure(1, figsize=(7.5,3))
                    gs = gridspec.GridSpec(1,3, height_ratios=[1], width_ratios=[1,1,0.06])
                    gs.update(left=0.1, right=0.9, bottom=0.1, top=0.93, wspace=0.2, hspace=0.03)
                    ax1 = plt.subplot(gs[0,0]) # Area for the first plot
                    ax2 = plt.subplot(gs[0,1]) # Area for the second plot
                    ax3 = plt.subplot(gs[0,2]) # Area for the second plot
                    im = ax1.imshow(current_image,cmap='CMRmap',origin='lower', interpolation='nearest',extent=[np.min(self.x_array),np.max(self.x_array),np.min(self.y_array),np.max(self.y_array)],vmin=np.nanmin(current_ma),vmax=np.nanmax(current_ma))
                    ax1.set_xlabel('X in px')
                    ax1.set_ylabel('Y in px')
                    ax1.contour(self.x_array,self.y_array,sky_med+Gaussian2D(m.params[0],m.params[1],m.params[2],m.params[3],m.params[4],np.radians(m.params[5]))(self.x_array,self.y_array),3,colors='w')
                    ax1.grid(True,c='w')
                    im2 = ax2.imshow(residuals,cmap='CMRmap',origin='lower', interpolation='nearest',extent=[np.min(self.x_array),np.max(self.x_array),np.min(self.y_array),np.max(self.y_array)],vmin=np.nanmin(current_image),vmax=np.nanmax(current_ma))#, extent=[np.min(self.x_array),np.max(self.x_array),np.min(self.y_array),np.max(y_array)])
                    ax2.set_xlabel('X in px')
                    ax2.grid(True,c='w')
                    fig.colorbar(im, cax=ax3)
                    if save is not None:
                        fig.savefig(save+'_{0:d}.pdf'.format(i))
            plt.figure(1)
            plt.clf()
            plt.plot(self.fit_result['CHI2_r'],label='chi2 r')
            plt.plot(self.fit_result['annulus_rms'],label='annulus')
            plt.plot(self.fit_result['asymmetry'],label='asymmetry')
            plt.plot(self.fit_result['circle_rms'],label='circle')
            plt.legend(frameon=False,loc='best')
        return 
        
if __name__ == '__main__':
#    import vip
#    ds9=vip.fits.vipDS9()
    print('OK')
#    cube = fits.getdata('/Users/jmilli/Documents/SPHERE/Sparta/2017-03-19/sparta_DTTS_cube_2017-03-19.fits')
#    cube = cube[5700:6100,:,:]
#    DTTS_peak_finder = Dtts_peak_finder(cube)
#    DTTS_peak_finder.fit_gaussian(verbose=False,plot=False)

#    plt.plot(DTTS_peak_finder.fit_result['CHI2_r'][DTTS_peak_finder.good_frames])

#    plt.plot(DTTS_peak_finder.fit_result['asymmetry'][DTTS_peak_finder.good_frames],label=)

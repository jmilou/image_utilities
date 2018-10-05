#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 15:26:53 2017

@author: jmilli
"""
import numpy as np
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import stats
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt
#from vip.phot import noise_per_annulus
#from vip.conf import time_ini, timing, sep
#from vip.var import frame_center, dist # necessary for the function stat_per_annulus
import photutils # necessary for the function stat_per_annulus

def contrast_curve_from_throughput(image, fwhm, pxscale, starphot,throughput=None,
                   sigma=5, inner_rad=1, wedge=(0,360),
                   student=True, transmission=None, smooth=True, plot=True,
                   dpi=100, debug=False, verbose=True, rawContrast=False,**algo_dict):
    """ Computes the contrast curve for a given SIGMA (*sigma*) level. The 
    contrast is calculated as sigma*noise/throughput. This implementation takes
    into account the small sample statistics correction proposed in Mawet et al.
    2014. 
    
    Parameters
    ----------
    image : array_like
        The reduced image.
    fwhm : float
        FWHM in pixels.
    pxscale : float
        Plate scale or pixel scale of the instrument. 
    starphot : int or float or 1d array
        If int or float it corresponds to the aperture photometry of the 
        non-coronagraphic PSF which we use to scale the contrast. If a vector 
        is given it must contain the photometry correction for each frame.
    throughput: tuple of 2 1d arrays, optional
        If not None, then the tuple contains a vector with the the radial distances [px] 
        and with the factors to be applied to the sensitivity (in this order). 
    sigma : int
        Sigma level for contrast calculation.
    inner_rad : int, optional
        Innermost radial distance to be considered in terms of FWHM.
    wedge : tuple of floats, optional
        Initial and Final angles for using a wedge. For example (-90,90) only
        considers the right side of an image.
    student : {True, False}, bool optional
        If True uses Student t correction to inject fake companion. 
    transmission : tuple of 2 1d arrays, optional
        If not None, then the tuple contains a vector with the factors to be 
        applied to the sensitivity and a vector of the radial distances [px] 
        where it is sampled (in this order). 
    smooth : {True, False}, bool optional
        If True the radial noise curve is smoothed with a Savitzky-Golay filter
        of order 2. 
    plot : {True, False}, bool optional 
        Whether to plot the final contrast curve or not. True by default.
    dpi : int optional 
        Dots per inch for the plots. 100 by default. 300 for printing quality.
    debug : {False, True}, bool optional
        Whether to print and plot additional info such as the noise, throughput,
        the contrast curve with different X axis and the delta magnitude instead
        of contrast.
    verbose : {True, False, 0, 1, 2} optional
        If True or 1 the function prints to stdout intermediate info and timing,
        if set to 2 more output will be shown. 
    rawContrast: {False, True}, bool optional by default False
        If True, then the profile is the average median profile. If False then 
        it is the std
    **algo_dict
        Any other valid parameter of the post-processing algorithms can be 
        passed here.
    
    Returns
    -------
    datafr : pandas dataframe
        Dataframe containing the sensitivity (Gaussian and Student corrected if
        Student parameter is True), the interpolated throughput, the distance in 
        pixels, the noise and the sigma corrected (if Student is True). 
    """  
    if not image.ndim == 2:
        raise TypeError('The input array is not an image')
    if transmission is not None:
        if not isinstance(transmission, tuple) or not len(transmission)==2:
            raise TypeError('transmission must be a tuple with 2 1d vectors')
    if isinstance(starphot, float) or isinstance(starphot, int):  pass
    else:
        if not starphot.shape[0] == image.shape[0]:
            raise TypeError('Correction vector has bad size')
        image = image.copy()
        for i in range(image.shape[0]):
            image[i] = image[i] / starphot[i]

    if verbose:
        if isinstance(starphot, float) or isinstance(starphot, int):
            msg0 = 'FWHM = {}, SIGMA = {},'
            msg0 += ' STARPHOT = {}'
            print(msg0.format(fwhm, sigma, starphot))
        else:
            msg0 = 'FWHM = {}, SIGMA = {}'
            print(msg0.format(fwhm, sigma))

    # throughput

    if throughput is not None:
        if isinstance(throughput,float):
            if verbose:
                print('Scalar throughput provided. Asssuming constant value for all separations')
            if throughput>1. or throughput<0.:
                raise ValueError('The throughput must be a float between 0. and 1.')
            vector_radd = np.arange(fwhm,image.shape[0]/2,fwhm)
            thruput_mean = np.ones_like(vector_radd)*throughput
        elif not isinstance(throughput, tuple) or len(throughput) != 2:
            raise TypeError('The throughput must be a 2-element tuple')
        elif throughput[0].shape[0] != throughput[1].shape[0]:
            raise TypeError('The throughput arrays must have the same lengths')
        else:
            vector_radd = throughput[0]
            thruput_mean = throughput[1]
    else:
        if verbose:
            print('No throughput  provided. Assuming 100%.')
        vector_radd = np.arange(fwhm,image.shape[0]/2,fwhm)
        thruput_mean = np.ones_like(vector_radd)
        
    # noise measured in the image, every px starting from 1*FWHM
    dico_noise = stat_per_annulus(image, separation=1, fwhm=fwhm, init_rad=fwhm,\
                                  wedge=wedge,verbose=False, debug=False)
    rad_samp = dico_noise['radius']
    noise_samp = dico_noise['std']
    if rawContrast:
        noise_samp = dico_noise['median']
    else:
        noise_samp = dico_noise['std']
#    noise_samp, rad_samp = noise_per_annulus(image, separation=1, fwhm=fwhm,
#                                             init_rad=fwhm, wedge=wedge)
    cutin1 = np.where(rad_samp.astype(int)==vector_radd.astype(int).min())[0][0]
    noise_samp = noise_samp[cutin1:]
    rad_samp = rad_samp[cutin1:]
    cutin2 = np.where(rad_samp.astype(int)==vector_radd.astype(int).max())[0][0]
    noise_samp = noise_samp[:cutin2+1]
    rad_samp = rad_samp[:cutin2+1]
        
    # interpolating the throughput vector, spline order 2
    f = InterpolatedUnivariateSpline(vector_radd, thruput_mean, k=2)
    thruput_interp = f(rad_samp)   
    
    # interpolating the transmission vector, spline order 1  
    if transmission is not None:
        trans = transmission[0]
        radvec_trans = transmission[1]     
        f2 = InterpolatedUnivariateSpline(radvec_trans, trans, k=1)
        trans_interp = f2(rad_samp)
        thruput_interp *= trans_interp

    if smooth:
        # smoothing the noise vector using a Savitzky-Golay filter
        win = int(noise_samp.shape[0]*0.1)
        if win%2==0.:  win += 1
        noise_samp_sm = savgol_filter(noise_samp, polyorder=2, mode='nearest',
                                      window_length=win)
    else:
        noise_samp_sm = noise_samp
    
    if debug:
        plt.rc("savefig", dpi=dpi)
        plt.figure(figsize=(8,4))
        plt.plot(vector_radd*pxscale, thruput_mean, '.', label='computed', 
                 alpha=0.6)
        plt.plot(rad_samp*pxscale, thruput_interp, ',-', label='interpolated', 
                 lw=2, alpha=0.5)
        plt.grid('on', which='both', alpha=0.2, linestyle='solid')
        plt.xlabel('Angular separation [arcsec]')
        plt.ylabel('Throughput')
        plt.legend(loc='best')
        plt.xlim(0, np.max(rad_samp*pxscale))
        
        plt.figure(figsize=(8,4))
        plt.plot(rad_samp*pxscale, noise_samp, '.', label='computed', alpha=0.6)
        plt.plot(rad_samp*pxscale, noise_samp_sm, ',-', label='noise smoothed', 
                 lw=2, alpha=0.5)
        plt.grid('on', alpha=0.2, linestyle='solid')
        plt.xlabel('Angular separation [arcsec]')
        plt.ylabel('Noise')
        plt.legend(loc='best')
        #plt.yscale('log')
        plt.xlim(0, np.max(rad_samp*pxscale))
    
    # calculating the contrast
    if isinstance(starphot, float) or isinstance(starphot, int):
        cont_curve_samp = ((sigma * noise_samp_sm)/thruput_interp)/starphot
    else:
        cont_curve_samp = ((sigma * noise_samp_sm)/thruput_interp)
    cont_curve_samp[np.where(cont_curve_samp<0)] = 1
    cont_curve_samp[np.where(cont_curve_samp>1)] = 1
        
    # calculating the Student corrected contrast
    if student:
        n_res_els = np.floor(rad_samp/fwhm*2*np.pi)
        ss_corr = np.sqrt(1 + 1/(n_res_els-1))
        sigma_corr = stats.t.ppf(stats.norm.cdf(sigma), n_res_els)*ss_corr
        if isinstance(starphot, float) or isinstance(starphot, int):
            cont_curve_samp_corr = ((sigma_corr * noise_samp_sm)/thruput_interp)/starphot
        else:    
            cont_curve_samp_corr = ((sigma_corr * noise_samp_sm)/thruput_interp)
        cont_curve_samp_corr[np.where(cont_curve_samp_corr<0)] = 1
        cont_curve_samp_corr[np.where(cont_curve_samp_corr>1)] = 1

    # plotting
    if plot or debug:
        if student:  
            label = ['Sensitivity (Gaussian)', 
                     'Sensitivity (Student-t correction)']
        else:  label = ['Sensitivity (Gaussian)']
        
        plt.rc("savefig", dpi=dpi)
        fig = plt.figure(figsize=(8,4))
        ax1 = fig.add_subplot(111)
        con1, = ax1.plot(rad_samp*pxscale, cont_curve_samp, '-', 
                         alpha=0.2, lw=2, color='green')
        con2, = ax1.plot(rad_samp*pxscale, cont_curve_samp, '.',
                         alpha=0.2, color='green')
        if student:
            con3, = ax1.plot(rad_samp*pxscale, cont_curve_samp_corr, '-', 
                             alpha=0.4, lw=2, color='blue')
            con4, = ax1.plot(rad_samp*pxscale, cont_curve_samp_corr, '.',
                             alpha=0.4, color='blue')
            lege = [(con1, con2), (con3, con4)]
        else:
            lege = [(con1, con2)]
        plt.legend(lege, label, fancybox=True, fontsize='medium')
        plt.xlabel('Angular separation [arcsec]')
        plt.ylabel(str(sigma)+' sigma contrast')
        plt.grid('on', which='both', alpha=0.2, linestyle='solid')
        ax1.set_yscale('log')
        ax1.set_xlim(0, np.max(rad_samp*pxscale))

        if debug:        
            fig2 = plt.figure(figsize=(8,4))
            ax3 = fig2.add_subplot(111)
            cc_mags = -2.5*np.log10(cont_curve_samp)
            con4, = ax3.plot(rad_samp*pxscale, cc_mags, '-', 
                             alpha=0.2, lw=2, color='green')
            con5, = ax3.plot(rad_samp*pxscale, cc_mags, '.', alpha=0.2,
                             color='green')
            if student:
                cc_mags_corr = -2.5*np.log10(cont_curve_samp_corr)
                con6, = ax3.plot(rad_samp*pxscale, cc_mags_corr, '-', 
                                 alpha=0.4, lw=2, color='blue')
                con7, = ax3.plot(rad_samp*pxscale, cc_mags_corr, '.', 
                                 alpha=0.4, color='blue')
                lege = [(con4, con5), (con6, con7)]
            else:
                lege = [(con4, con5)]
            plt.legend(lege, label, fancybox=True, fontsize='medium')
            plt.xlabel('Angular separation [arcsec]')
            plt.ylabel('Delta magnitude')
            plt.gca().invert_yaxis()
            plt.grid('on', which='both', alpha=0.2, linestyle='solid')
            ax3.set_xlim(0, np.max(rad_samp*pxscale))
            ax4 = ax3.twiny()
            ax4.set_xlabel('Distance [pixels]')
            ax4.plot(rad_samp, cc_mags, '', alpha=0.)
            ax4.set_xlim(0, np.max(rad_samp)) 

    if student:
        datafr = pd.DataFrame({'sensitivity (Gauss)': cont_curve_samp,
                               'sensitivity (Student)':cont_curve_samp_corr,
                               'throughput': thruput_interp,
                               'distance': rad_samp, 'noise': noise_samp_sm,
                               'sigma corr':sigma_corr})
    else:
        datafr = pd.DataFrame({'sensitivity (Gauss)': cont_curve_samp,
                               'throughput': thruput_interp,
                               'distance': rad_samp, 'noise': noise_samp_sm})
    if verbose:
        print('Finished the noise calculation')
    return datafr


def stat_per_annulus(array, separation, fwhm, init_rad=None, wedge=(0,360),
                      verbose=False, debug=False):
    """ Measures some statistics of an image such as the median, mean and 
    standard deviation of apertures defined in
    each annulus with a given separation.

    Parameters
    ----------
    array : array_like
        Input frame.
    separation : float
        Separation in pixels of the centers of the annuli measured from the
        center of the frame.
    fwhm : float
        FWHM in pixels.
    init_rad : float
        Initial radial distance to be used. If None then the init_rad = FWHM.
    wedge : tuple of floats, optional
        Initial and Final angles for using a wedge. For example (-90,90) only
        considers the right side of an image. Be careful when using small
        wedges, this leads to computing a standard deviation of very small
        samples (<10 values).
    verbose : {False, True}, bool optional
        If True prints information.
    debug : {False, True}, bool optional
        If True plots the positioning of the apertures.

    Returns
    -------
    a dictionnary with keys:
        std : array_like, Vector with the standard deviation value per annulus.
        mean : array_like, Vector with the median value per annulus.
        median : array_like, Vector with the mean value per annulus.
        radius : array_like, Vector with the radial distances values.

    """
    def find_coords(rad, sep, init_angle, fin_angle):
        angular_range = fin_angle-init_angle
        npoints = (np.deg2rad(angular_range)*rad)/sep   #(2*np.pi*rad)/sep
        ang_step = angular_range/npoints   #360/npoints
        x = []
        y = []
        for i in range(int(npoints)):
            newx = rad * np.cos(np.deg2rad(ang_step * i + init_angle))
            newy = rad * np.sin(np.deg2rad(ang_step * i + init_angle))
            x.append(newx)
            y.append(newy)
        return np.array(y), np.array(x)
    #___________________________________________________________________

    if not array.ndim==2:
        raise TypeError('Input array is not a frame or 2d array')
    if not isinstance(wedge, tuple):
        raise TypeError('Wedge must be a tuple with the initial and final angles')

    init_angle, fin_angle = wedge

    centery, centerx = array.shape[1]//2 , array.shape[0]//2
    n_annuli = int(np.floor((centery)/separation))

    x = centerx
    y = centery
    vector_std = []
    vector_med = []
    vector_mean = []
    vector_radd = []
    vector_nresel = []
    if verbose:  print('{} annuli'.format(n_annuli-1))

    if init_rad is None:  init_rad = fwhm

    if debug:
        _, ax = plt.subplots(figsize=(6,6))
        ax.imshow(array, origin='lower', interpolation='nearest',
                  alpha=0.5, cmap='gray')

    for i in range(n_annuli-1):
        y = centery + init_rad + separation*(i)
        rad = np.sqrt((centery-y)**2+(centerx-x)**2)
        yy, xx = find_coords(rad, fwhm, init_angle, fin_angle)
        yy += centery
        xx += centerx

        apertures = photutils.CircularAperture((xx, yy), fwhm/2.)
        fluxes = photutils.aperture_photometry(array, apertures)
        fluxes = np.array(fluxes['aperture_sum'])

        std_ann = np.std(fluxes)
        med_ann = np.median(fluxes)
        mean_ann = np.mean(fluxes)
        vector_std.append(std_ann)
        vector_radd.append(rad)
        vector_med.append(med_ann)
        vector_mean.append(mean_ann)
        vector_nresel.append(len(fluxes))
        
        if debug:
            for i in range(xx.shape[0]):
                # Circle takes coordinates as (X,Y)
                aper = plt.Circle((xx[i], yy[i]), radius=fwhm/2., color='r',
                              fill=False, alpha=0.8)
                ax.add_patch(aper)
                cent = plt.Circle((xx[i], yy[i]), radius=0.8, color='r',
                              fill=True, alpha=0.5)
                ax.add_patch(cent)

        if verbose:
            print('Radius(px) = {0:.0f}, mean = {1:3.1e}, med = {2:3.1e}, std = {3:3.1e} '.format(rad, mean_ann, med_ann, std_ann))

    return {'std':np.array(vector_std), 'radius':np.array(vector_radd),\
            'mean':np.array(vector_mean),'median':np.array(vector_med),\
            'nresels':vector_nresel}


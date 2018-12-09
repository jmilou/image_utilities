#!/usr/bin/env python
#
# Original filename: moffat_centroid.py
#

from scipy import optimize
from astropy.io import fits
import numpy as np

def moffat(dict_param, x, y):
    tmp = ((x - dict_param['x0'])**2 +  (y - dict_param['y0'])**2) / dict_param['alpha']**2
    if 'offset' in dict_param.keys():
        return dict_param['offset'] + dict_param['I0'] / (1 + tmp)**dict_param['beta']
    else:
        return dict_param['I0'] / (1 + tmp)**dict_param['beta']
        
def moffat_saturated(dict_param, x, y,satlevel):
    return np.clip( moffat(dict_param,x,y), -100, satlevel)

def errfunc(list_param, x, y, f, satlevel):
    # we have to convert p to a dict
    dict_param = {'x0':list_param[0],'y0':list_param[1],'alpha':list_param[2],'beta':list_param[3],'I0':list_param[4]}
    if len(list_param) == 6: # if we want to fit an offset
        dict_param['offset'] = list_param[5]
    return moffat_saturated(dict_param, x, y, satlevel) - np.clip(f,-100,satlevel) 
    
def moffat_centroid_naco(frame, img=None, lastcen=None,satlevel=None,box=None, verbose=False, no_offset=False):

    """

    Function moffat_centroid_naco fits a Moffat profile to an image.  It
    returns the centroid of the best-fit profile as a list [y, x].
    Arguments:

    1.  The original filename.  If no img array is supplied, it will
        be read from this file.

    Optional arguments:
    2.  A 2D array to centroid.
    3.  The first guess for the centroid.  Default is the center of
        the image.
    4. The saturation level (default is 10000)
    5. The size of the box for centering (default 50)
    6. verbose = True if you want to check the parameters
    """
    
    if np.any(img) == None:
        hdulist_psf = fits.open(frame)
        img = hdulist_psf[0].data
        hdulist_psf.close()

    if lastcen is None:
        lastcen = [img.shape[0] // 2, img.shape[1] // 2]
    lastcen[0] = int(lastcen[0])
    lastcen[1] = int(lastcen[1])

    if satlevel is None:
        satlevel = 10000.
        # satlevel = stats.scoreatpercentile(img2, 99)

    if box is None:
        box = np.min([lastcen[0],lastcen[1],img.shape[0],img.shape[1],
                      img.shape[0]-lastcen[0],img.shape[1]-lastcen[1],51]) 
        # the default box size is 51px

    if np.mod(box, 2) == 0: # we want box to be odd
        box -= 1
        if verbose :
            print('[Warning] The box size was decreased to {0:3d} px'.format(box))
 
    img_extracted = np.ndarray((box, box))
    img_extracted[:, :] = img[lastcen[1] - box // 2 : lastcen[1] + box // 2 + 1,
                       lastcen[0] - box // 2 : lastcen[0] + box // 2 + 1]
                       
    x = np.linspace(0, box - 1., box) - box // 2
    y = np.linspace(0, box - 1., box) - box // 2
    y,x = np.meshgrid(x, y)
    img_extracted = np.reshape(img_extracted, (-1))
    x = np.reshape(x, (-1))
    y = np.reshape(y, (-1))
    
    # Now we select only unsaturated pixels
    x = x.compress(img_extracted < satlevel)
    y = y.compress(img_extracted < satlevel)
    img_extracted = img_extracted.compress(img_extracted < satlevel)
    # img_extracted = filter(lambda t : t<satlevel , img_extracted)

    FWHM0 = 2.5 #px
    beta0 = 2.0
    alpha0 = FWHM0 / (2 * np.sqrt(2.0**(1.0/beta0) -1) )
    x0 = 0.
    y0 = 0.
    I0 = 1.e5
    p0 = [x0,y0,alpha0,beta0,I0]
    if no_offset == False : 
        p0.append(0.)
    # p0={'alpha':alpha0,'x0':0,'y0':0,'I0':1.e5,'beta':beta0}
    # if no_offset == False :
    #     p0['offset'] = 0.
    p1, success = optimize.leastsq(errfunc, p0,args=(x, y, img_extracted, satlevel))
    dict_p1 = {}
    dict_p1['x0'] = p1[0] + lastcen[0]
    dict_p1['y0'] = p1[1] + lastcen[1]
    dict_p1['alpha'] = p1[2]
    dict_p1['beta'] = p1[3]
    dict_p1['I0'] = p1[4]
    dict_p1['FWHM'] = dict_p1['alpha'] * 2 * np.sqrt(2.0**(1.0/dict_p1['beta']) -1 )
    if no_offset == False : 
        dict_p1['offset'] = p1[5]
    else:
        dict_p1['offset'] = 0
    if verbose :
        print('FWHM = {0:6.2f} px'.format(dict_p1['FWHM']))
        print('alpha = {0:6.2f}'.format(dict_p1['alpha']))
        print('beta = {0:6.2f}'.format(dict_p1['beta']))
        print('Center = [{0:6.2f} , {1:6.2f}]'.format(dict_p1['x0'],dict_p1['y0']))
        print('Maximum = {0:6.3e}'.format(dict_p1['I0']))
        print('Offset = {0:6.2f}'.format(dict_p1['offset']))
    return dict_p1


    # -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 11:42:57 2015

@author: jmilli
"""
import sys
from sympy import Symbol, nsolve
import math
import numpy as np
import matplotlib.pyplot as plt
#from scipy import ndimage
sys.path.append('/Users/jmilli/Dropbox/lib_py/image_utilities')
import rotation_images as rot
from numpy.linalg import eig, inv
import matplotlib.pylab as plt

def ellipse_points(a,b,R,precision=0.2*math.pi/180, step_R=0.1,plot=True):

    """

    Function ellipse_points finds points on an ellipse equaly spaced
    Arguments:
    1. a: semi-major axis of the ellipse
    2. b: semi-minor axis of the ellipse
    3. R: spacing between points
    Optional arguments:
    4. precision: the precision in radians on spacing
    5. step_R: the step in spacing between each iteration
    """
    x = Symbol('x')
    y = Symbol('y')
    ellipse = (x/a)**2 + (y/b)**2 - 1
    t_final=math.pi/2
    iter_nb=0    
    continue_loop = True
    while continue_loop:
        iter_nb += 1
        if iter_nb > 1:
            print('Iterations: {0:.0f}, deviation at final position {1:4.2f} degrees'.format(iter_nb-1,(t-t_final)*180/math.pi))
        t=0 #math.pi/10
        x_sol=[a*math.cos(t)]
        y_sol=[b*math.sin(t)]
        t_sol=[t]
        while t < t_final-precision:
            x0 = a*math.cos(t)
            y0 = b*math.sin(t)
            cercle = (x-x0)**2 + (y-y0)**2 -R**2
            trynextguess=True
            nbguessiter=0
            while (trynextguess and nbguessiter < 10):
                try:
                    derivative= [-a*math.sin(t),b*math.cos(t)]
                    direction = R/np.linalg.norm(derivative)*np.array(derivative)
                    guess = np.array([x0,y0])+direction
                    sol=nsolve((ellipse,cercle), (x, y), (guess[0],guess[1]))
                    trynextguess=False
                except ValueError as e:
                    nbguessiter += 1
                    print(e)
                    print('Initial guess changed. We retry: {0:4.0f} iterations'.format(
                    nbguessiter))
                    t+=math.atan(R/4/a)
            #print(sol)            
            t =  math.acos(float(sol[0])/a)
            t_sol.append(t)
            x_sol.append(a*math.cos(t))
            y_sol.append(b*math.sin(t))
        if math.fabs(t-t_final) < precision: 
            continue_loop = False
        else:
            R-=step_R
    print('Number of iterations: {0:4.0f}'.format(iter_nb))
    print('Deviation in degrees at final position = {0:4.2f}'.format(
        (t-t_final)*180/math.pi))
    print('Spacing between points = {0:4.2f}'.format(R))

    if plot:   
        nb_points = 100
        theta = np.arange(0,math.pi/2,math.pi/2/nb_points)
        x_ellipse = np.array([a*math.cos(i) for i in theta])
        y_ellipse = np.array([b*math.sin(i) for i in theta])
        plt.plot(x_sol,y_sol, 'ro')
        plt.plot(x_ellipse,y_ellipse)
        plt.plot([0,a],[0,0])
        plt.plot([0,0],[0,b])
        plt.axis([0,a, 0, b])
        plt.axis('equal')   # ajout
        plt.show()

    return t_sol

   
  
def elliptical_mask(size,a,b,epsilon=2.,delta=2.,yc=None,xc=None,theta=0):

    """
    Function ellitical_mask builds an elliptical mask. Two ellipses of semi major 
    axis a-epsilon and a+espislon and of semi-minor axis b-delta and b+delta are built.
    The mask is 0 everywhere outside the 2 ellipses and 1 within the 2 ellipses.
    Arguments:
        1. a: semi-major axis of the ellipse
        2. b: semi-minor axis of the ellipse
    Optional arguments:
        4. epsilon: 2*epsilon+1 is the difference between the inner and outer ellipse. 
        By default it is 2px
        5. delta: 2*epsilon+1 is the difference between the inner and outer ellipse. 
        By default it is 2px
        6.yc: the center of the ellipse in y. By default, size/2
        7.xc: the center of the ellipse in x. By default, size/2
        8. theta: the position angle of the semi-major axis of the ellipse, measured 
        anti-clockwise from the horizontal
    Output
      id_inner: indices of the pixels nested within the 2 ellipse

    """
    
    x1 = np.arange(0,size)
    y1 = np.arange(0,size)
    x,y = np.meshgrid(y1,x1)
    if yc == None:
        yc = size/2
    if xc == None:
        xc = size/2
    ellipse_ext = (x-xc)**2/(a+delta)**2+(y-yc)**2/(b+epsilon)**2-1
    ellipse_int = (x-xc)**2/(a-delta)**2+(y-yc)**2/(b-epsilon)**2-1
    if theta != 0:
        ellipse_ext = rot.frame_rotate(ellipse_ext,-theta)
        ellipse_int = rot.frame_rotate(ellipse_int,-theta)
    id_inner_ellipse = np.where((ellipse_ext < 0) * (ellipse_int > 0))
    return id_inner_ellipse

def elliptical_mask_advanced(size,a1,b1,a2,b2,xc1=None,yc1=None,yc2=None,
                             xc2=None,theta1=0,theta2=0):

    """
    Function ellitical_mask builds an elliptical mask. Two ellipses of semi major 
    axis a1 and a2 and of semi-minor axis b1 and b2 are built.
    The mask is 0 everywhere outside the 2 ellipses and 1 within the 2 ellipses.
    Arguments:
        1. size: the size of the image
        2. a1: semi-major axis of the inner ellipse
        3. b1: semi-minor axis of the inner ellipse
        4. a2: semi-major axis of the outer ellipse
        5. b2: semi-minor axis of the outer ellipse
    Optional arguments:
        6.yc1: the x center of the ellipse in y. By default, size/2
        7.xc1: the y center of the ellipse in x. By default, size/2
        8. theta1: the position angle of the semi-major axis of the inner ellipse, measured 
        anti-clockwise from the horizontal
    Output
      id_inner: indices of the pixels nested within the 2 ellipse

    """    
    x1 = np.arange(0,size)
    y1 = np.arange(0,size)
    x,y = np.meshgrid(y1,x1)
    if yc1 == None:
        yc1 = size/2
    if xc1 == None:
        xc1 = size/2
    if yc2 == None:
        yc2 = size/2
    if xc2 == None:
        xc2 = size/2
    ellipse_int = (x-xc1)**2/a1**2+(y-yc1)**2/b1**2-1
    ellipse_ext = (x-xc2)**2/a2**2+(y-yc2)**2/b2**2-1
    if theta1 != 0:
        ellipse_int = rot.frame_rotate(ellipse_int,-theta1)
    if theta2 != 0:
        ellipse_ext = rot.frame_rotate(ellipse_ext,-theta2)
    id_inner_ellipse = np.where((ellipse_ext < 0) * (ellipse_int > 0))
    id_outer_ellipse = np.where((ellipse_ext > 0) + (ellipse_int < 0))
    return id_inner_ellipse,id_outer_ellipse

def ellipse_polynomial_coeff(a,b,x0,y0,pa):
     """
     This function returns the polynomial coefficient of an ellipse which is 
     parametrized through a semi-major axis a, a semi-minor axis b, an offset 
     (x0,y0) and a position angle pa measured from North counter-clockwise. The 
     output is an array called coeff such that the ellipse equation is
     coeff[0]*x**2 + coeff[1]*x*y + coeff[2]*y**2 + coeff[3]*x + coeff[4]*y + coeff[5] = 0
     with coeff[5]=1     
     """
     trigo_pa=-pa-math.pi/2
     cosa=np.cos(trigo_pa)
     sina=np.sin(trigo_pa)
     coeff=np.zeros(6)
     coeff[0]=a**2*cosa**2+b**2*sina**2
     coeff[1]=2*cosa*sina*(b**2-a**2)
     coeff[2]=a**2*sina**2+b**2*cosa**2
     coeff[3]=a**2*(-2*cosa**2*x0+2*cosa*sina*y0)+b**2*(-2*cosa*sina*y0 - 2*sina**2*x0)
     coeff[4]=a**2*(2*cosa*sina*x0 - 2*sina**2*y0)+b**2*(- 2*cosa**2*y0 - 2*cosa*sina*x0)
     coeff[5]=-a**2*b**2+a**2*(cosa**2*x0**2 - 2*cosa*sina*x0*y0 + sina**2*y0**2)+b**2*(cosa**2*y0**2+sina**2*x0**2+ 2*cosa*sina*x0*y0)    
     return coeff/coeff[5]


###############################################################################
###############################################################################
##  Algebraic solution for an ellipse fitting
## from http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
###############################################################################
###############################################################################

def fitEllipse(x,y):
    """
    This function minimizes 
    a[0]*x**2 + a[1]*x*y + a[2]*y**2 + a[3]*x + a[4]*y + a[5] 
    for a set of points (x,y) and returns the coefficients.
    """
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    return a

def ellipse_center(coeff):
    """
    This function converts a set of 6 polynomial coefficients defined as 
    coeff[0]*x**2 + coeff[1]*x*y + coeff[2]*y**2 + coeff[3]*x + coeff[4]*y + coeff[5] 
    to the ellipse parameters in a new frame aligned with the axis of the ellipse. It returns
    the offset of the ellipse center in this new frame.
    Adapted from http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
    """
    a,b,c,d,f,g = coeff[0], coeff[1]/2., coeff[2], coeff[3]/2., coeff[4]/2., coeff[5]
    delta = b*b-a*c
    if delta ==0:
        print('Warning the ellipse is degenerate: delta=0 (single point)')
    x0=(c*d-b*f)/delta
    y0=(a*f-b*d)/delta
    return np.array([x0,y0])

def ellipse_angle_of_rotation(coeff):
    """
    This function converts a set of 6 polynomial coefficients defined as 
    coeff[0]*x**2 + coeff[1]*x*y + coeff[2]*y**2 + coeff[3]*x + coeff[4]*y + coeff[5] 
    to the ellipse parameters in a new frame aligned with the axis of the ellipse. It returns
    the position angle of the ellipse.
    Adapted from http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
    """
    a,b,c,d,f,g = coeff[0] , coeff[1]/2, coeff[2], coeff[3]/2, coeff[4]/2, coeff[5]
    if (a == c):
        print('Warning: the ellipse is degenerate to a circle, position angle set to 0 by default')
        return 0
    return 0.5*np.arctan(2*b/(a-c))

#def ellipse_axis_length( a ):
#    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
#    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
#    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
#    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
#    res1=np.sqrt(up/down1)
#    res2=np.sqrt(up/down2)
#    return np.array([res1, res2])
    
def ellipse_axis_length(coeff):
    """
    This function converts a set of 6 polynomial coefficients defined as 
    coeff[0]*x**2 + coeff[1]*x*y + coeff[2]*y**2 + coeff[3]*x + coeff[4]*y + coeff[5] 
    to the ellipse parameters in a new frame aligned with the axis of the ellipse. It returns
    the semi-major and semi-minor axis of the ellipse.
    Adapted from http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
    """
    a,b,c,d,f,g = coeff[0] , coeff[1]/2, coeff[2], coeff[3]/2, coeff[4]/2, coeff[5]
    up = 2*(a*f**2+c*d**2+g*b**2-2*b*d*f-a*c*g)
#    print((a-c)*(a-c))
    down1=(b**2-a*c)*( np.sqrt((a-c)**2+4*b**2)-(a+c))
    down2=(b**2-a*c)*(-np.sqrt((a-c)**2+4*b**2)-(a+c))
#    print(down1,down2)
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])


###############################################################################
###############################################################################
##  Least square fit
###############################################################################
###############################################################################

def chi2(param_model, theta, rho, rho_error):
    """
    This functions defines a chi squared between measurements given as (theta,rho)
    and an ellipse parametrized in the sky plabe by param_model=x0, y0, a, b, alpha
    The error of each point is defined as the distance between the point of the 
    ellipse at the same theta and rho.
    """
    x0, y0, a, b, alpha = param_model
    x = rho*np.cos(theta)
    y = rho*np.sin(theta)
    distance_data_to_ell_center = np.sqrt((x-x0)**2+(y-y0)**2)
    p=(y0-y)/(x0-x)
    phi = np.arctan(a/b*(p*np.cos(alpha)-np.sin(alpha))/(p*np.sin(alpha)+np.cos(alpha)))
    distance_ell_to_ell_center = np.sqrt( a**2*np.cos(phi)**2+b**2*np.sin(phi)**2)
    sigma2 = rho_error**2
    return np.sum((distance_data_to_ell_center-distance_ell_to_ell_center)**2/sigma2)
    
def chi2_from_deprojected_ellipse(orbital_param_model, theta, rho, rho_error):
    """
    This functions defines a chi squared between measurements given as (theta,rho)
    and an ellipse parametrized in the orbital plane by (a,e,itilt,omega,Omega).
    the angles must be expressed in radians.
    The error of each point is defined as the distance between the point of the 
    ellipse at the same theta and rho.
    """
#    a,e,itilt,omega,Omega=orbital_param_model
    a,b,x0,y0,alpha=projected_param_from_ellipse_param(*orbital_param_model[0:6],verbose=False)
    skyplane_param_model=x0,y0,a,b,alpha
    return chi2(skyplane_param_model, theta, rho, rho_error)

###############################################################################
###############################################################################
## Deprojection of the ellipse
###############################################################################
###############################################################################
  
def deprojection_from_poly_coeff(coeff,verbose=True):
    """
    This function takes in input the ellipse polynomial values a such as
    coeff[0]*x**2 + coeff[1]*x*y + coeff[2]*y**2 + coeff[3]*x + coeff[4]*y + coeff[5] 
    and return the deprojected parameters of the ellipse :
    omega = argument of pericenter
    Omega = longitude of ascending node
    a = semi-major axis
    e = eccentricity
    """    
    # This nomenclature is from Smart 1930
    A=coeff[0]/coeff[5]
    H=coeff[1]/2./coeff[5]
    B=coeff[2]/coeff[5]
    G=coeff[3]/2./coeff[5]
    F=coeff[4]/2./coeff[5]
    tan2Omega=(2*(H-F*G))/(F**2-G**2+A-B)
#    print('    tan(2Omega)={0:5.2f}'.format(tan2Omega))
    Omega=(np.arctan(tan2Omega))/2 
    tan2ioverp2=2*(H-F*G)/np.sin(2*Omega)
    if tan2ioverp2 < 0:
        Omega=(np.arctan(tan2Omega)+math.pi)/2 
        tan2ioverp2=2*(H-F*G)/np.sin(2*Omega)
        if verbose:
            print('Warning: increase Omega by pi/2 to avoid inconsistency')            
    p=np.sqrt(2/(F**2+G**2-A-B-tan2ioverp2))
    itilt=np.arctan(p*np.sqrt(tan2ioverp2))
    denom_tanomega=G*np.cos(Omega)+F*np.sin(Omega)
#    print('    denom tan(omega)={0:5.2f}'.format(denom_tanomega))
    if denom_tanomega != 0:
        omega=np.arctan((F*np.cos(Omega)-G*np.sin(Omega))*np.cos(itilt)/(G*np.cos(Omega)+F*np.sin(Omega)))
    else:
        omega=0
    e=-p/np.cos(omega)*(G*np.cos(Omega)+F*np.sin(Omega))
    true_a=p/(1-e**2)
    if verbose:
        a,b=ellipse_axis_length(coeff)
        itilt_before=np.arccos(np.min([a,b])/np.max([a,b]))
        pa=ellipse_angle_of_rotation(coeff)
        x0,y0=ellipse_center(coeff)
        offset_distance=np.sqrt(x0**2+y0**2)
        omega_before=np.arctan(y0/x0) #+270
        e_before=offset_distance/(b)
        print('Parameters of the ellipse before deprojection')
        print('    a={0:5.2f}'.format(np.max([a,b])))
        print('    e={0:5.3f}'.format(e_before))
        print('    offset={0:5.2f}'.format(offset_distance))
        print('    direction of offset={0:5.2f} deg (from W ccw)'.format(np.rad2deg(omega_before)))
        print('    Omega={0:5.2f} deg'.format(np.rad2deg(pa)))
        print('    i={0:5.2f} deg'.format(np.rad2deg(itilt_before)))
        
        print('Parameters of the ellipse after deprojection')
        print('    a={0:5.2f}'.format(true_a))
        print('    e={0:5.3f}'.format(e))
        print('    p={0:5.3f}'.format(p))
        print('    omega={0:5.2f} deg'.format(np.rad2deg(omega)))
        print('    Omega={0:5.2f} deg'.format(np.rad2deg(Omega)))
        print('    i={0:5.2f} deg'.format(np.rad2deg(itilt)))
    return [true_a, e, omega, Omega,itilt]

def deprojection_from_ellipse_param(a,b,x0,y0,pa,verbose=True):
    """
    This function takes in input the ellipse parameters
    param=a,b,x0,y0,pa (in radian) and
    returns the deprojected parameters of the ellipse :
    a = semi-major axis
    e = eccentricity
    omega = argument of pericenter in radian
    Omega = longitude of ascending node in radian
    i = inclination in radian
    """
    coeff=ellipse_polynomial_coeff(a,b,x0,y0,pa)
    print(coeff)
    return deprojection_from_poly_coeff(coeff,verbose=verbose)
    
#coeff = projected_coeff_from_ellipse_param(a,e,i,omega,Omega)
def projected_coeff_from_ellipse_param(a,e,i,omega,Omega):
    """
    This function takes in input true orbital parameters of an ellipse (a,e,i,
    omega,Omega), the angles being in radians,
    and projects them on the plane of the sky. It returns the polynomial
    coefficent of the ellipse in the plane of the sky (notation from Smart 1930)
    defined as 
    coeff[0]*x**2 + coeff[1]*x*y + coeff[2]*y**2 + coeff[3]*x + coeff[4]*y + coeff[5] 
    """
    n3 = np.cos(i)
    cosomega=np.cos(omega)
    cosOmega=np.cos(Omega)
    sinomega=np.sin(omega)
    sinOmega=np.sin(Omega)
    l1 = cosOmega*cosomega-sinOmega*sinomega*n3
    m1 = sinOmega*cosomega+cosOmega*sinomega*n3
    l2 =-cosOmega*sinomega-sinOmega*cosomega*n3
    m2 =-sinOmega*sinomega+cosOmega*cosomega*n3
    b=a*np.sqrt(1-e**2)
    f = 1./(e**2-1)
    A = f/n3**2*(m2**2/a**2+m1**2/b**2)
    B = f/n3**2*(l2**2/a**2+l1**2/b**2)
    H =-f/n3**2*(l2*m2/a**2+l1*m1/b**2)
    G = f*e*m2 / (a*n3)
    F =-f*e*l2 / (a*n3)
    coeff = [A, 2*H, B, 2*G, 2*F, 1.]
    return coeff

def projected_param_from_ellipse_param(a,e,i,omega,Omega,verbose=True):
    coeff = projected_coeff_from_ellipse_param(a,e,i,omega,Omega)
    if verbose:
        print(coeff)
    x0,y0 = ellipse_center(coeff)
    alpha = ellipse_angle_of_rotation(coeff)
    a,b = ellipse_axis_length(coeff)
    return a,b,x0,y0,alpha
#    return deprojection_from_poly_coeff(coeff,verbose=verbose)

def plot_ellipse(a,b,x0,y0,pa,verbose=True):
    R = np.arange(0,2*np.pi, 0.01)
    x = x0 + a*np.cos(R)*np.cos(pa) - b*np.sin(R)*np.sin(pa)
    y = y0 + a*np.cos(R)*np.sin(pa) + b*np.sin(R)*np.cos(pa)
    if verbose:
        print('x0={0:5.2f}  , y0={1:5.2f}'.format(x0,y0))
        print('a={0:5.2f}  , b={1:5.2f}'.format(a,b))
        print('position angle={0:5.2f}'.format(np.rad2deg(pa)))
    plt.plot(x,y,'r-')
    plt.plot([x0],[y0],'ro')
    plt.grid()
    
if __name__=='__main__':      
   res = ellipse_points(100.,50.,4.,precision=0.2*math.pi/180, step_R=0.1,plot=True)

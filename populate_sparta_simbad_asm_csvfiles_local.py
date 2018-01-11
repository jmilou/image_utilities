#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 14:12:54 2017

@author: cpannetier


This notebook reads all sparta data files located in /data/datalake/fits/ao and populates an index with the data from those files. 
In addition to the data contained in the header and in the fits tables, it queries simbad to resolve the star and retrieve the magnitude, spectral type, etc... 
It also queries the asm index and interpolates the dimm, tau0, wind values, etc...

"""

import os,glob,sys
from shutil import copyfile
import datetime
from astropy.io import fits, ascii
from astropy.time import Time
from elasticsearch import Elasticsearch,ElasticsearchException
from astropy.time import Time
from astropy.coordinates import ICRS, Galactic, FK4, FK5
from astroquery.simbad import Simbad
Simbad.ROW_LIMIT = 30 # now any query fetches at most 30 rows
Simbad.TIMEOUT = 62 # we wait 62s until we 
import numpy as np
import matplotlib.pyplot as plt
import astropy.coordinates as coord
import astropy.units as u

from scipy.interpolate import interp1d
import pdb #        pdb.set_trace() 
import pprint
import warnings
import time
import pandas as pd
import subprocess
import pdb
#from ecmwf_utilities import request_ecmwf

from plot_sparta_data_cyril import plot_sparta_data_cyril
import dtts_peak_finder as d

#from elasticsearch_utilities import return_asm_dico #weigthed_average , 
# to avoid to print the RuntimeWarning: invalid value encountered in less
warnings.filterwarnings("ignore",category=RuntimeWarning)
# to avoid to print the UserWarning: Warning: The script line number 3 raised an error: No astronomical object found :  (error.line, error.msg))
warnings.filterwarnings("ignore",category=UserWarning)

def get_infos_from_txtfiles(txtfilename):
    """
    Populate a csv files with target name and fits files associated to the txtfile
    """
    
    try:
        with open(txtfilename, "r") as myfile:
            fitsfiles = []
            for line in myfile:
                if line.startswith('Target:'):            
                    target = str.strip(line.split('Target:',1)[1])

#                if line.startswith('SPHERE_GEN_SPARTA'):
#                    fitsfiles.append(str.strip(line.split('\t')[1])+'.fits')
            fitsfile = txtfilename.replace('NL.txt','fits').split("/")[-1]    
                    
    except Exception as e:
        print('Problem while reading {0:s}. Returning an empty dictionnary.'.format(txtfilename))
        print(e)
        target = np.nan        
    
    night, hour = txtfilename.split(".")[1].split("T")
    if int(hour[:2]) < 14: 
        night_datetime = datetime.datetime.strptime(night, "%Y-%m-%d")
        night_datetime += datetime.timedelta(days=-1)
        night = datetime.datetime.strftime(night_datetime, "%Y-%m-%d")
    
    return (target, fitsfile, night)

#### First, we define the different useful functions:
    
#Function which will get infos from simbad and return a dictionnary with all these infos

def get_star_properties_from_simbad(fitsfilename, target, previousDoc=None):
    """
    Returns a dictionary with the properties of the star, after searching in the local catalog or querying 
    simbad if it was not in the local catalog.
    If no star is found, it returns an empty dictionnary
    The algorithm searches for a star brighter than V=16 within 10 arcsec of the 
    RA/DEC coordinates.
    If there is only one, it is considered to be the AO target. 
    If there is none or more than one, it looks for the target name as defined by the user and 
    checks if it is resolved by Simbad.
    If it is not resolved, it considers as the AO target the object closer to the telescope RA/DEC.
    If there was no target in the search radius, it enlarges the search field to 20 arcsec and selects 
    the object closer to the telescope RA/DEC.
    """
    
    global catalog_pd
    header = fits.getheader(fitsfilename)

    ra = header['RA']*u.degree
    dec = header['DEC']*u.degree
    coords = coord.SkyCoord(ra,dec)
    if previousDoc != None:
        if 'RA' in previousDoc.keys() and 'DEC' in previousDoc.keys() and previousDoc['RA']==coords.ra.to_string(unit=u.hourangle,sep=' ') and previousDoc['DEC']==coords.dec.to_string(unit=u.degree,sep=' '):
            print('Same star as previously. Copying the simbad values in the new Simbad dictionary')
            simbad_dico = {}
            for key,val in previousDoc.items():
                if key.startswith('simbad'):
                    simbad_dico[key]=val
            return simbad_dico
    date = Time(header['DATE-OBS'])
    customSimbad0=customSimbad
    customSimbad0.add_votable_fields('ra(2;A;ICRS;J2000;2000)',\
                                     'dec(2;D;ICRS;J2000;2000)')
    customSimbad0.add_votable_fields('ra(2;A;FK5;J{0:.3f};2000)'.format(date.jyear),\
                                     'dec(2;D;FK5;J{0:.3f};2000)'.format(date.jyear))
    search = customSimbad0.query_region(coords,radius=search_radius)
    # search is an object astropy.table.column.MaskedColumn
    if search is None:
        print('No star identified for the RA/DEC pointing. Querying the target name')
            # get the star from target name
        simbad_dico = get_dico_star_properties_from_simbad_target_name_search(target)
        if 'simbad_FLUX_V' in simbad_dico.keys():  
            nb_stars = -1 # nothing else to be done.
        else:
            print('No star corresponding to the target name. Enlarging the search to {0:.0f} arcsec'.format(search_radius_alt.value))
            search = customSimbad0.query_region(coords,radius=search_radius_alt)
            #time.sleep(1) # we add a delay to avoid doing more than 6 queries / s.
            if search is None:
                print('No star identified for the RA/DEC pointing. Stopping the search.')
                nb_stars = 0
            else:
                validSearch = search[search['FLUX_V']<16.]
                nb_stars = len(validSearch)                
    else:
        nb_stars = len(search)
        validSearch = search[search['FLUX_V']<16.]
        nb_stars = len(validSearch)    
    if nb_stars==0:
        print('No star identified for the pointing position.')
        # get the star from target name if we have it in the text file.
        simbad_dico = get_dico_star_properties_from_simbad_target_name_search(target)
        # if we found a star, we add the distance between ICRS coordinates and pointing
        if 'simbad_RA_ICRS' in simbad_dico.keys() and 'simbad_DEC_ICRS' in simbad_dico.keys():
            coords_ICRS_str = ' '.join([simbad_dico['simbad_RA_ICRS'],simbad_dico['simbad_DEC_ICRS']])
            coords_ICRS = coord.SkyCoord(coords_ICRS_str,frame=ICRS,unit=(u.hourangle,u.deg))
            sep_pointing_ICRS = coords.separation(coords_ICRS).to(u.arcsec).value
            simbad_dico['simbad_separation_RADEC_ICRSJ2000']=sep_pointing_ICRS
        # if we found a star, we add the distance between Simbad current coordinates and pointing
        if 'simbad_RA_current' in simbad_dico.keys() and 'simbad_DEC_current' in simbad_dico.keys():
            coords_current_str = ' '.join([simbad_dico['simbad_RA_current'],simbad_dico['simbad_DEC_current']])
            coords_current = coord.SkyCoord(coords_current_str,frame=ICRS,unit=(u.hourangle,u.deg))
            sep_pointing_current = coords.separation(coords_current).to(u.arcsec).value
            simbad_dico['simbad_separation_RADEC_current']=sep_pointing_current
    elif nb_stars>0:
        if nb_stars ==1:
            i_min=0
            print('One star found: {0:s} with V={1:.1f}'.format(\
                  validSearch['MAIN_ID'][i_min].decode('UTF-8'),validSearch['FLUX_V'][i_min]))
            choix = i_min
        else:
            print('{0:d} stars identified within {1:.0f} arcsec. Querying the target name'.format(nb_stars,search_radius.value)) 
            # First we query the target name
            simbad_dico = get_dico_star_properties_from_simbad_target_name_search(target)
            if ('simbad_MAIN_ID' in simbad_dico):
                # the star was resolved and we assume there is a single object corresponding to the search 
                i_min=0
                choix = i_min
            else:
                print('Target not resolved or not in the list. Selecting the closest star.')
                sep_list = []
                brightness_list = []
                for key in validSearch.keys():
                    if key.startswith('RA_2_A_FK5_'):
                        key_ra_current_epoch = key
                    elif key.startswith('DEC_2_D_FK5_'):
                        key_dec_current_epoch = key
                for i in range(nb_stars):
                    ra_i = validSearch[key_ra_current_epoch][i]
                    dec_i = validSearch[key_dec_current_epoch][i]
                    coord_str = ' '.join([ra_i,dec_i])
                    coords_i = coord.SkyCoord(coord_str,frame=FK5,unit=(u.hourangle,u.deg))
                    sep_list.append(coords.separation(coords_i).to(u.arcsec).value)
                    brightness_list.append(validSearch['FLUX_V'][i])
                i_min = np.argmin(sep_list)
                min_sep = np.min(sep_list)
                V_min = np.argmin(brightness_list)
                print('The closest star is: {0:s} with V={1:.1f} at {2:.2f} arcsec'.format(\
                                                                                           validSearch['MAIN_ID'][i_min].decode('UTF-8'),validSearch['FLUX_V'][i_min],min_sep))

                if sep_list[V_min] - min_sep < 3:
                    print('We took the brightest star which is: {0:s} with V={1:.1f} at {2:.2f} arcsec'.format(\
                                                                                            validSearch['MAIN_ID'][V_min].decode('UTF-8'),validSearch['FLUX_V'][V_min],sep_list[V_min]))
                    choix = V_min
                else:
                    choix = i_min

        simbad_dico = populate_simbad_dico(validSearch,choix)
    simbad_dico['DEC'] = coords.dec.to_string(unit=u.degree,sep=' ')
    simbad_dico['RA'] = coords.ra.to_string(unit=u.hourangle,sep=' ')
    # if we found a star, we add the distance between ICRS coordinates and pointing
    if 'simbad_RA_ICRS' in simbad_dico.keys() and 'simbad_DEC_ICRS' in simbad_dico.keys():
        coords_ICRS_str = ' '.join([simbad_dico['simbad_RA_ICRS'],simbad_dico['simbad_DEC_ICRS']])
        coords_ICRS = coord.SkyCoord(coords_ICRS_str,frame=ICRS,unit=(u.hourangle,u.deg))
        sep_pointing_ICRS = coords.separation(coords_ICRS).to(u.arcsec).value
        simbad_dico['simbad_separation_RADEC_ICRSJ2000']=sep_pointing_ICRS
    # if we found a star, we add the distance between Simbad current coordinates and pointing
    if 'simbad_RA_current' in simbad_dico.keys() and 'simbad_DEC_current' in simbad_dico.keys():
        coords_current_str = ' '.join([simbad_dico['simbad_RA_current'],simbad_dico['simbad_DEC_current']])
        coords_current = coord.SkyCoord(coords_current_str,frame=ICRS,unit=(u.hourangle,u.deg))
        sep_pointing_current = coords.separation(coords_current).to(u.arcsec).value
        simbad_dico['simbad_separation_RADEC_current']=sep_pointing_current
        
    return simbad_dico


#Function which will get the info from Simbad, using the name of the target, and return a dictionnary

def get_dico_star_properties_from_simbad_target_name_search(target):
    """
    Returns a dictionary with the properties of the star, after querying simbad
    using the target name as found in the header of the program.
    If no star is found returns an empty dictionnary
    """
    
    simbad_dico = {}
    simbadsearch = customSimbad.query_object(target)
    if simbadsearch is None:
        # failure
        return simbad_dico
    else:
        # successful search
        return populate_simbad_dico(simbadsearch,0)

    
#Function which will update the catalog csv file with the new simbad information

def update_catalog(simbad_dico,target):
    """
    Returns the updated catalog of stars with a new entry of a simbad dico
    """
    if type(target) != str:
            target = simbad_dico['simbad_MAIN_ID']
            
    sub_cat = catalog_pd.loc[catalog_pd['MAIN_ID'] == target]
    if len(sub_cat)==0:
        dico_with_target = simbad_dico.copy()
        dico_with_target['MAIN_ID']=target   
        updated_catalog_pd = catalog_pd.append(dico_with_target, ignore_index=True)
        print('Catalog updated with the entry {0:s}'.format(target))
        return updated_catalog_pd
    
    else: 
        return catalog_pd
    
#Functions used in "get_dico_star_properties_from_simbad_target_name_search" and "get_star_properties_from_simbad"
#Translate the info got in a simbadSearch format into a dictionnary

def populate_simbad_dico(simbad_search_list,i):
    """
    Method not supposed to be used outside the query_simbad method
    Given the result of a simbad query (list of simbad objects), and the index of 
    the object to pick, creates a dictionary with the entries needed.
    """    
    simbad_dico = {}
    for key in simbad_search_list.keys():
        if key in ['MAIN_ID','SP_TYPE','ID_HD','OTYPE','OTYPE_V','OTYPE_3']: #strings
            if not simbad_search_list[key].mask[i]:
                simbad_dico['simbad_'+key] = simbad_search_list[key][i].decode('UTF-8')
        elif key in ['FLUX_V', 'FLUX_R', 'FLUX_I', 'FLUX_J', 'FLUX_H', 'FLUX_K','PMDEC','PMRA']: #floats
            if not simbad_search_list[key].mask[i]:
                simbad_dico['simbad_'+key] = float(simbad_search_list[key][i])
        elif key.startswith('RA_2_A_FK5_'): 
            simbad_dico['simbad_RA_current'] = simbad_search_list[key][i]      
        elif key.startswith('DEC_2_D_FK5_'): 
            simbad_dico['simbad_DEC_current'] = simbad_search_list[key][i]
        elif key=='RA':
            simbad_dico['simbad_RA_ICRS'] = simbad_search_list[key][i]
        elif key=='DEC':
            simbad_dico['simbad_DEC_ICRS'] = simbad_search_list[key][i] 
            
    # In case magnitude R is not included in Simbad, we calculate it using spectral type and magnitude V        
    if 'simbad_FLUX_R' not in simbad_dico.keys() and type(simbad_dico['simbad_MAIN_ID']) != float: # sometimes the name is 'nan' so we can't find the target
        sp_type_str = simbad_dico['simbad_SP_TYPE']
        print(sp_type_str)  
        c=color(sp_type_str)
        if np.isfinite(c):
            magR = simbad_dico['simbad_FLUX_V']-c
            simbad_dico['simbad_FLUX_R'] = magR
        else:
            print('Spectral type unknown')
            
    return simbad_dico


def read_color_table():
    """
    Read the csv file Johnson_color_stars.txt, built from the website
    http://www.stsci.edu/~inr/intrins.html and that gives the Johnson colors 
    of stars depending on their spectral type.
    """
    tablename=path_output+'Johnson_color_stars.txt'
    tab = pd.read_csv(tablename)
    return tab

def extract_spectral_type_code(sp_type_str):
    """
    Function that uses the spectral type given by Simbad (for instance
    G2IV, F5V, G5V+DA1.0, M1.4, M1.5, or F0IV) and that returns 
    the code from 0=B0.0 to 49=M4.0 (same convention as in 
    http://www.stsci.edu/~inr/intrins.html)    
    """
    if not isinstance(sp_type_str,str):
        if isinstance(sp_type_str,unicode):
            sp_type_str=str(sp_type_str)
        else:
            print('Argument is not a string. Returning')
    spectral_type_letter = (sp_type_str[0:1]).upper()
    if spectral_type_letter=='B':
        offset_code = 0.
    elif spectral_type_letter=='A':
        offset_code = 10.
    elif spectral_type_letter=='F':
        offset_code = 20.
    elif spectral_type_letter=='G':
        offset_code = 30.
    elif spectral_type_letter=='K':
        offset_code = 40.
    elif spectral_type_letter=='M':
        offset_code = 48.
    else:
        print('The spectral letter extracted from {0:s} is not within B, A, F, G, K, M.'.format(spectral_type_letter))
        offset_code = 100.
    if sp_type_str[2:3] == '.' :
        spectral_type_number = float(sp_type_str[1:4])
    else:
        try:
            spectral_type_number = float(sp_type_str[1:2])
        except:
            spectral_type_number = 0
    return offset_code+spectral_type_number


def color(sp_type_str,filt='V-R'):
    """
    Reads the table from http://www.stsci.edu/~inr/intrins.html and returns the 
    color of the star corresponding to the spectral type given in input.
    Input:
        - sp_type_str: a string representing a spectral type, such as that returned 
            by a simbad query. Some examples of such strings are
            G2IV, F5V, G5V+DA1.0, M1.4, M1.5, or F0IV
        - filt: a string representing a color, to be chose between
            U-B, B-V, V-R, V-I, V-J, V-H, V-K, V-L, V-M, V-N. By default it is 
            V-R. 
    """
    table = read_color_table()
    if filt not in table.keys():
        print('The color requested "{0:s}" is not in the color table.'.format(filt))
        return
    code = extract_spectral_type_code(sp_type_str)
    if code > 100:
        return np.nan
    else:
        interp_function = interp1d(table['Code'],table[filt],bounds_error=True)
        try:
            col = interp_function(code)
        except ValueError as e:
            print('ValueError: {0:s}'.format(str(e)))
            print('The spectral type code',code,'is out of range')
            print('Returning NaN')   
            return np.nan
    return float(col)



def query_asm(start_date, end_date, asm=True, mass_dimm=True, dimm=True, old_dimm=False,\
              lhatpro=True, sphere_ambi=True, slodar=True, scidar=False, debug=True, path='.'):

    
    """
    Query ASM data y populate corresponding csv files:
        - mass_dimm_dates.csv
        - dimm_dates.csv
        - old_dimm_dates.csv
        - slodar_dates.csv
        - lhatpro_dates.csv
        - sphere_ambi_dates.csv
        - asm_dates.csv
        
    """
    
    if mass_dimm == True:
        # We query the ASM database to get the tau0 from the MASS-DIMM
        start_date_asm_str = start_date
        end_date_asm_str = end_date
        print('Querying mass-dimm data')
        existing_db = [i for i in os.listdir(path) if os.path.isfile(os.path.join(path,i)) and 'mass_dimm_' in i]
        if len(existing_db)>0:
            end_dates = [i[-14:-4] for i in existing_db]
            last_request = max(end_dates)
            print("A csv file already exists, until the date", last_request)
            if last_request > start_date_asm_str:
                last_request_dt = datetime.datetime.strptime(last_request, "%Y-%m-%d")
                start_date_dt = last_request_dt + datetime.timedelta(days=-1)
                start_date_asm_str = datetime.datetime.strftime(start_date_dt, "%Y-%m-%d")
            existing_mass_df = pd.read_csv(os.path.join(path_output,'mass_dimm_{0:s}_{1:s}.csv'.format(start_date,last_request)),skiprows=0,skipfooter=5,engine='python')
            if len(existing_mass_df.keys())<2:
                print('No data to be read in the mass-dimm file.')
                print('Empty data in {0:s}'.format(os.path.join(path_output,'mass_dimm_{0:s}_{1:s}.csv'.format(start_date,last_request))))
                existing_db = []
            else:
                existing_mass_df = existing_mass_df.loc[existing_mass_df['date'] < start_date_asm_str] 
        else:
            last_request='2015-01-01'
          
                
        request_asm_str = ['wget','-O',os.path.join(path_output,'mass_dimm_{0:s}_{1:s}.csv'.format(start_date,end_date)),\
                           'http://safwebint01.hq.eso.org/wdb/wdb/asmcomm/mass_paranal/query?wdbo=csv&start_date={0:s}..{1:s}&top=1000000000&tab_fwhm=1&tab_fwhmerr=0&tab_tau=1&tab_tauerr=0&tab_tet=0&tab_teterr=0&tab_alt=0&tab_alterr=0&tab_fracgl=1&tab_turbfwhm=1&tab_tau0=1&tab_tet0=0&tab_turb_alt=0&tab_turb_speed=1'.format(start_date_asm_str,end_date_asm_str)]
        if debug:
            print(' '.join(request_asm_str))
        output,error = subprocess.Popen(request_asm_str,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()
        if debug:
            print(output.decode('UTF8'))
        if error != None:
            print('Error during the request of the ASM MASS-DIMM database:')
            print(error)
    
        # Now we read the MASS-DIMM file
        try:
            mass_df = pd.read_csv(os.path.join(path_output,'mass_dimm_{0:s}_{1:s}.csv'.format(start_date,end_date)),skiprows=1,skipfooter=5,engine='python')

            if debug:
                if len(mass_df.keys())<2:
                    print('No data to be read in the mass-dimm file.')
                    raise IOError('Empty data in {0:s}'.format(os.path.join(path_output,'mass_dimm_{0:s}_{1:s}.csv'.format(start_date,end_date))))
                else:
                    print('The MASS-DIMM file contains {0:d} values.'.format(len(mass_df)))
            mass_df.rename(columns={'Date time': 'date',\
                            'MASS Tau0 [s]':'MASS_tau0',\
                            'MASS-DIMM Cn2 fraction at ground':'MASS-DIMM_fracgl',\
                            'MASS-DIMM Tau0 [s]':'MASS-DIMM_tau0',\
                            'MASS-DIMM Turb Velocity [m/s]':'MASS-DIMM_turb_speed',\
                            'MASS-DIMM Seeing ["]':'MASS-DIMM_seeing',\
                            'Free Atmosphere Seeing ["]':'MASS_freeatmos_seeing'}, inplace=True)        
            if len(existing_db)>0:
                mass_df = existing_mass_df.append(mass_df, ignore_index=True)
            time_mass_dimm_asm = Time(list(mass_df['date']),format='isot',scale='utc')
            mass_df.to_csv(os.path.join(path_output,'mass_dimm_{0:s}_{1:s}.csv'.format(start_date,end_date)), index=False)
        except Exception as e:
            time_mass_dimm_asm=None
            if debug:
                print(e)        
                print("The plot won't contain any MASS-DIMM data.")

    if dimm == True:
        # Now we query the ASM database to get the seeing from the DIMM
        print('Querying dimm data')
        start_date_asm_str = start_date
        end_date_asm_str = end_date
        existing_db = [i for i in os.listdir(path) if os.path.isfile(os.path.join(path,i)) and 'dimm_' in i[:7]]
        if len(existing_db)>0:
            end_dates = [i[-14:-4] for i in existing_db]
            last_request = max(end_dates)
            print("A csv file already exists, until the date", last_request)
            if last_request > start_date_asm_str:
                last_request_dt = datetime.datetime.strptime(last_request, "%Y-%m-%d")
                start_date_dt = last_request_dt + datetime.timedelta(days=-1)
                start_date_asm_str = datetime.datetime.strftime(start_date_dt, "%Y-%m-%d")
            existing_dimm_df = pd.read_csv(os.path.join(path_output,'dimm_{0:s}_{1:s}.csv'.format(start_date,last_request)),skiprows=0,skipfooter=5,engine='python')
            if len(existing_dimm_df.keys())<2:
                print('No data to be read in the dimm file.')
                print('Empty data in {0:s}'.format(os.path.join(path_output,'dimm_{0:s}_{1:s}.csv'.format(start_date,last_request))))
                existing_db = []
            else:
                existing_dimm_df = existing_dimm_df.loc[existing_dimm_df['date'] < start_date_asm_str]
        else:
            last_request='2015-01-01'    
            
        request_asm_str = ['wget','-O',os.path.join(path_output,'dimm_{0:s}_{1:s}.csv'.format(start_date,end_date)),\
                           'http://safwebint01.hq.eso.org/wdb/wdb/asmcomm/dimm_paranal/query?wdbo=csv&start_date={0:s}..{1:s}&top=1000000000&tab_fwhm=1&tab_rfl=0&tab_rfl_time=0'.format(\
                           start_date_asm_str,end_date_asm_str)]
        output,error = subprocess.Popen(request_asm_str,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()
        if debug:
            print(' '.join(request_asm_str))
            print(output.decode('UTF8'))
        if error != None:
            print('Error during the request of the ASM DIMM database:')
            print(error)
        
        # Now we read the DIMM file
        try:
            dimm_df = pd.read_csv(os.path.join(path_output,'dimm_{0:s}_{1:s}.csv'.format(start_date,end_date)),skiprows=1,skipfooter=5,engine='python')
            
            if debug:
                if len(dimm_df.keys())<2:
                    print('No data to be read in the dimm file.')
                    raise IOError('Empty data in {0:s}'.format(os.path.join(path_output,'dimm_{0:s}_{1:s}.csv'.format(start_date,end_date))))
                else:
                    print('The DIMM file contains {0:d} values.'.format(len(dimm_df)))
            dimm_df.rename(columns={'Date time': 'date',\
                            'DIMM Seeing ["]':'dimm_seeing'}, inplace=True)
            if len(existing_db)>0:
                dimm_df = existing_dimm_df.append(dimm_df, ignore_index=True)
            time_dimm_asm = Time(list(dimm_df['date']),format='isot',scale='utc')
            dimm_df.to_csv(os.path.join(path_output,'dimm_{0:s}_{1:s}.csv'.format(start_date,end_date)),index=False)
        except Exception as e:
            time_dimm_asm=None
            if debug:
                print(e)
                print("The plot won't contain any DIMM data.")
    
    if old_dimm == True:
        # Now we query the old dimm in case data were taken before 2016-04-04T10:08:39
    
        print('Querying old dimm data')
        start_date_asm_str = start_date
        end_date_asm_str = end_date
        existing_db = [i for i in os.listdir(path) if os.path.isfile(os.path.join(path,i)) and 'old_dimm_' in i]
        if len(existing_db)>0:
            end_dates = [i[-14:-4] for i in existing_db]
            last_request = max(end_dates)
            print("A csv file already exists, until the date", last_request)
            if last_request > start_date_asm_str:
                last_request_dt = datetime.datetime.strptime(last_request, "%Y-%m-%d")
                start_date_dt = last_request_dt + datetime.timedelta(days=-1)
                start_date_asm_str = datetime.datetime.strftime(start_date_dt, "%Y-%m-%d")
            existing_old_dimm_df = pd.read_csv(os.path.join(path_output,'old_dimm_{0:s}_{1:s}.csv'.format(start_date,last_request)),skiprows=0,skipfooter=5,engine='python')
            if len(existing_old_dimm_df.keys())<2:
                print('No data to be read in the mass-dimm file.')
                print('Empty data in {0:s}'.format(os.path.join(path_output,'old_dimm_{0:s}_{1:s}.csv'.format(start_date,last_request))))
                existing_db = []
            else:
                existing_old_dimm_df = existing_old_dimm_df.loc[existing_old_dimm_df['date'] < start_date_asm_str]
        else:
            last_request='2015-01-01'     
                        
        request_asm_str = ['wget','-O',os.path.join(path_output,'old_dimm_{0:s}_{1:s}.csv'.format(start_date,end_date)),\
                                  'http://archive.eso.org/wdb/wdb/asm/historical_ambient_paranal/query?wdbo=csv&start_date={0:s}..{1:s}&top=1000000000&tab_fwhm=1&tab_airmass=0&tab_rfl=0&tab_tau=1&tab_tet=0'.format(\
                                  start_date_asm_str,end_date_asm_str)]
        output,error = subprocess.Popen(request_asm_str,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()
        if debug:
            print(' '.join(request_asm_str))
            print(output.decode('UTF8'))
        if error != None:
            print('Error during the request of the ASM OLD DIMM database:')
            print(error)
        # Now we read the old DIMM file
        try:
            old_dimm_df = pd.read_csv(os.path.join(path_output,'old_dimm_{0:s}_{1:s}.csv'.format(start_date,end_date)),skiprows=1,skipfooter=5,engine='python')
        
            if debug:
                if len(old_dimm_df.keys())<2:
                    print('No data to be read in the old dimm file.')
                    raise IOError('Empty data in {0:s}'.format(os.path.join(path_output,'old_dimm_{0:s}_{1:s}.csv'.format(start_date,end_date))))
                else:
                    print('The old DIMM file contains {0:d} values.'.format(len(old_dimm_df)))
            old_dimm_df.rename(columns={'Date time': 'date',\
                                'DIMM Seeing ["]':'old_dimm_seeing',\
                                'Tau0 [s]':'old_dimm_tau0'}, inplace=True)
            if len(existing_db)>0:
                old_dimm_df = existing_old_dimm_df.append(old_dimm_df, ignore_index=True)    
            time_old_dimm_asm = Time(list(old_dimm_df['date']),format='isot',scale='utc')
            old_dimm_df.to_csv(os.path.join(path_output,'old_dimm_{0:s}_{1:s}.csv'.format(start_date,end_date)),index=False)
        except Exception as e:
            time_old_dimm_asm=None
            if debug:
                print(e)
                print("The plot won't contain any old DIMM data.")

    if slodar == True:
        # Now we query the ASM database to get the seeing from the SLODAR
        print('Querying slodar data')
        start_date_asm_str = start_date
        end_date_asm_str = end_date
        existing_db = [i for i in os.listdir(path) if os.path.isfile(os.path.join(path,i)) and 'soldar_' in i]
        if len(existing_db)>0:
            end_dates = [i[-14:-4] for i in existing_db]
            last_request = max(end_dates)
            print("A csv file already exists, until the date", last_request)
            if last_request > start_date_asm_str:
                last_request_dt = datetime.datetime.strptime(last_request, "%Y-%m-%d")
                start_date_dt = last_request_dt + datetime.timedelta(days=-1)
                start_date_asm_str = datetime.datetime.strftime(start_date_dt, "%Y-%m-%d")
            existing_slodar_df = pd.read_csv(os.path.join(path_output,'slodar_{0:s}_{1:s}.csv'.format(start_date,last_request)),skiprows=0,skipfooter=5,engine='python')
            
            if len(existing_slodar_df.keys())<2:
                print('No data to be read in the slodar file.')
                print('Empty data in {0:s}'.format(os.path.join(path_output,'slodar_{0:s}_{1:s}.csv'.format(start_date,last_request))))
                existing_db = []
            else:
                existing_slodar_df = existing_slodar_df.loc[existing_slodar_df['date'] < start_date_asm_str]
        else:
            last_request='2015-01-01'     
            
        request_asm_str = ['wget','-O',os.path.join(path_output,'slodar_{0:s}_{1:s}.csv'.format(start_date,end_date)),\
                'http://safwebint01.hq.eso.org/wdb/wdb/asmcomm/slodar_paranal/query?wdbo=csv&start_date={0:s}..{1:s}&top=1000000000&tab_cnsqs_uts=1&tab_fracgl300=1&tab_fracgl500=1&tab_hrsfit=1&tab_fwhm=1'.format(start_date_asm_str,end_date_asm_str)]
        output,error = subprocess.Popen(request_asm_str,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()
        if debug:
            print(' '.join(request_asm_str))
            print(output.decode('UTF8'))
        if error != None:
            print('Error during the request of the ASM SLODAR database:')
            print(error)
        
        # Now we read the SLODAR file
        try:
            slodar_df = pd.read_csv(os.path.join(path_output,'slodar_{0:s}_{1:s}.csv'.format(start_date,end_date)),skiprows=1,skipfooter=5,engine='python')
    
            if debug:
                if len(slodar_df.keys())<2:
                    print('No data to be read in the slodar file.')
                    raise IOError('Empty data in {0:s}'.format(os.path.join(path_output,'slodar_{0:s}_{1:s}.csv'.format(start_date,end_date))))
                else:
                    print('The slodar file contains {0:d} values.'.format(len(slodar_df)))
            slodar_df.rename(columns={'Date time': 'date','Cn2 above UTs [10**(-15)m**(1/3)]':'Cn2_above_UT',\
                              'Cn2 fraction below 300m':'slodar_GLfrac_300m',\
                              'Cn2 fraction below 500m':'slodar_GLfrac_500m',\
                              'Surface layer profile [10**(-15)m**(1/3)]':'slodar_surface_layer',\
                              'Seeing ["]':'slodar_seeing'}, inplace=True)
            if len(existing_db)>0:
                slodar_df = existing_slodar_df.append(slodar_df, ignore_index=True)
            wave_nb=2*np.pi/lam
            time_slodar_asm = Time(list(slodar_df['date']),format='isot',scale='utc')   
            slodar_df['slodar_r0_above_UT'] = np.power(0.423*(wave_nb**2)*slodar_df['Cn2_above_UT']*1.e-15,-3./5.)
            slodar_df['slodar_seeing_above_UT']= np.rad2deg(lam/slodar_df['slodar_r0_above_UT'])*3600.
            slodar_df['slodar_Cn2_total'] = np.power(slodar_df['slodar_seeing']/2.e7,1./0.6) # in m^1/3
            slodar_df['slodar_surface_layer_fraction'] = slodar_df['slodar_surface_layer']*1.e-15 / slodar_df['slodar_Cn2_total']
            slodar_df.to_csv(os.path.join(path_output,'slodar_{0:s}_{1:s}.csv'.format(start_date,end_date)),index=False)
        except KeyError as e:
            if debug:
                print(e)
                print("The plot won't contain any SLODAR data.")
    #    except ascii.core.InconsistentTableError as e:
    #         if debug:
    #            print(e)
    #            print('There was probably only one SLODAR data point.')
    #            print("The plot won't contain any SLODAR data.")       
        except Exception as e:
            if debug:
                print(e)        
                print("The plot won't contain any SLODAR data.")


    if sphere_ambi == True:
        # Now we query the telescope seeing
        print('Querying SPHERE header data')
        start_date_asm_str = start_date
        end_date_asm_str = end_date
        existing_db = [i for i in os.listdir(path) if os.path.isfile(os.path.join(path,i)) and 'sphere_ambi_' in i]
        if len(existing_db)>0:
            end_dates = [i[-14:-4] for i in existing_db]
            last_request = max(end_dates)
            print("A csv file already exists, until the date", last_request)
            if last_request > start_date_asm_str:
                last_request_dt = datetime.datetime.strptime(last_request, "%Y-%m-%d")
                start_date_dt = last_request_dt + datetime.timedelta(days=-1)
                start_date_asm_str = datetime.datetime.strftime(start_date_dt, "%Y-%m-%d")
            existing_sphere_df = pd.read_csv(os.path.join(path_output,'sphere_ambi_{0:s}_{1:s}.csv'.format(start_date,last_request)),skiprows=0,skipfooter=5,engine='python')
            
            if len(existing_sphere_df.keys())<2:
                print('No data to be read in the sphere_ambi file.')
                print('Empty data in {0:s}'.format(os.path.join(path_output,'sphere_ambi_{0:s}_{1:s}.csv'.format(start_date,last_request))))
                existing_db = []
            else:
                existing_sphere_df = existing_sphere_df.loc[existing_sphere_df['date'] < start_date_asm_str]
        else:
            last_request='2015-01-01'     
              
        request_sphere = ['wget','-O',os.path.join(path_output,'sphere_ambi_{0:s}_{1:s}.csv'.format(start_date,end_date)),\
                'http://archive.eso.org/wdb/wdb/eso/sphere/query?wdbo=csv&night={0:s}..{1:s}&top=1000000000&tab_prog_id=0&tab_dp_id=0&tab_ob_id=0&tab_exptime=0&tab_dp_cat=0&tab_tpl_start=0&tab_dp_type=0&tab_dp_tech=0&tab_seq_arm=0&tab_ins3_opti5_name=0&tab_ins3_opti6_name=0&tab_ins_comb_vcor=0&tab_ins_comb_iflt=0&tab_ins_comb_pola=0&tab_ins_comb_icor=0&tab_det_dit1=0&tab_det_seq1_dit=0&tab_det_ndit=0&tab_det_read_curname=0&tab_ins2_opti2_name=0&tab_det_chip_index=0&tab_ins4_comb_rot=0&tab_ins1_filt_name=0&tab_ins1_opti1_name=0&tab_ins1_opti2_name=0&tab_ins4_opti11_name=0&tab_tel_ia_fwhm=1&tab_tel_ia_fwhmlin=1&tab_tel_ia_fwhmlinobs=1&tab_tel_ambi_windsp=0&tab_night=1&tab_fwhm_avg=0'.format(start_date_asm_str,end_date_asm_str)]
        output,error = subprocess.Popen(request_sphere,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()
        if debug:
            print(' '.join(request_sphere))
            print(output.decode('UTF8'))
        if error != None:
            print('Error during the request of the SPHERE database:')
            print(error)
    
        try:
            sphere_df = pd.read_csv(os.path.join(path_output,'sphere_ambi_{0:s}_{1:s}.csv'.format(start_date,end_date)),skiprows=1,skipfooter=5,engine='python') # 1st line is bank
            
            if debug:
                if len(sphere_df.keys())<2:
                    print('No data to be read in the SPHERE header file.')
                    raise IOError('Empty data in {0:s}'.format(os.path.join(path_output,'sphere_ambi_{0:s}_{1:s}.csv'.format(start_date,end_date))))
                else:
                    print('The SPHERE header file contains {0:d} values.'.format(len(sphere_df)))
            sphere_df.rename(columns={'DATE OBS': 'date'}, inplace=True) # To make it compatible with the other csv files.
            if len(existing_db)>0:
                sphere_df = existing_sphere_df.append(sphere_df, ignore_index=True)
            sphere_keys_to_drop= ['Release Date','Object','RA','DEC','Target Ra Dec','Target l b','OBS Target Name']
            for sphere_key_to_drop in sphere_keys_to_drop:
                sphere_df.drop(sphere_key_to_drop, axis=1, inplace=True)
            time_sphere = Time(list(sphere_df['date']),format='iso',scale='utc')        
            sphere_df.to_csv(os.path.join(path_output,'sphere_ambi_{0:s}_{1:s}.csv'.format(start_date,end_date)),index=False)
        except Exception as e:
            if debug:
                print(e)        
                print("The plot won't contain any data from the SPHERE science files headers.")


    if asm == True:
        # Now we query the meteo tower to get the wind speed, direction and temperature
        print('Querying ASM data')
        start_date_asm_str = start_date
        end_date_asm_str = end_date
        existing_db = [i for i in os.listdir(path) if os.path.isfile(os.path.join(path,i)) and 'asm_' in i[:5]]
        if len(existing_db)>0:
            end_dates = [i[-14:-4] for i in existing_db]
            last_request = max(end_dates)
            print("A csv file already exists, until the date", last_request)
            if last_request > start_date_asm_str:
                last_request_dt = datetime.datetime.strptime(last_request, "%Y-%m-%d")
                start_date_dt = last_request_dt + datetime.timedelta(days=-1)
                start_date_asm_str = datetime.datetime.strftime(start_date_dt, "%Y-%m-%d")
            existing_asm_df = pd.read_csv(os.path.join(path_output,'asm_{0:s}_{1:s}.csv'.format(start_date,last_request)),skiprows=0,skipfooter=5,engine='python')
            if len(existing_asm_df.keys())<2:
                print('No data to be read in the mass-dimm file.')
                print('Empty data in {0:s}'.format(os.path.join(path_output,'mass_dimm_{0:s}_{1:s}.csv'.format(start_date,last_request))))
                existing_db = []
            else:
                existing_asm_df = existing_asm_df.loc[existing_asm_df['date'] < start_date_asm_str]
        else:
            last_request='2015-01-01'     
            
        request_asm = ['wget','-O',os.path.join(path_output,'asm_{0:s}_{1:s}.csv'.format(start_date,end_date)),\
                'http://archive.eso.org/wdb/wdb/asm/meteo_paranal/query?wdbo=csv&start_date={0:s}..{1:s}&top=1000000000&tab_press=0&tab_presqnh=0&tab_temp1=1&tab_temp2=0&tab_temp3=0&tab_temp4=0&tab_tempdew1=0&tab_tempdew2=0&tab_tempdew4=0&tab_dustl1=0&tab_dustl2=0&tab_dusts1=0&tab_dusts2=0&tab_rain=0&tab_rhum1=0&tab_rhum2=0&tab_rhum4=0&tab_wind_dir1=1&tab_wind_dir1_180=0&tab_wind_dir2=0&tab_wind_dir2_180=0&tab_wind_speed1=1&tab_wind_speed2=0&tab_wind_speedu=0&tab_wind_speedv=0&tab_wind_speedw=0'.format(start_date_asm_str,end_date_asm_str)]
        output,error = subprocess.Popen(request_asm,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()
        if debug:
            print(' '.join(request_asm))
            print(output.decode('UTF8'))
        if error != None:
            print('Error during the request of the ASM database:')
            print(error)
        try:
            asm_df = pd.read_csv(os.path.join(path_output,'asm_{0:s}_{1:s}.csv'.format(start_date,end_date)),skiprows=1,skipfooter=5,engine='python') # 1st line is bank
            
            if debug:
                if len(asm_df.keys())<2:
                    print('No data to be read in the ASM file.')
                    raise IOError('Empty data in {0:s}'.format(os.path.join(path_output,'asm_{0:s}_{1:s}.csv'.format(start_date,end_date))))
                else:
                    print('The ASM file contains {0:d} values.'.format(len(asm_df)))
            asm_df.rename(columns={'Date time': 'date','Air Temperature at 30m [C]':'air_temperature_30m[deg]',\
                              'Wind Direction at 30m (0/360) [deg]':'winddir_30m',\
                              'Wind Speed at 30m [m/s]':'windspeed_30m'}, inplace=True)
            if len(existing_db)>0:
                asm_df = existing_asm_df.append(asm_df, ignore_index=True)    
                time_asm = Time(list(asm_df['date']),format='isot',scale='utc')
            asm_df.to_csv(os.path.join(path_output,'asm_{0:s}_{1:s}.csv'.format(start_date,end_date)),index=False)
        except Exception as e:
            if debug:
                print(e)        
                print("The plot won't contain any data from the ASM")

    if lhatpro == True:
        # Now we query the lhatpro
        print('Querying Lhatpro data')
        start_date_asm_str = start_date
        end_date_asm_str = end_date
        existing_db = [i for i in os.listdir(path) if os.path.isfile(os.path.join(path,i)) and 'lhatpro_' in i]
        if len(existing_db)>0:
            end_dates = [i[-14:-4] for i in existing_db]
            last_request = max(end_dates)
            print("A csv file already exists, until the date", last_request)
            if last_request > start_date_asm_str:
                last_request_dt = datetime.datetime.strptime(last_request, "%Y-%m-%d")
                start_date_dt = last_request_dt + datetime.timedelta(days=-1)
                start_date_asm_str = datetime.datetime.strftime(start_date_dt, "%Y-%m-%d")
            existing_lhatpro_df = pd.read_csv(os.path.join(path_output,'lhatpro_{0:s}_{1:s}.csv'.format(start_date,last_request)),skiprows=0,skipfooter=5,engine='python')
            if len(existing_lhatpro_df.keys())<2:
                print('No data to be read in the lhatpro file.')
                print('Empty data in {0:s}'.format(os.path.join(path_output,'lhatpro_{0:s}_{1:s}.csv'.format(start_date,last_request))))
                existing_db = []
            else:
                existing_lhatpro_df = existing_lhatpro_df.loc[existing_lhatpro_df['date'] < start_date_asm_str]
        else:
            last_request='2015-01-01'        
        request_lhatpro = ['wget','-O',os.path.join(path_output,'lhatpro_{0:s}_{1:s}.csv'.format(start_date,end_date)),\
                'http://safwebint01.hq.eso.org/wdb/wdb/asmcomm/lhatpro_paranal/query?wdbo=csv&start_date={0:s}..{1:s}&top=1000000000&tab_integration=0&tab_lwp0=0'.format(start_date_asm_str,end_date_asm_str)]
        output,error = subprocess.Popen(request_lhatpro,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()
        if debug:
            print(' '.join(request_lhatpro))
            print(output.decode('UTF8'))
        if error != None:
            print('Error during the request of the Lhatpro database:')
            print(error)
        try:
            lhatpro_df = pd.read_csv(os.path.join(path_output,'lhatpro_{0:s}_{1:s}.csv'.format(start_date,end_date)),skiprows=1,skipfooter=5,engine='python') # 1st line is bank
            
            if debug:
                if len(lhatpro_df.keys())<2:
                    print('No data to be read in the Lhatpro file.')
                    raise IOError('Empty data in {0:s}'.format(os.path.join(path_output,'lhatpro_{0:s}_{1:s}.csv'.format(start_date,end_date))))
                else:
                    print('The Lhatpro file contains {0:d} values.'.format(len(lhatpro_df)))
            lhatpro_df.rename(columns={'Date time': 'date','IR temperature [Celsius]':'lhatpro_IR_temperature[Celsius]',\
                              'Precipitable Water Vapour [mm]':'lhatpro_pwv[mm]'}, inplace=True)
            if len(existing_db)>0:
                lhatpro_df = existing_lhatpro_df.append(lhatpro_df, ignore_index=True)    
            lhatpro_df.to_csv(os.path.join(path_output,'lhatpro_{0:s}_{1:s}.csv'.format(start_date,end_date)),index=False)
        except Exception as e:
            if debug:
                print(e)        
                print("The plot won't contain any data from the Lhatpro")
                
    if scidar == True:
        # Now we query the scidar
        print('Querying Scidar data')
        start_date_asm_str = start_date
        end_date_asm_str = end_date
        existing_db = [i for i in os.listdir(path) if os.path.isfile(os.path.join(path,i)) and 'scidar_' in i]
        
        scidar_df = pd.DataFrame({"date":[],"scidar_r0":[],"scidar_tau0":[],"scidar_theta":[]})
        
        if len(existing_db)>0:
            end_dates = [i[-14:-4] for i in existing_db]
            last_request = max(end_dates)
            print("A csv file already exists, until the date", last_request)
            if last_request > start_date_asm_str:
                last_request_dt = datetime.datetime.strptime(last_request, "%Y-%m-%d")
                start_date_dt = last_request_dt + datetime.timedelta(days=-1)
                start_date_asm_str = datetime.datetime.strftime(start_date_dt, "%Y-%m-%d")
            existing_scidar_df = pd.read_csv(os.path.join(path_output,'scidar_{0:s}_{1:s}.csv'.format(start_date,last_request)),skiprows=0,skipfooter=5,engine='python')
            if len(existing_scidar_df.keys())<2:
                print('No data to be read in the scidar file.')
                print('Empty data in {0:s}'.format(os.path.join(path_output,'scidar_{0:s}_{1:s}.csv'.format(start_date,last_request))))
                existing_db = []
            else:
                existing_scidar_df = existing_scidar_df.loc[existing_scidar_df['date'] < start_date_asm_str]
                scidar_df = existing_scidar_df.append(scidar_df, ignore_index=True)
        else:
            last_request='2015-01-01'
        list_scidarfiles = os.listdir('/home/jmilli/datalake_sphere/scidar')
        for scidarfile in list_scidarfiles:
            
            date_dt = datetime.datetime.strptime(scidarfile[:8], "%Y%m%d")
            date_str = datetime.datetime.strftime(date_dt, "%Y-%m-%d")
            if date_str > last_request:
                print(scidarfile)
                scidar_df_temp = pd.read_table('/home/jmilli/datalake_sphere/scidar'+scidarfile,skiprows=6,engine='python',\
                                               names=["date","scidar_r0","scidar_theta","scidar_tau0"],usecols=[0,2,4,6]) # 1st line is bank
                scidar_df_temp['date'] = date_str+'T'+scidar_df_temp['date']
                scidar_df_temp['scidar_seeing'] = np.rad2deg(lam / scidar_df_temp['scidar_r0'])*3600
                scidar_df = scidar_df.append(scidar_df_temp, ignore_index=True)
                print(scidar_df.head())
    
        if debug:
            if len(scidar_df.keys())<2:
                print('No data to be read in the Scidar file.')
                raise IOError('Empty data in {0:s}'.format(os.path.join(path_output,'scidar_{0:s}_{1:s}.csv'.format(start_date,end_date))))
            else:
                print('The Scidar file contains {0:d} values.'.format(len(scidar_df)))
           
        scidar_df.to_csv(os.path.join(path_output,'scidar_{0:s}_{1:s}.csv'.format(start_date,end_date)),index=False)


#    # Now we query the ECMWF data
#    print('Querying ECMWF data')
#    pd_ecmwf = request_ecmwf(time_min,time_max)
#    if pd_ecmwf is not None:
#        pd_ecmwf.to_csv(os.path.join(path_output,'ecmwf_{0:s}.csv'.format(end_date)),index=False)
#        time_ecmwf = Time(list(pd_ecmwf['date']))#,format='isot',scale='utc')        


# Let's define the function to query sparta data into sparta fits files

def validParam(AtmPerfParamsTable, data_type=None):
    """
    Returns the indices of valid parameters from the AtmPerfParamsTable table.
    We check that the strehl is  below 98% and r0 below 1m with a positive wind 
    speed smaller than 100m/s (to return a valid tau0).
    """
    
    if data_type==None:
        print("You might put 'ATM', 'VisLoop' or 'IRLoop' in entry")
        return
    elif data_type == 'ATM':
        sr = AtmPerfParamsTable['StrehlRatio']
        r0 = AtmPerfParamsTable['R0']
        ws = AtmPerfParamsTable['WindSpeed']
        validID = []
        for i in range(len(sr)):
            if sr[i]>0. and sr[i]<0.98 and r0[i]>0. and r0[i]<1. and ws[i]>0. and ws[i]<100.:
                validID.append(i)
    elif data_type == 'VisLoop':
        FLUX_avg = AtmPerfParamsTable['Flux_avg']
        FLUX_std = AtmPerfParamsTable['Flux_std']
        validID = []
        for i in range(len(FLUX_avg)):
            if FLUX_avg[i]> 0.:
                validID.append(i)
    elif data_type == 'IRLoop':
        FLUX_avg = AtmPerfParamsTable['Flux_avg']
        FLUX_std = AtmPerfParamsTable['Flux_std']
        validID = []
        for i in range(len(FLUX_avg)):
            if FLUX_avg[i]> 0.:
                validID.append(i)
    return validID


def populate_AtmPerfParams_df(fitsfilename, obs_list, SCIENCE=True, ATM=True, VIS=True, IR=True, DTTS=True):
    """
    Creates a new document in the index sphere_sparta (doctype AtmPerfParams) 
    with the entry corresponding to the atmospheric 
    parameter of the sparta fits file fitsfilename
    It returns the inserted new document.
    """
    hduList = fits.open(fitsfilename)
    header = hduList[0].header
    coords_J2000 = coord.SkyCoord(header['RA']*u.degree,header['DEC']*u.degree)  
    
    header = hduList[0].header
    dico_science = {'TXTFILE':[], 'MAIN_ID':[]}
    for keyword, value in header.items():
        if keyword == 'DATE':
            keyword = '@timestamp'
        if SCIENCE:
            dico_science[keyword] = [value]
            dico_science['TXTFILE'] = obs_list['TXTFILE']
            dico_science['MAIN_ID'] = obs_list['MAIN_ID']
        else:
            dico_science[keyword] = []
    
    ScienceParams = pd.DataFrame(dico_science)        

    AtmParams = pd.DataFrame({'@timestamp': [], 'SPARTA STREHL':[],\
               'SPARTA WINDSPEED': [],'SPARTA SEEING OBS':[],\
               'SPARTA R0 OBS':[],'SPARTA TAU0 OBS':[],\
                'TEL ALT':[],'AIRMASS':[],\
                'SPARTA SEEING':[],\
                'SPARTA R0':[],\
                'SPARTA TAU0':[]})
    VisLoopParams = pd.DataFrame({'@timestamp': [],'FLUX_avg':[], 'FLUX_std':[]})
    IRLoopParams = pd.DataFrame({'@timestamp': [], 'FLUX_avg':[],'FLUX_std':[],'DTTPPos_avg':[],'DTTPPos_std':[],\
                                     'DTTPRes_avg':[],'DTTPRes_std':[]})
    DTTSParams = pd.DataFrame({"@timestamp":[],"LWE_strength":[],"SNR":[],"CHI2":[],"AMP":[]})
    
        
    if ATM:
        AtmPerfParams = hduList['AtmPerfParams'].data
        validID = validParam(AtmPerfParams, data_type='ATM')
        if len(validID)<1:
            print("No ATM data")
        else:
            timestamps_sparta_list = Time((AtmPerfParams["Sec"]+AtmPerfParams["USec"]/1.e6)[validID],format='unix')
            timestamp_sparta_end = timestamps_sparta_list[-1]
            timestamp_tcs = Time(header['DATE'])    
            delta_t = timestamp_tcs-timestamp_sparta_end
            delta_t_value_second = delta_t.to(u.second).value
            timestamps_sparta_list_corrected = timestamps_sparta_list + delta_t
            timestamps_datetime_atm = timestamps_sparta_list_corrected.datetime

#            print('{0:d} valid point. Offset between Sparta and TCS of {1:.0f}s or {2:.4f} year'.format(len(validID),delta_t_value_second,delta_t.to(u.year).value))

            StrehlRatio = AtmPerfParams['StrehlRatio'][validID]
            R0 = AtmPerfParams['R0'][validID]
            WindSpeed = AtmPerfParams['WindSpeed'][validID]
            # We compute tau0 and the seeing from r0
            T0 = 0.314*R0/WindSpeed
            Seeing = np.rad2deg(lam/R0)*3600


            for i,timestamp in enumerate(timestamps_datetime_atm):

                current_coords_altaz = coords_J2000.transform_to(coord.AltAz(obstime=timestamps_sparta_list_corrected[i],location=location))
                altitude = float(current_coords_altaz.alt.value)
                airmass = float(current_coords_altaz.secz)     
                dico_atm = {'@timestamp': [timestamp], 'SPARTA STREHL':[float(StrehlRatio[i])],\
                       'SPARTA WINDSPEED': [float(WindSpeed[i])],'SPARTA SEEING OBS':[float(Seeing[i])],\
                       'SPARTA R0 OBS':[float(R0[i])],'SPARTA TAU0 OBS':[float(T0[i])],\
                        'TEL ALT':[altitude],'AIRMASS':[airmass],\
                        'SPARTA SEEING':[float(Seeing[i])/np.power(airmass,3./5.)],\
                        'SPARTA R0':[float(R0[i])*np.power(airmass,3./5.)],\
                        'SPARTA TAU0':[float(T0[i])*np.power(airmass,3./5.)] }
                AtmParams = AtmParams.append(pd.DataFrame(dico_atm),ignore_index=True)
            AtmParams['TXTFILE'] = obs_list['TXTFILE']
            AtmParams['MAIN_ID'] = obs_list['MAIN_ID']

    if VIS:    
        # Now we add the avergae photon flux on the WFS.
        

        VisLoopParams_fits = hduList['VisLoopParams'].data
        validID = validParam(VisLoopParams_fits, data_type='VisLoop')
        if len(validID)<1:
            print("No VisLoop data")
        else:
            timestamps_sparta_list = Time((VisLoopParams_fits["Sec"]+VisLoopParams_fits["USec"]/1.e6)[validID],format='unix')
            timestamp_sparta_end = timestamps_sparta_list[-1]
            timestamp_tcs = Time(header['DATE'])    
            delta_t = timestamp_tcs-timestamp_sparta_end
            delta_t_value_second = delta_t.to(u.second).value
            timestamps_sparta_list_corrected = timestamps_sparta_list + delta_t
            timestamps_datetime_Vis = timestamps_sparta_list_corrected.datetime

#           print('{0:d} valid point. Offset between VISLOOP and TCS of {1:.0f}s or {2:.4f} year'.format(len(validID),delta_t_value_second,delta_t.to(u.year).value))

            VisLoop_flux_avg = VisLoopParams_fits['Flux_avg'][validID]
            VisLoop_flux_std = VisLoopParams_fits['Flux_std'][validID]

            for i,timestamp in enumerate(timestamps_datetime_Vis):
                dico_Vis = {'@timestamp':[timestamp],'FLUX_avg':[float(VisLoop_flux_avg[i])],'FLUX_std':[float(VisLoop_flux_std[i])]}
                VisLoopParams = VisLoopParams.append(pd.DataFrame(dico_Vis),ignore_index=True)

            VisLoopParams['TXTFILE'] = obs_list['TXTFILE']
            VisLoopParams['MAIN_ID'] = obs_list['MAIN_ID']

    if IR:
        # Now we add the average photon flux on the WFS.
        

        IRLoopParams_fits = hduList['IRLoopParams'].data
        validID = validParam(IRLoopParams_fits, data_type = 'IRLoop')
        if len(validID)<1:
            print("No IRLoop data")
        else:
            timestamps_sparta_list = Time((IRLoopParams_fits["Sec"]+IRLoopParams_fits["USec"]/1.e6)[validID],format='unix')
            timestamp_sparta_end = timestamps_sparta_list[-1]
            timestamp_tcs = Time(header['DATE'])    
            delta_t = timestamp_tcs-timestamp_sparta_end
            delta_t_value_second = delta_t.to(u.second).value
            timestamps_sparta_list_corrected = timestamps_sparta_list + delta_t
            timestamps_datetime_IR = timestamps_sparta_list_corrected.datetime

#            print('{0:d} valid point. Offset between IRLOOP and TCS of {1:.0f}s or {2:.4f} year'.format(len(validID),delta_t_value_second,delta_t.to(u.year).value))

            IRLoop_flux_avg = IRLoopParams_fits['Flux_avg'][validID]
            IRLoop_flux_std = IRLoopParams_fits['Flux_std'][validID]
            IRLoop_DTTPos_avg = IRLoopParams_fits['DTTPPos_avg'][validID]
            IRLoop_DTTPos_std = IRLoopParams_fits['DTTPPos_std'][validID]
            IRLoop_DTTRes_avg = IRLoopParams_fits['DTTPRes_avg'][validID]
            IRLoop_DTTRes_std = IRLoopParams_fits['DTTPRes_std'][validID]


            for i,timestamp in enumerate(timestamps_datetime_IR):
                dico_IR = {'@timestamp': [timestamp],'FLUX_avg':[float(IRLoop_flux_avg[i])],'FLUX_std':[float(IRLoop_flux_std[i])],\
                           'DTTPPos_avg':[float(IRLoop_DTTPos_avg[i])],'DTTPPos_std':[float(IRLoop_DTTPos_std[i])],\
                           'DTTPRes_avg':[float(IRLoop_DTTRes_avg[i])],'DTTPRes_std':[float(IRLoop_DTTRes_std[i])]}
                IRLoopParams = IRLoopParams.append(pd.DataFrame(dico_IR),ignore_index=True)

            IRLoopParams['TXTFILE'] = obs_list['TXTFILE']
            IRLoopParams['MAIN_ID'] = obs_list['MAIN_ID']
    
    if DTTS:
        data_DTTS = hduList['IRPixelAvgFrame'].data
        if data_DTTS.size <= 0:
            print("No DTTS data")
        else:
            cube_DTTS = data_DTTS['Pixels']
            cube_DTTS_inshape = np.resize(cube_DTTS,(len(cube_DTTS),32,32))

            DTTS_peak_finder = d.Dtts_peak_finder(cube_DTTS_inshape)
            DTTS_peak_finder.fit_gaussian(verbose=False,plot=False)
            DTTS_peak_finder.fit_result['strength']

            timestamps_sparta_list = Time((data_DTTS["Sec"]+data_DTTS["USec"]/1.e6)[DTTS_peak_finder.good_frames],format='unix')

            if len(timestamps_sparta_list) <= 0:
                print("No good frames")
            else:    
                timestamp_sparta_end = timestamps_sparta_list[-1]
                timestamp_tcs = Time(header['DATE'])   
                delta_t = timestamp_tcs-timestamp_sparta_end
                delta_t_value_second = delta_t.to(u.second).value
                timestamps_sparta_list_corrected = timestamps_sparta_list + delta_t
                timestamps_datetime_DTTS = timestamps_sparta_list_corrected.datetime

                for i,timestamp in enumerate(timestamps_datetime_DTTS):
                    dico_DTTS = {'@timestamp': [timestamp],"LWE_strength":[float(DTTS_peak_finder.fit_result['strength'][i])],\
                                 "SNR":[float(DTTS_peak_finder.fit_result['strength'][i]/DTTS_peak_finder.fit_result['threshold'][i])],\
                               "CHI2":[float(DTTS_peak_finder.fit_result['CHI2_r'][i])],"AMP":[float(DTTS_peak_finder.fit_result['AMP'][i])]}
                    DTTSParams = DTTSParams.append(pd.DataFrame(dico_DTTS),ignore_index=True)

            DTTSParams['TXTFILE'] = obs_list['TXTFILE']
            DTTSParams['MAIN_ID'] = obs_list['MAIN_ID']

    hduList.close()
    return ScienceParams, AtmParams, VisLoopParams, IRLoopParams, DTTSParams



"""
We start the algorithm
"""

# Directory where we can find all the sparta fits
path_sparta = '/Users/cpannetier/Documents/Data/SPARTA'

# Directory of the script
path_root = os.getcwd()

# Directory where you save all your csv files
path_output = '/Users/cpannetier/Documents/Data/ASM data'

if not os.path.exists(path_output):
    os.mkdir(path_output)
    
files_fits=sorted(glob.glob(os.path.join(path_sparta,'SPHER*.fits')))
files_txt=sorted(glob.glob(os.path.join(path_sparta,'SPHER*.txt')))
nfiles_fits = len(files_fits)
nfiles_txt = len(files_txt)
print('There are {0:d} fits files in {1:s}'.format(nfiles_fits,path_sparta))
print('There are {0:d} txt files in {1:s}'.format(nfiles_txt,path_sparta))  
print('So %f txtfiles are missing' % (nfiles_fits - nfiles_txt))    
  


"""
First we read all the txt files and write the target names corresponding in a csv file obs_list.csv. 
If it doesn't exist, we create it. If it already exists, we complete it with the new observations.
"""
    
# obs_list is a Panda DataFrame({"TXTFILE":[], "MAIN_ID":[], "FITSFILE":[], "Night":[]})
try:
    obs_list = pd.read_csv(path_output+'/obs_list.csv')
except:
    print("WARNING ! File obs_list.csv does'nt exist, we create it")
    obs_list = pd.DataFrame({"TXTFILE":[], "MAIN_ID":[], "FITSFILE":[], "Night":[]})
    obs_list.to_csv(path_output+"obs_list.csv")
    
additional_obs = pd.DataFrame({"TXTFILE":[], "MAIN_ID":[], "FITSFILE":[], "Night":[]})
for txtfile in files_txt:
    if txtfile.split('/')[-1] not in obs_list['TXTFILE'].values:
        name, fits_name, night = get_infos_from_txtfiles(txtfile)
        file_name = txtfile.split("/")[-1]
        additional_obs = additional_obs.append(pd.DataFrame({"TXTFILE":[file_name], "MAIN_ID":[name], \
                                                     "FITSFILE":[fits_name], "Night":[night]}), \
                                       ignore_index=True)
additional_obs.to_csv(path_output+"obs_list.csv", mode='a', header=False, index=False)


"""
Now we have the list of the targets and their associated txt files.
We can create the directories of the nights and all the corresponding csv: simbad infos, sparta atm, 
sparta IRLOOP, sparta VISLOOP.
"""

#### Creation of the directories, identified by night:
#### and we populate the directories with the corresponding fitsfiles

# Read csv
obs_list = pd.read_csv(path_output+'/obs_list.csv')

for i in range(len(obs_list)):
    obs_directory = path_output+'sparta_'+obs_list['Night'].values[i]
    print(obs_directory)
    if not os.path.exists(obs_directory):
        try:
            os.mkdir(obs_directory)
        except:
            print(obs_directory, "not created")
    fitsfilename = obs_list['FITSFILE'].values[i]
    if not os.path.exists(os.path.join(obs_directory, fitsfilename)):
        copyfile(os.path.join(path_sparta, fitsfilename), os.path.join(obs_directory, fitsfilename))        
            
            
## Now, we browse the directories and add corresponding csv to them.

#### Simbad csv

#Parameters for the Simbad query:
search_radius = 10*u.arcsec # we search in a 10arcsec circle.
search_radius_alt = 20*u.arcsec # in case nothing is found, we enlarge the search
customSimbad = Simbad()
customSimbad.add_votable_fields('flux(V)','flux(R)','flux(I)','flux(J)','flux(H)',\
                                'flux(K)','id(HD)','sp','otype','otype(V)','otype(3)',\
                               'propermotions')
print('Simbad instance initialized.')
latitude =  -24.6268*u.degree
longitude = -70.4045*u.degree  
altitude = 2648.0*u.meter 
location = coord.EarthLocation(lon=longitude,lat=latitude,height=altitude)

# Parameters for the conversion of atmospheric values:
lam = 500.e-9 # Wavelength for the Sparta seeing and tau0.


## We read the panda dataframe where we'll put all the target we find with Simbad to 
## avoid asking Simbad for the same target twice

# If it doesn't exist, we create it
try:
    catalog_pd = pd.read_csv(os.path.join(path_output,"catalog_simbad_complete.csv"))

except:
    print("WARNING ! catalog_pd not found, we create it.")
    catalog_pd = pd.DataFrame({'MAIN_ID':[],'DEC':[],'RA':[],'simbad_DEC_ICRS':[],\
                       'simbad_DEC_current':[],'simbad_FLUX_H':[],'simbad_FLUX_J':[],\
                       'simbad_FLUX_K':[],'simbad_FLUX_V':[],'simbad_FLUX_R':[],\
                       'simbad_FLUX_I':[],'simbad_ID_HD':[],'simbad_MAIN_ID':[],\
                       'simbad_OTYPE':[],'simbad_OTYPE_3':[],'simbad_OTYPE_V':[],\
                       'simbad_PMDEC':[],'simbad_PMRA':[],'simbad_RA_ICRS':[],\
                       'simbad_RA_current':[],'simbad_SP_TYPE':[],\
                       'simbad_separation_RADEC_ICRSJ2000':[],'simbad_separation_RADEC_current':[]}, dtype='str')
    catalog_pd.to_csv(os.path.join(path_output,"catalog_simbad_complete.csv"))
    
    
    
## We browse all the directories, night by night, to add Simbad csv file in each

start_time = time.time()
list_directories = sorted(os.listdir(path_output))
k=0
for directory in [s for i,s in enumerate(list_directories) if 'sparta' in s]:

    night = directory.split("_")[-1]
    
    df_night = obs_list.loc[obs_list['Night'] == directory.split("_")[-1]]
    simbad_dico = {'MAIN_ID':[],'DEC':[],'RA':[],'simbad_DEC_ICRS':[],\
                   'simbad_DEC_current':[],'simbad_FLUX_H':[],'simbad_FLUX_J':[],\
                   'simbad_FLUX_K':[],'simbad_FLUX_V':[],'simbad_FLUX_R':[],\
                   'simbad_FLUX_I':[],'simbad_ID_HD':[],'simbad_MAIN_ID':[],\
                   'simbad_OTYPE':[],'simbad_OTYPE_3':[],'simbad_OTYPE_V':[],\
                   'simbad_PMDEC':[],'simbad_PMRA':[],'simbad_RA_ICRS':[],\
                   'simbad_RA_current':[],'simbad_SP_TYPE':[],\
                   'simbad_separation_RADEC_ICRSJ2000':[],'simbad_separation_RADEC_current':[]}
    dico_with_night = simbad_dico.copy()
    dico_with_night['Night'] = []
    try:
        simbad_night = pd.read_csv(os.path.join(path_output,directory,"simbad_"+night+".csv"))
    except:
        simbad_night = pd.DataFrame(dico_with_night)
    df_night_reduced = df_night.drop_duplicates(subset='MAIN_ID', keep='first', inplace=False).reset_index(drop=True)
    
    for i in range(len(df_night_reduced)):
        if df_night_reduced['MAIN_ID'].values[i] not in catalog_pd['MAIN_ID'].values:
            fitsfilename = df_night_reduced['TXTFILE'].values[i].replace('NL.txt','fits')
            simbad_dico = get_star_properties_from_simbad(path_output+os.path.join(directory,fitsfilename), df_night_reduced['MAIN_ID'].values[i])
            catalog_pd = update_catalog(simbad_dico,df_night_reduced['MAIN_ID'].values[i])
            catalog_pd.to_csv(os.path.join(path_output,"catalog_simbad_complete.csv"),index=False)
        else:
            print(df_night_reduced['MAIN_ID'].values[i], 'found in the main catalog')
        if df_night_reduced['MAIN_ID'].values[i] not in simbad_night['MAIN_ID'].values:
            simbad_night = simbad_night.append(catalog_pd.loc[catalog_pd['MAIN_ID']==df_night_reduced['MAIN_ID'].values[i]], ignore_index=True)
        else:
            print(df_night_reduced['MAIN_ID'].values[i], 'found in the simbad night catalog')
    simbad_night = simbad_night.assign(Night=night)
    simbad_night.to_csv(os.path.join(path_output,directory,"simbad_"+night+".csv"),index=False)
    
print("\n --- SIMBAD querying time: %s seconds ---" % (time.time() - start_time))    



## Calculation and saving of asm data
### We query all the asm data from 2015-01-01 and save it in different csv files corresponding to each instrument

start_time = time.time()
yesterday = datetime.datetime.now() + datetime.timedelta(days=-1)
yesterday = datetime.datetime.strftime(yesterday,"%Y-%m-%d")
query_asm('2015-01-01',yesterday, asm=False, mass_dimm=False, dimm=False, old_dimm=True,\
              lhatpro=False, sphere_ambi=False, slodar=False, scidar=False, path = path_output)#yesterday)

print("\n --- ASM querying time: %s seconds ---" % (time.time() - start_time))    
    
# Read csv
obs_list = pd.read_csv(path_output+'/obs_list.csv')
start_time = time.time()

list_directories = sorted(os.listdir(path_output))
k=0
for directory in [s for i,s in enumerate(list_directories) if 'sparta' in s]:
    night = directory.split("_")[-1]
    print("\n\nTHE NIGHT IS:", night)
    df_night = obs_list.loc[obs_list['Night'] == night]
    df_night = df_night.reset_index(drop=True)
    
    SCIENCE,ATM,VIS,IR,DTTS = True,True,True,True,True
    
    try:
        ScienceParams = pd.read_csv(os.path.join(path_output,directory,'ScienceParams_{}.csv'.format(night)))
        print("SCIENCE csv loaded")
        SCIENCE = False
    except:
        ScienceParams = pd.DataFrame({})
        print("We'll create SCIENCE csv file")
        
    try:
        AtmParams = pd.read_csv(os.path.join(path_output,directory,'AtmParams_{}.csv'.format(night)))
        print("ATM csv loaded")
        ATM = False
    except:
        AtmParams = pd.DataFrame({'TXTFILE':[],'MAIN_ID':[],'@timestamp': [], \
                                    'SPARTA STREHL':[],'SPARTA WINDSPEED': [],'SPARTA SEEING OBS':[],\
                                    'SPARTA R0 OBS':[],'SPARTA TAU0 OBS':[],'TEL ALT':[],'AIRMASS':[],\
                                    'SPARTA SEEING':[],'SPARTA R0':[],'SPARTA TAU0':[] })
        print("We'll create ATM csv file")
        
        
    try:
        VisLoopParams = pd.read_csv(os.path.join(path_output,directory,'VisLoopParams_{}.csv'.format(night)))
        print("VisLoop csv loaded")
        VIS = False
    except:
        VisLoopParams = pd.DataFrame({'TXTFILE':[],'MAIN_ID':[],'@timestamp': [], \
                                    'FLUX_avg':[], 'FLUX_std':[]})                
        print("We'll create VIS csv file")       
        
        
    try:    
        IRLoopParams = pd.read_csv(os.path.join(path_output,directory,'IRLoopParams_{}.csv'.format(night)))
        print("IRLoop csv loaded")
        IR = False
    except:
        IRLoopParams = pd.DataFrame({'TXTFILE':[],'MAIN_ID':[],'@timestamp': [], \
                                    'FLUX_avg':[],'FLUX_std':[],'DTTPPos_avg':[],'DTTPPos_std':[],'DTTPRes_avg':[],\
                                    'DTTPRes_std':[]})
        print("We'll create IR csv file")
    
    try:
        DTTSParams = pd.read_csv(os.path.join(path_output,directory,'DTTSParams_{}.csv'.format(night)))
        print("DTTS csv loaded")
        DTTS = False
    except:
        DTTSParams = pd.DataFrame({'TXTFILE':[],'MAIN_ID':[],'@timestamp': [], \
                                    'LWE_strength':[], 'SNR':[],'AMP':[]})
        print("We'll create DTTS csv file")
        
    if SCIENCE or ATM or VIS or IR or DTTS: 
        for i in range(len(df_night)):

            fitsfile = df_night['FITSFILE'].values[i]
            temp_ScienceParams, temp_AtmParams, temp_VisLoopParams, temp_IRLoopParams,temp_DTTSParams = populate_AtmPerfParams_df(path_sparta+fitsfile, df_night.loc[i], SCIENCE=SCIENCE, ATM=ATM, VIS=VIS, IR=IR, DTTS=DTTS)
            
            ScienceParams = ScienceParams.append(temp_ScienceParams, ignore_index=True)
            AtmParams = AtmParams.append(temp_AtmParams, ignore_index=True)
            VisLoopParams = VisLoopParams.append(temp_VisLoopParams, ignore_index=True)
            IRLoopParams = IRLoopParams.append(temp_IRLoopParams, ignore_index=True)
            DTTSParams = DTTSParams.append(temp_DTTSParams, ignore_index=True)
            
        ScienceParams.to_csv(os.path.join(path_output,directory,'ScienceParams_{}.csv'.format(night)),index=False)    
        AtmParams.to_csv(os.path.join(path_output,directory,'AtmParams_{}.csv'.format(night)),index=False)
        VisLoopParams.to_csv(os.path.join(path_output,directory,'VisLoopParams_{}.csv'.format(night)),index=False)
        IRLoopParams.to_csv(os.path.join(path_output,directory,'IRLoopParams_{}.csv'.format(night)),index=False)
        DTTSParams.to_csv(os.path.join(path_output,directory,'DTTSParams_{}.csv'.format(night)),index=False)
    else:
        print("csv files already exist, we don't look for data")

print("--- %s seconds ---" % (time.time() - start_time))
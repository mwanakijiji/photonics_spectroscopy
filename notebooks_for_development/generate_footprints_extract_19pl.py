#!/usr/bin/env python
# coding: utf-8

# makes footprints for 19-port PL

import glob as glob
import pandas as pd
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.sparse.linalg import lsmr
from utils2 import fake_19pl_data, basic_fcns
import cv2
import scipy

#####
'''USER INPUTS'''
# 'empirical': empirical 19PL data; acts as reference frame
# 'empirical_translated': empirical 19PL data, with translation (already dark-subtracted)
# 'fake_profiles': extraction profiles themselves (for testing)
# 'fake_injected': fake data with injected spectra
data_choice = 'fake_injected'

len_spec = 250 # length of spectra and their profiles, in pixels (x only for now) ## ## expand to enable arbitrary shapes
sigma_all = 1 # sigma width of extraction profiles (and fake spectra, if applicable)
#####

stem = '/Users/bandari/Documents/git.repos/photonics_spectroscopy/notebooks_for_development/data/19pl/'

# read in a test frame / dark
dark_apapane = stem + 'raw/apapanedark.fits'
test_array = fits.open(dark_apapane)[0].data
dark = test_array

x_extent = np.shape(test_array)[1]
y_extent = np.shape(test_array)[0]

# read in a broadband frame (already dark-subtracted)
# (this will only get used if applicable)
bb_frame_19pl = stem + 'dark_subted/19PL_bb_irnd1.0_optnd3.0.fits'
bb_array = fits.open(bb_frame_19pl)[0].data

# read in a translated broadband frame (already dark-subtracted)
# (this will only get used if applicable)
bb_frame_19pl_translated = stem + 'dark_subted/19PL_bb_irnd1.0_optnd3.0_translated.fits'
bb_array_translated = fits.open(bb_frame_19pl)[0].data

# ersatz variance frame
hdul = fits.open('./data/stacked.fits')
test_data = hdul[0].data[0,:,:] # all 3 spectra
array_variance = hdul[0].data[1,:,:]

# replace zeros
array_variance[array_variance == 0] = np.median(array_variance)

# relative positions of all the outputs 
# (can move around with a common offset in (x,y))

# 19 PL starting positions (x,y) from a narrowband frame, translated
# so that x=0,y=0 corresponds to leftmost part of leftmost lenslet
# (note these are RELATIVE (x,y) positions; the important thing is that they define the starting positions of the spectra)
rel_pos = {'0':(39,81),
           '1':(29,115),
           '2':(38,113),
           '3':(46,111),
           '4':(23,108),
           '5':(32,106),
           '6':(40,105),
           '7':(49,103),
           '8':(17,101),
           '9':(26,100),
           '10':(34,98),
           '11':(43,96),
           '12':(51,94),
           '13':(20,93),
           '14':(28,91),
           '15':(37,90),
           '16':(45,88),
           '17':(22,84),
           '18':(31,83)}

# dict to hold the extracted spectra, 19PL
eta_flux = {'0':np.zeros(x_extent),
    '1':np.zeros(x_extent),
    '2':np.zeros(x_extent),
    '3':np.zeros(x_extent),
    '4':np.zeros(x_extent),
    '5':np.zeros(x_extent),
    '6':np.zeros(x_extent),
    '7':np.zeros(x_extent),
    '8':np.zeros(x_extent),
    '9':np.zeros(x_extent),
    '10':np.zeros(x_extent),
    '11':np.zeros(x_extent),
    '12':np.zeros(x_extent),
    '13':np.zeros(x_extent),
    '14':np.zeros(x_extent),
    '15':np.zeros(x_extent),
    '16':np.zeros(x_extent),
    '17':np.zeros(x_extent),
    '18':np.zeros(x_extent)}

# dicts to hold the extracted spectrum x-values on detector
#eta_x = {}
#eta_y = {}

# dict to hold the wavel solution for each extracted spectrum
eta_wavel = {}

# will hold wavelength solutions, one for each of the 19 ports
wavel_soln_ports = {}

# read in wavelength solution as a function of displacement from an arbitrary point
df_wavel_soln_shift = pd.read_pickle('soln_wavelength_xy_shift_20230612.pkl') # valid in (1005, 1750)
import ipdb; ipdb.set_trace()
# sort wavelengths
df_wavel_soln_shift.sort_values(by='wavel', inplace=True)
## ## OVERWRITE ABOVE and make a cleaner wavelength soln made to imitate the one I found
m_slope = (3.9-0)/(163+143)
b_int = 143*m_slope
df_wavel_soln_shift['y_shift'] = df_wavel_soln_shift['x_shift']*m_slope + b_int

# subtract offset in x
df_wavel_soln_shift['x_shift_zeroed'] = df_wavel_soln_shift['x_shift']-np.min(df_wavel_soln_shift['x_shift'])
df_wavel_soln_shift['y_shift_zeroed'] = df_wavel_soln_shift['y_shift']-np.min(df_wavel_soln_shift['y_shift'])
df_wavel_soln_shift = df_wavel_soln_shift.astype({'wavel':'float'}) # these are read in as strings

# make canvas_array on which profiles will be placed
canvas_array = np.zeros(np.shape(test_array))
dict_profiles = {}

'''
# pickle something
import pickle
canvas_array2 = np.zeros(np.shape(test_array))
for key, coord_xy in rel_pos.items():
    canvas_array2 += dict_profiles[key]
with open('junk.pkl', 'wb') as handle:
    pickle.dump(canvas_array2, handle)
'''

# define detector array D and any shift in (x,y) we have to account for for placing the profiles
if data_choice == 'empirical':
    # real 19PL data
    D = bb_array 
    y_shift, x_shift = 0., 0.

elif data_choice == 'empirical_translated':
    height, width = bb_array.shape[:2]
    T = np.float32([[1, 0, 4.3], [0, 1, 2.8]]) # third elements are sizes of translation, in pixels
    tonight_frame = cv2.warpAffine(bb_array-dark, T, (width, height)) # dark-subtn apparently necessary to avoid numerical issues
    D = tonight_frame
    # find shift relative to reference frame
    ref_frame = bb_array
    corr = scipy.signal.correlate2d(ref_frame, tonight_frame, boundary='symm', mode='same')
    y_abs, x_abs = np.unravel_index(np.argmax(corr), corr.shape)  # y, x: tonight's image is displaced by this much from the reference
    y_shift, x_shift = 0.5*height-y_abs,0.5*width-x_abs

elif data_choice == 'fake_injected':
    # fake injected spectra
    test_spec_data = fake_19pl_data.fake_injected(shape_pass = np.shape(canvas_array), rel_pos_pass = rel_pos, sigma = sigma_all)
    D = test_spec_data
    y_shift, x_shift = 0., 0.

elif data_choice == 'fake_profiles':
    # simple profiles themselves are treated as spectra
    # load a pickled frame
    #with open('fake_simple_profiles_width0pt25.pkl', 'rb') as handle: # don't forget to set sigma_pass in basic_fcns.simple_profile()
    #with open('fake_simple_profiles_width0pt5.pkl', 'rb') as handle:
    #with open('fake_simple_profiles_width1pt0.pkl', 'rb') as handle:
    #with open('fake_simple_profiles_width2pt0.pkl', 'rb') as handle:
    with open('fake_simple_profiles_width3pt0.pkl', 'rb') as handle:
        D = pickle.load(handle)
    y_shift, x_shift = 0., 0.

import ipdb; ipdb.set_trace()

# loop over each spectrum's starting position and 
# 1. generate a full spectrum profile
# 2. calculate a wavelength solution
for key, coord_xy in rel_pos.items():

    spec_x_left = np.add(coord_xy[0],-x_shift)
    spec_y_left = np.add(coord_xy[1],-y_shift)

    # place profile on detector, while removing translation of frame relative to a reference frame
    profile_this_array = basic_fcns.simple_profile(array_shape=np.shape(test_array), 
                                x_left=spec_x_left, 
                                y_left=spec_y_left, 
                                len_spec=len_spec, 
                                sigma_pass=sigma_all)
    
    # accumulate these onto an array that will let us look at the total footprint
    canvas_array += profile_this_array

    # save single profiles in an array
    dict_profiles[key] = profile_this_array

    ## calculate wavelength soln for this one spectrum 

    # note we account for the image translation here by
    # 1. adding values x_shift, y_shift to the spectrum starting points, and
    # 2. adding those resulting coordinates to the zeroed x,y of the wavelength soln footprint (i.e., the starting (x,y)=(0,0))

    # adjust (x,y) appropriately

    '''
    df_wavel_soln_shift['x_abs_spec_' + str(key)] = np.add(np.add(coord_xy[0],-x_shift),df_wavel_soln_shift['x_shift_zeroed'])
    df_wavel_soln_shift['y_abs_spec_' + str(key)] = np.add(np.add(coord_xy[1],-y_shift),df_wavel_soln_shift['y_shift_zeroed'])

    # values only
    x_pix_locs = df_wavel_soln_shift['x_abs_spec_' + str(key)].values

    # test vals here for testing; just single value
    #y_pix_locs = df_wavel_soln_shift['y_abs_spec_' + str(key)].values[0]*np.ones(len( df_wavel_soln_shift['y_abs_spec_' + str(key)])) # one val only
    
    y_pix_locs = df_wavel_soln_shift['y_abs_spec_' + str(key)].values
    '''

    x_pix_locs = df_wavel_soln_shift['x_shift_zeroed']
    y_pix_locs = df_wavel_soln_shift['y_shift_zeroed']

    # solve
    import ipdb; ipdb.set_trace()
    ## ## THIS SHOULD BE DONE AT THE BEGINNING SOMEHOW, TO AVOID RE-DOING IT EACH TIME

    # fit coefficients based on (x,y) coords of given spectrum and the set of basis wavelengths
    fit_coeffs = basic_fcns.find_coeffs(
        x_pix_locs,
        y_pix_locs,
        df_wavel_soln_shift['wavel'].values
        )
    
    #plt.clf()

    #plt.plot(x_pix_locs,df_wavel_soln_shift['wavel'].values)
    #plt.show()

    # put x,y values of trace into dicts
    #eta_x[str(key)] = np.arange(np.shape(canvas_array)[1])
    ## ## PLACEHOLDER FOR NOW; CHANGE LATER TO ALLOW MULTIPLE VALUES OF Y
    #eta_y[str(key)] = np.add(coord_xy[1],-y_shift)
    
    # put best-fit coeffs in dictionary
    wavel_soln_ports[str(key)] = fit_coeffs

    # best-fit wavel values (not really important here, since abcissa is just the sampling from the narrowband data, but useful for checking)
    df_wavel_soln_shift['wavel_bestfit_' + str(key)] = basic_fcns.wavel_from_func((x_pix_locs,y_pix_locs), 
                                                                                  fit_coeffs[0], 
                                                                                  fit_coeffs[1], 
                                                                                  fit_coeffs[2], 
                                                                                  fit_coeffs[3], 
                                                                                  fit_coeffs[4])
    #plt.plot(x_pix_locs,df_wavel_soln_shift['wavel'].values)


# check overlap is good
'''
plt.imshow(np.add((1e4)*canvas_array,bb_array), origin='lower')
plt.title('profiles + data to check overlap')
plt.show()
#plt.savefig('junk_overlap.png')
'''

# fake data for testing (this treats the simple profiles as spectra)
bb_fake = canvas_array

#plt.imshow(bb_fake)
#plt.show()

#### 19 spectra extraction

# loop over detector cols
for col in range(0,x_extent): 
    
    # initialize matrices; we will solve for
    # c_mat.T * x.T = b_mat.T to get x
    c_mat = np.zeros((len(eta_flux),len(eta_flux)), dtype='float')
    b_mat = np.zeros((len(eta_flux)), dtype='float')

    # loop over pixels in col
    for pix_num in range(0,y_extent):
        
        # nested loop over matrix elements by (row, col)
        # there's doubtlessly a faster way to do this
        for mat_row in range(0,len(dict_profiles)):
                
            # these are done all at once for speed; there must be a better way
            c_mat[mat_row][0] += dict_profiles[str(int(mat_row))][pix_num,col] * dict_profiles['0'][pix_num,col] / array_variance[pix_num,col]
            c_mat[mat_row][1] += dict_profiles[str(int(mat_row))][pix_num,col] * dict_profiles['1'][pix_num,col] / array_variance[pix_num,col]
            c_mat[mat_row][2] += dict_profiles[str(int(mat_row))][pix_num,col] * dict_profiles['2'][pix_num,col] / array_variance[pix_num,col]
            c_mat[mat_row][3] += dict_profiles[str(int(mat_row))][pix_num,col] * dict_profiles['3'][pix_num,col] / array_variance[pix_num,col]
            c_mat[mat_row][4] += dict_profiles[str(int(mat_row))][pix_num,col] * dict_profiles['4'][pix_num,col] / array_variance[pix_num,col]
            c_mat[mat_row][5] += dict_profiles[str(int(mat_row))][pix_num,col] * dict_profiles['5'][pix_num,col] / array_variance[pix_num,col]
            c_mat[mat_row][6] += dict_profiles[str(int(mat_row))][pix_num,col] * dict_profiles['6'][pix_num,col] / array_variance[pix_num,col]
            c_mat[mat_row][7] += dict_profiles[str(int(mat_row))][pix_num,col] * dict_profiles['7'][pix_num,col] / array_variance[pix_num,col]
            c_mat[mat_row][8] += dict_profiles[str(int(mat_row))][pix_num,col] * dict_profiles['8'][pix_num,col] / array_variance[pix_num,col]
            c_mat[mat_row][9] += dict_profiles[str(int(mat_row))][pix_num,col] * dict_profiles['9'][pix_num,col] / array_variance[pix_num,col]
            c_mat[mat_row][10] += dict_profiles[str(int(mat_row))][pix_num,col] * dict_profiles['10'][pix_num,col] / array_variance[pix_num,col]
            c_mat[mat_row][11] += dict_profiles[str(int(mat_row))][pix_num,col] * dict_profiles['11'][pix_num,col] / array_variance[pix_num,col]
            c_mat[mat_row][12] += dict_profiles[str(int(mat_row))][pix_num,col] * dict_profiles['12'][pix_num,col] / array_variance[pix_num,col]
            c_mat[mat_row][13] += dict_profiles[str(int(mat_row))][pix_num,col] * dict_profiles['13'][pix_num,col] / array_variance[pix_num,col]
            c_mat[mat_row][14] += dict_profiles[str(int(mat_row))][pix_num,col] * dict_profiles['14'][pix_num,col] / array_variance[pix_num,col]
            c_mat[mat_row][15] += dict_profiles[str(int(mat_row))][pix_num,col] * dict_profiles['15'][pix_num,col] / array_variance[pix_num,col]
            c_mat[mat_row][16] += dict_profiles[str(int(mat_row))][pix_num,col] * dict_profiles['16'][pix_num,col] / array_variance[pix_num,col]
            c_mat[mat_row][17] += dict_profiles[str(int(mat_row))][pix_num,col] * dict_profiles['17'][pix_num,col] / array_variance[pix_num,col]
            c_mat[mat_row][18] += dict_profiles[str(int(mat_row))][pix_num,col] * dict_profiles['18'][pix_num,col] / array_variance[pix_num,col]

            # b_mat is just 1D, so use mat_row as index
            b_mat[mat_row] += D[pix_num,col] * dict_profiles[str(int(mat_row))][pix_num,col] / array_variance[pix_num,col]
    
    # solve for the following transform:
    # x * c_mat = b_mat  -->  c_mat.T * x.T = b_mat.T
    eta_flux_mat_T, istop, itn, normr, normar, norma, conda, normx = \
            lsmr(c_mat.transpose(), b_mat.transpose())
    
    eta_flux_mat =  eta_flux_mat_T.transpose()
    
    for eta_flux_num in range(0,len(eta_flux)):
        eta_flux[str(eta_flux_num)][col] = eta_flux_mat[eta_flux_num]

import ipdb; ipdb.set_trace()
# apply wavelength solutions
for key, coord_xy in rel_pos.items():
    # note the (x,y) coordinates stretch over entire detector, not just the region sampled for the wavelength soln

    #import ipdb; ipdb.set_trace()

    # indices of pixels across entire detector, in x
    x_pix_locs_whole_detector = np.arange(np.shape(canvas_array)[1]) ## ## MAKE MORE ACCURATE LATER
    ## ?
    y_pix_locs_whole_detector = np.add(coord_xy[1],-y_shift)*np.ones(len(x_pix_locs_whole_detector)) ## ## MAKE MORE ACCURATE LATER
    coeffs_this = wavel_soln_ports[str(key)]
    import ipdb; ipdb.set_trace()

    ##
    eta_flux[str(key)] 
    ##

    x_pix_whole_spectrum = np.arange(0, len_spec, 1) # rel_pos[str(key)][0],
    y_pix_whole_spectrum = np.arange(0, len_spec, 1)

    # return wavelengths from pre-existing wavelength soln
    ## ## PROBLEN LIES AT THIS STEP; SOME SOLUTIONS GO NEGATIVE
    eta_wavel[str(key)] = basic_fcns.wavel_from_func((x_pix_locs_whole_detector,y_pix_locs_whole_detector), 
                                                     coeffs_this[0], coeffs_this[1], coeffs_this[2],
                                                     coeffs_this[3], coeffs_this[4])
import ipdb; ipdb.set_trace()
'''
for key, coord_xy in rel_pos.items():
    plt.plot(eta_wavel[str(key)],eta_flux[str(key)]+2000*float(key))
plt.show()
'''

# plots for a given detector array


'''
plt.scatter(
'''

# check overlap of spectra and profiles

plt.imshow(D+canvas_array, origin='lower')
plt.title('D+canvas_array')
plt.show()


# check c_matrix
#plt.imshow(c_mat)
#plt.colorbar()
#plt.show()

# overplot wavelength solns
for i in range(0,len(eta_flux)):
    plt.plot(eta_wavel[str(i)])
plt.show()



x_talk_mean = {}
x_talk_median = {}
x_talk_stdev = {}
# tests on the extracted spectra
for i in range(0,len(eta_flux)):
    # indices we will use to quantify cross-talk
    idx_xtalk = np.logical_and(eta_wavel[str(i)]>1150,eta_wavel[str(i)]<1400)

    x_talk_mean[str(i)] = np.mean(eta_flux[str(i)][idx_xtalk])
    x_talk_median[str(i)] = np.median(eta_flux[str(i)][idx_xtalk])
    x_talk_stdev[str(i)] = np.std(eta_flux[str(i)][idx_xtalk])


# xtalk plots
for i in range(0,len(eta_flux)):
    plt.scatter(i,x_talk_mean[str(i)])
plt.title('mean')
plt.savefig('junk_mean.png')
plt.clf()
for i in range(0,len(eta_flux)):
    plt.scatter(i,x_talk_median[str(i)])
plt.title('median')
plt.savefig('junk_median.png')
plt.clf()
for i in range(0,len(eta_flux)):
    plt.scatter(i,x_talk_stdev[str(i)])
plt.title('stdev')
plt.savefig('junk_stdev.png')
plt.clf()


# check cross-talk: offset retrievals
for i in range(0,len(eta_flux)):
    plt.plot(eta_wavel[str(i)], np.add(eta_flux[str(i)],0.1*i))
    plt.annotate(str(i), (1100,1+0.1*i), xytext=None)
plt.title('retrievals+offsets')
import ipdb; ipdb.set_trace()
plt.savefig('junk_offset_retrievals.png')


# check cross-talk: compare retrievals, truth
'''
plt.clf()
plt.plot(eta_flux[str(i)][coord_xy[0]:coord_xy[0]+100], label='retrieved')
plt.plot(np.array(spec_fake['flux_norm'][0:100]), label='truth')
plt.legend()
plt.title('retrievals and truth (assumed to be stellar test spectrum)')
plt.show()

# check cross-talk: residuals
for i in range(0,len(eta_flux)):
    resids = eta_flux[str(i)][coord_xy[0]:coord_xy[0]+100] - np.array(spec_fake['flux_norm'][0:100])
    #test_data[coord_xy[1],coord_xy[0]:coord_xy[0]+100] = np.array(spec_fake['flux_norm'][0:100])
    plt.plot(resids)
plt.title('residuals')
plt.show()

## plots
# check fake spectra, with offsets
for i in range(0,len(eta_flux)):
    plt.plot(np.add(eta_flux[str(i)],0.1*i))
    plt.annotate(str(i), (0,0.1*i), xytext=None)
plt.title('retrieved spectra, with offsets')
plt.savefig('junk_19specs2.png')
plt.show()

# check fake spectra, without offsets
for i in range(0,len(eta_flux)):
    plt.plot(eta_flux[str(i)])
plt.title('retrieved spectra, without offsets')
plt.show()
'''


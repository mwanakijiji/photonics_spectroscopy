#!/usr/bin/env python
# coding: utf-8

# makes footprints for 19-port PL

import glob as glob
import pandas as pd
import os
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
data_choice = 'empirical_translated'
#####

stem = '/Users/bandari/Documents/git.repos/photonics_spectroscopy/notebooks_for_development/data/19pl/'

# read in a test frame / dark
dark_apapane = stem + 'raw/apapanedark.fits'
test_array = fits.open(dark_apapane)[0].data
dark = test_array

x_extent = np.shape(test_array)[1]
y_extent = np.shape(test_array)[0]

# read in a broadband frame (already dark-subtracted)
bb_frame_19pl = stem + 'dark_subted/19PL_bb_irnd1.0_optnd3.0.fits'
bb_array = fits.open(bb_frame_19pl)[0].data

# read in a translated broadband frame (already dark-subtracted)
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

# actual 19 PL starting positions (x,y)
rel_pos = {'0':(203,81),
           '1':(193,115),
           '2':(202,113),
           '3':(210,111),
           '4':(187,108),
           '5':(196,106),
           '6':(204,105),
           '7':(213,103),
           '8':(181,101),
           '9':(190,100),
           '10':(198,98),
           '11':(207,96),
           '12':(215,94),
           '13':(184,93),
           '14':(192,91),
           '15':(201,90),
           '16':(209,88),
           '17':(186,84),
           '18':(195,83)}

# dict to hold the extracted spectra, 19PL
eta = {'0':np.zeros(x_extent),
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

# grid of distances between profiles (for measuring effect of changing distance from each other)
#for prox_pixel in range(30,31):#30,2):

# make canvas_array on which profiles will be placed
canvas_array = np.zeros(np.shape(test_array))
x_offset = 0 #17-181 for true positions
y_offset = 0 #-1 for true positions
dict_profiles = {}

# loop over each spectrum's starting position and generate a profile
for key, coord_xy in rel_pos.items():

    profile_this_array = basic_fcns.simple_profile(array_shape=np.shape(test_array), 
                                x_left=np.add(coord_xy[0],x_offset), 
                                y_left=np.add(coord_xy[1],y_offset), 
                                len_spec=250, 
                                sigma_pass=1)
    
    canvas_array += profile_this_array
    
    # save single profiles in an array
    dict_profiles[key] = profile_this_array

# check overlap is good

plt.imshow(np.add((1e4)*canvas_array,bb_array), origin='lower')
plt.show()
#plt.savefig('junk_overlap.png')


# fake data for testing (this treats the simple profiles as spectra)
bb_fake = canvas_array

#plt.imshow(bb_fake)
#plt.show()

#### 19 spectra extraction

# define detector array D
if data_choice == 'empirical':
    # real 19PL data
    D = bb_array 
    y_shift, x_shift = 0., 0.

elif data_choice == 'empirical_translated':
    # fake injected spectra
    empirical_trans = fake_19pl_data.fake_injected(shape_pass = np.shape(canvas_array), rel_pos_pass = rel_pos)
    D = empirical_trans
    # find shift relative to reference frame
    ref_frame = bb_array
    tonight_frame = D
    corr = scipy.signal.correlate2d(ref_frame, tonight_frame, boundary='symm', mode='same')
    y_shift, x_shift = np.unravel_index(np.argmax(corr), corr.shape)  # y, x: tonight's image is displaced by this much from the reference

elif data_choice == 'fake_injected':
    # fake injected spectra
    test_spec_data = fake_19pl_data.fake_injected(shape_pass = np.shape(canvas_array), rel_pos_pass = rel_pos)
    D = test_spec_data
    y_shift, x_shift = 0., 0.

elif data_choice == 'fake_profiles':
    # simple profiles themselves are treated as spectra
    D = bb_fake
    y_shift, x_shift = 0., 0.


# loop over detector cols
for col in range(0,x_extent): 
    
    # initialize matrices; we will solve for
    # c_mat.T * x.T = b_mat.T to get x
    c_mat = np.zeros((len(eta),len(eta)), dtype='float')
    b_mat = np.zeros((len(eta)), dtype='float')

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
    eta_mat_T, istop, itn, normr, normar, norma, conda, normx = \
            lsmr(c_mat.transpose(), b_mat.transpose())
    
    eta_mat =  eta_mat_T.transpose()
    
    for eta_num in range(0,len(eta)):
        eta[str(eta_num)][col] = eta_mat[eta_num]


# plots for a given detector array

# check overlap of spectra and profiles
plt.imshow(D+canvas_array, origin='lower')
plt.title('D+canvas_array')
plt.show()

# check c_matrix
#plt.imshow(c_mat)
#plt.colorbar()
#plt.show()

# check cross-talk: offset retrievals
for i in range(0,len(eta)):
    plt.plot(np.add(eta[str(i)],0.1*i))
plt.title('retrievals+offsets')
plt.show()

# check cross-talk: compare retrievals, truth
'''
plt.clf()
plt.plot(eta[str(i)][coord_xy[0]:coord_xy[0]+100], label='retrieved')
plt.plot(np.array(spec_fake['flux_norm'][0:100]), label='truth')
plt.legend()
plt.title('retrievals and truth (assumed to be stellar test spectrum)')
plt.show()

# check cross-talk: residuals
for i in range(0,len(eta)):
    resids = eta[str(i)][coord_xy[0]:coord_xy[0]+100] - np.array(spec_fake['flux_norm'][0:100])
    #test_data[coord_xy[1],coord_xy[0]:coord_xy[0]+100] = np.array(spec_fake['flux_norm'][0:100])
    plt.plot(resids)
plt.title('residuals')
plt.show()

## plots
# check fake spectra, with offsets
for i in range(0,len(eta)):
    plt.plot(np.add(eta[str(i)],0.1*i))
    plt.annotate(str(i), (0,0.1*i), xytext=None)
plt.title('retrieved spectra, with offsets')
plt.savefig('junk_19specs2.png')
plt.show()

# check fake spectra, without offsets
for i in range(0,len(eta)):
    plt.plot(eta[str(i)])
plt.title('retrieved spectra, without offsets')
plt.show()
'''


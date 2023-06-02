#!/usr/bin/env python
# coding: utf-8

# makes footprints for 19-port PL

import glob as glob
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.sparse.linalg import lsmr

stem = '/Users/bandari/Documents/git.repos/photonics_spectroscopy/notebooks_for_development/data/19pl/'

# read in a test frame / dark
dark_apapane = stem + 'raw/apapanedark.fits'
test_array = fits.open(dark_apapane)[0].data
dark = test_array

# read in a broadband frame
raw_bb_frame_19pl = stem + 'dark_subted/19PL_bb_irnd1.0_optnd3.0.fits'
bb_array = fits.open(raw_bb_frame_19pl)[0].data

# dark subtract
bb_array = np.subtract(bb_array,dark)

# ersatz variance frame
hdul = fits.open('./data/stacked.fits')
test_data = hdul[0].data[0,:,:] # all 3 spectra
array_variance = hdul[0].data[1,:,:]

# replace zeros
array_variance[array_variance == 0] = np.median(array_variance)

# relative positions of all the outputs 
# (can move around with a common offset in (x,y))

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

# gaussian profile (kind of confusing: coordinates are (lambda, x), instead of (x,y) )
def gaus1d(x_left, len_spec, x_pass, lambda_pass, mu_pass, sigma_pass=1):
    '''
    x_left: x coord of leftmost pixel of spectrum (y coord is assumed to be mu_pass)
    len_spec: length of spectrum [pix]
    x_pass: grid of y-coords in coord system of input
    lambda_pass: grid of x-coords in coord system of input
    mu_pass: profile center (in x_pass coords)
    sigma_pass: profile width (in x_pass coords)
    '''
    
    # condition for lambda axis to be inside footprint
    lambda_cond = np.logical_and(lambda_pass >= x_left, lambda_pass < x_left+len_spec)
    
    #plt.imshow(lambda_cond)
    #plt.show()
    
    # profile spanning entire array
    profile = (1./(sigma_pass*np.sqrt(2.*np.pi))) * np.exp(-0.5 * np.power((x_pass-mu_pass)/sigma_pass,2.) )
    
    # mask regions where there is zero signal
    profile *= lambda_cond
    
    # normalize columns of nonzero signal
    profile = np.divide(profile, np.nanmax(profile))
    
    # restore regions of zero signal as zeros (instead of False)
    profile[~lambda_cond] = 0.
    
    return profile


# wrapper to make the enclosing profile of a spectrum
def simple_profile(array_shape, x_left, y_left, len_spec, sigma_pass=1):
    # make simple 1D Gaussian profile in x-direction
    '''
    shape_array: shape of array
    x_left: x-coord of leftmost point of spectrum
    y_left: y-coord of leftmost point of spectrum
    len_spec: length of spectrum (in x-dir)
    sigma_pass: sigma width of profile
    '''
    #(x_left, len_spec, x_pass, mu_pass, sigma_pass)
    array_profile = np.zeros(array_shape)

    xgrid, ygrid = np.meshgrid(np.arange(0,np.shape(array_profile)[1]),np.arange(0,np.shape(array_profile)[0]))
    array_profile = gaus1d(x_left=x_left, len_spec=len_spec, x_pass=ygrid, lambda_pass=xgrid, mu_pass=y_left, sigma_pass=1)

    #plt.imshow(array_profile)
    #plt.show()
    
    # normalize it such that the marginalization in x (in (x,lambda) space) is 1
    # (with a perfect Gaussian profile in x this is redundant)
    
    test2 = array_profile[:,x_left:x_left+len_spec]
    
    array_profile[:,x_left:x_left+len_spec] = np.divide(array_profile[:,x_left:x_left+len_spec],np.sum(array_profile[:,x_left:x_left+len_spec], axis=0))
    
    return array_profile

# make canvas_array
canvas_array = np.zeros(np.shape(test_array))
x_offset = 17-181
y_offset = -1
dict_profiles = {}
# loop over each spectrum's starting position
#for coord_xy in rel_pos.values():
for key, coord_xy in rel_pos.items():
    
    print(key, coord_xy)

    profile_this_array = simple_profile(array_shape=np.shape(test_array), 
                                x_left=np.add(coord_xy[0],x_offset), 
                                y_left=np.add(coord_xy[1],y_offset), 
                                len_spec=8, 
                                sigma_pass=1)
    
    canvas_array += profile_this_array
    
    # save single profiles in an array
    dict_profiles[key] = profile_this_array

# check overlap is good
'''
plt.imshow(np.add((1e4)*canvas_array,bb_array), origin='lower')
plt.savefig('junk_overlap.png')
'''

# fake data for testing (just simple profiles)
bb_fake = canvas_array

#plt.imshow(bb_fake)
#plt.show()


# 19 spectra extraction

# define test data
#D = bb_array
D = bb_fake

# extent of detector in x-dir
x_extent = np.shape(test_array)[1]
y_extent = np.shape(test_array)[0]

# initialize dict to hold flux for each spectrum
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

# loop over detector cols
for col in range(0,x_extent): 
    
    # initialize matrix
    c_mat = np.zeros((19,19), dtype='float')
    b_mat = np.zeros((19), dtype='float')

    # loop over pixels in col
    for pix_num in range(0,y_extent):
        
        # nested loop over matrix elements by (row, col)
        # there's doubtlessly a faster way to do this
        for mat_row in range(0,len(dict_profiles)):
            for mat_col in range(0,len(dict_profiles)):
                
                c_mat[mat_row][mat_col] += dict_profiles[str(int(mat_row))][pix_num,col] * dict_profiles[str(int(mat_col))][pix_num,col] / array_variance[pix_num,col]

            # b_mat is just 1D, so use mat_row as index
            b_mat[mat_row] += D[pix_num,col] * dict_profiles[str(int(mat_row))][pix_num,col] / array_variance[pix_num,col]
    
    # solve for the following transform:
    # x * c_mat = b_mat  -->  c_mat.T * x.T = b_mat.T
    eta_mat_T, istop, itn, normr, normar, norma, conda, normx = \
               lsmr(c_mat.transpose(), b_mat.transpose())
    
    eta_mat =  eta_mat_T.transpose()
    
    for eta_num in range(0,len(eta)):
        eta[str(eta_num)][col] = eta_mat[eta_num]

# check c_matrix
'''
plt.imshow(c_mat)
plt.colorbar()
plt.show()
'''

# check fake spectra, with offsets
for i in range(0,len(eta)):
    
    plt.plot(np.add(eta[str(i)],0.1*i))
    plt.annotate(str(i), (0,0.1*i), xytext=None)

plt.savefig('junk_19specs2.png')
plt.show()

# check fake spectra, without offsets
for i in range(0,len(eta)):
    
    plt.plot(eta[str(i)])

plt.show()





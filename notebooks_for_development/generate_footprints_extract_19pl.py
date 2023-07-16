#!/usr/bin/env python
# coding: utf-8

import glob as glob
import pandas as pd
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
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

# initial frames and parameters:
#   x_extent: length of x dim of arrays
#   y_extent: " " " " y " "
#   bb_array: broadband frame (used if applicable)
#   bb_array_translated: broadband frame translated in (x,y) (used if applicable)
#   array_variance: 2D variance across detector
#   test_array: 2D sample array (which helps define parameters later on)
x_extent, y_extent, bb_array, bb_array_translated, dark, array_variance, test_array = basic_fcns.init_frames(stem)

# make canvas_array onto which all profiles (i.e., the meta-footprint) will be placed
# (useful for testing, or generating placeholder fake data that is the same as the profiles)
canvas_array = np.zeros(np.shape(test_array))

# initialized values of things:
#   rel_pos: relative positions of ports
#   eta_flux: dict to hold flux
#   wavel_soln_ports: dict to hold wavelength solutions for each port
rel_pos, eta_flux, wavel_soln_ports = basic_fcns.infostack(x_extent, y_extent)

# empirical wavelength soln for each spectrum, based on a single series of narrowband images
df_wavel_solns_shift = basic_fcns.wavel_solns_empirical(rel_pos)

# degree of (x,y) translation, and detector readout D
x_shift, y_shift, D = basic_fcns.get_detector_array_and_shift(data_choice, 
                                                              bb_array, 
                                                              dark, 
                                                              canvas_array, 
                                                              rel_pos, 
                                                              sigma=sigma_all)

import ipdb; ipdb.set_trace()
# generate wavelength solutions (for all pixels in spectra, not just from narrowband footprints)
# INPUTS: 
# 1.    relative positions of each spectrum
# 2.    empirical wavelength basis set for each spectrum (after removing (x,y) offset of each spectrum)
# ---> polynomial fit to basis sets ---> 
# OUTPUTS:
# 1.    full but simple x0,y0,lambda array
wavel_solns_full = basic_fcns.get_full_wavel_solns(x_extent, y_extent, rel_pos, df_wavel_solns_shift)
import ipdb; ipdb.set_trace()
# return 2D profiles of spectra
dict_profiles = basic_fcns.gen_spec_profile(rel_pos, 
                                            x_shift, 
                                            y_shift, 
                                            canvas_array, 
                                            test_array, 
                                            df_wavel_solns_shift, 
                                            len_spec, 
                                            sigma_all)


# check profile/spectrum overlap is good
'''
plt.imshow(np.add((1e4)*canvas_array,bb_array), origin='lower')
plt.title('profiles + data to check overlap')
plt.show()
#plt.savefig('junk_overlap.png')
'''

# fake data for testing (this treats the simple profiles as spectra) (obsolete?)
#bb_fake = canvas_array

#plt.imshow(bb_fake)
#plt.show()

# extract flux of all 19 ports
eta_flux = basic_fcns.extract_19pl(x_extent, y_extent, eta_flux, dict_profiles, D, array_variance)

# map wavel solns to fluxes
eta_all = basic_fcns.apply_wavel_solns(x_shift, y_shift, len_spec, rel_pos, canvas_array, wavel_soln_ports, eta_flux, wavel_solns_full)

import ipdb; ipdb.set_trace()

'''
# check extracted flux
for key, coord_xy in rel_pos.items():
    plt.plot(eta_wavel[str(key)],eta_flux[str(key)]+2000*float(key))
plt.show()
'''

# check overlap of spectra and profiles
'''
plt.imshow(D+canvas_array, origin='lower')
plt.title('D+canvas_array')
plt.show()
'''

# check c_matrix
#plt.imshow(c_mat)
#plt.colorbar()
#plt.show()

# overplot wavelength solns
'''
for i in range(0,len(eta_flux)):
    plt.plot(eta_wavel[str(i)])
plt.show()
'''

'''
# cross-talk tests

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
'''

'''
# check cross-talk: compare retrievals, truth

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

'''
# pickle stuff
import pickle
canvas_array2 = np.zeros(np.shape(test_array))
for key, coord_xy in rel_pos.items():
    canvas_array2 += dict_profiles[key]
with open('junk.pkl', 'wb') as handle:
    pickle.dump(canvas_array2, handle)
'''
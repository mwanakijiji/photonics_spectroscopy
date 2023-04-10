#!/usr/bin/env python
# coding: utf-8

# This generates an instrument response matrix based on input data with 
# illumination from different pairs of <lenslet,lambda>

import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import pandas as pd
from astropy.io import fits

from simulator import make_detector_readout, detector_responses
from utils import extraction, map_to_wavel

stem_abs = '/Users/bandari/Documents/git.repos/photonics_spectroscopy/'

# no noise, large footprint:
'''
file_name_empirical = stem_abs + 'src/stellar_empirical_no_noise.pkl'
file_name_white_scan = stem_abs + 'src/white_light_no_noise.pkl'
'''
# no noise, small footprint:
#file_name_empirical = stem_abs + 'src/stellar_empirical_no_noise_small_footprint.pkl'
#file_name_empirical = stem_abs + 'src/stellar_empirical_no_noise.pkl'
#file_name_white_scan = stem_abs + 'src/white_light_no_noise.pkl'

file_name_empirical = stem_abs+'junk_empirical.pkl'
file_name_response = stem_abs+"junk_response.pkl"
#file_name_scan_write = 'junk_scan.pkl'
wavel_model = stem_abs+'junk_wavel_model.pkl'
file_name_A_matrix = 'junk_A_matrix.pkl'

# array sizes
x_size_arrays = 1000
y_size_arrays = 100

# number of command-response pairs
M_basis_set_cmds = 100

#######################
## IF SIMULATION, MAKE THE DATA

'''
array_2d_w_spec = make_detector_readout.white_light_scan(file_name_scan_write=file_name_empirical, N_cmd=100, angle=0, 
                                                         height=0.1, x_size=x_size_arrays, y_size=y_size_arrays)
'''

# make a 2D detector array and save it
array_2d_w_spec = make_detector_readout.fake_readout_1_spec(file_name_empirical_write=file_name_empirical,
                                                            angle=2, height=0.5, 
                                                            x_size=x_size_arrays, y_size=y_size_arrays)

# obtain cube of detector response basis set
# convention: [ slice , x pixel , y pixel ]
poke_matrix, test_cmds, test_response, test_wavel = detector_responses.fake_white_scan(file_name_responses_write=file_name_response,
                                                                               angle=2, height=0.5, N_cmd = M_basis_set_cmds, 
                                                                               x_size=x_size_arrays, y_size=y_size_arrays)

#######
## GENERATE THE RESPONSE MATRIX
response_matrix =  detector_responses.response_matrix(file_name_response_basis_read=file_name_response, 
                                                      file_name_A_matrix_write=file_name_A_matrix)

#######################
## GENERATE WAVELENGTH MODEL
## ## IMPROVEMENT NEEDED: DONT JUST USE COMMAND NUMBER AS INPUT TO THE FIT

# generate wavelength model from calibration frames
# test commands have to be reformatted into a 1D array
test_cmds_by_number = np.arange(len(test_wavel))
test_model = map_to_wavel.gen_model(training_channels=test_cmds_by_number,
                                    training_wavel_steps=test_wavel, 
                                    file_name_write = wavel_model)

#######################
## DO RAW EXTRACTION OF FLUX

#plt.imshow(test_response[50,:,:])
#plt.show()
# do the decomposition with an empirical frame and the response info
# file_name_response_basis: .pkl file containing [response_matrix, test_cmds, test_response]
test = extraction.extract(file_name_response_basis_read = file_name_response, 
                          file_name_A_matrix_read = file_name_A_matrix,
                          file_name_empirical_read = file_name_empirical)

#######################
## MAP WAVELENGTH SOLUTION TO FLUX
# extracted channels are defined with a stand-in for now
test_wavel_mapping = map_to_wavel.apply_model(extracted_channels = test_cmds_by_number, 
                                        file_name_wavel_model_read = wavel_model)

#######################
## TBD: FLUX CALIBRATION


#######################

# reaquire 'truth' for comparison
stem_spec = '/Users/bandari/Documents/git.repos/rrlfe/src/model_spectra/rrmods_all/original_ascii_files'
# this is just read in to determine length and wavelengths
spec_truth = pd.read_csv(stem_spec + '/700020m30.smo', delim_whitespace=True, names=['wavel','flux','noise'])

plt.clf()
fig, axs = plt.subplots(2,1)
axs[0].imshow(array_2d_w_spec)
axs[0].set_title('empirical array')
axs[1].imshow(test_response[50])
axs[1].set_title('Test response (slice 50)')
plt.savefig('test_ingredients.png')

plt.clf()
plt.plot(test_wavel_mapping, np.divide(test,np.max(test)), color='red',label='extracted')
plt.plot(spec_truth['wavel'], np.divide(spec_truth['flux'],np.max(spec_truth['flux'])), color='blue',label='truth')
plt.legend()
plt.title('Test extraction (flux normalized to max)')
plt.savefig('test_extract.png')
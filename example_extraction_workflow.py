#!/usr/bin/env python
# coding: utf-8

# This generates an instrument response matrix based on input data with 
# illumination from different pairs of <lenslet,lambda>

import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle

from simulator import make_detector_readout, detector_responses
from utils import extraction

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

file_name_empirical = stem_abs+'test_empirical.pkl'
file_name_response = stem_abs+"test_response.pkl"

# array sizes
x_size_arrays = 1000
y_size_arrays = 100

# number of command-response pairs
M_basis_set_cmds = 100

# make a 2D detector array and save it
array_2d_w_spec = make_detector_readout.fake_readout(file_name_empirical_write=file_name_empirical,
                                                     x_size=x_size_arrays, y_size=y_size_arrays)

# obtain cube of detector responses
# convention: [ slice , x pixel , y pixel ]
response_matrix, test_cmds, test_response = detector_responses.fake_white_scan(
    N_cmd = M_basis_set_cmds, x_size=x_size_arrays, y_size=y_size_arrays, 
    file_name_responses_write=stem_abs+"test_response.pkl")

# do the decomposition with an empirical frame and the response info
# file_name_response_basis: .pkl file containing [response_matrix, test_cmds, test_response]
test = extraction.extract(file_name_response_basis_read = file_name_response, 
                          file_name_empirical_read = file_name_empirical)


plt.clf()
fig, axs = plt.subplots(2,1)
axs[0].imshow(array_2d_w_spec)
axs[0].set_title('empirical array')
axs[1].imshow(test_response[50])
axs[1].set_title('Test response (slice 50)')
plt.savefig('test_ingredients.png')

plt.clf()
plt.plot(test)
plt.title('Test extraction')
plt.savefig('test_extract.png')
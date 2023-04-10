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

file_name_empirical = stem_abs+'junk_empirical.pkl' # fake 2D data frame
#file_name_scan_write = 'junk_scan.pkl'
wavel_model = stem_abs+'junk_wavel_model.pkl'
file_name_A_matrix = 'junk_A_matrix.pkl' # response matrix

import ipdb; ipdb.set_trace()

#######################
## DO RAW EXTRACTION OF FLUX

#plt.imshow(test_response[50,:,:])
#plt.show()
# do the decomposition with an empirical frame and the response info
# file_name_response_basis: .pkl file containing [response_matrix, test_cmds, test_response]
test = extraction.extract(file_name_A_matrix_read = file_name_A_matrix,
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
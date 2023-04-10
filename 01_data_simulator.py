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

file_name_empirical = stem_abs+'junk_empirical.pkl' # fake 2D data frame
file_name_raw_response = stem_abs+"junk_response.pkl"#_angle1_height0pt8.pkl" # raw responses (poke matrices, but no response matrix)
#file_name_scan_write = 'junk_scan.pkl'
wavel_model = stem_abs+'junk_wavel_model.pkl'

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
poke_matrix, test_cmds, test_response, test_wavel = detector_responses.fake_white_scan(file_name_responses_write=file_name_raw_response,
                                                                               angle=2, height=0.5, N_cmd = M_basis_set_cmds, 
                                                                               x_size=x_size_arrays, y_size=y_size_arrays)
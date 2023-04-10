import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import pandas as pd
from astropy.io import fits

from simulator import make_detector_readout, detector_responses
from utils import extraction, map_to_wavel

file_name_raw_response = stem_abs+"junk_response.pkl"#_angle1_height0pt8.pkl" # raw responses (poke matrices, but no response matrix)
#file_name_scan_write = 'junk_scan.pkl'
wavel_model = stem_abs+'junk_wavel_model.pkl'
file_name_A_matrix = 'junk_A_matrix.pkl' # response matrix


#######
## GENERATE THE RESPONSE MATRIX
response_matrix =  detector_responses.response_matrix(file_name_response_basis_read=file_name_raw_response, 
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
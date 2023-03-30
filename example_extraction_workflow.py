#!/usr/bin/env python
# coding: utf-8

# This generates an instrument response matrix based on input data with 
# illumination from different pairs of <lenslet,lambda>

import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle

from simulator import make_detector_readout, detector_responses

# array sizes
x_size_arrays = 1000
y_size_arrays = 100

# number of command-response pairs
M_basis_set_cmds = 100

# obtain 2D detector array
array_2d_w_spec = make_detector_readout.fake_readout(x_size=x_size_arrays, y_size=y_size_arrays)

# obtain cube of detector responses
# convention: [ slice , x pixel , y pixel ]
response_matrix, test_cmds, test_response = detector_responses.fake_white_scan(N_cmd = M_basis_set_cmds, x_size=x_size_arrays, y_size=y_size_arrays)

# do the decomposition
test = extraction.extract()
#!/usr/bin/env python
# coding: utf-8

# This generates an instrument response matrix based on input data with 
# illumination from different pairs of <lenslet,lambda>

import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle

from simulator.detector_responses import toy_single_spectrum
from simulator.test_commands import toy_commands

# number of command-response pairs
M_basis_set_cmds = 11

# obtain test cube of detector responses
# convention: [ slice , x pixel , y pixel ]
test_response = toy_single_spectrum(num_responses = M_basis_set_cmds)

# obtain cube of commands (note this is not the 'poke' matrix yet)
# convention: [ cmd_number , input signal ]
test_cmds = toy_commands(num_cmds = M_basis_set_cmds)

# total number of detector pixels
N_pixels = np.shape(test_response)[1]*np.shape(test_response)[2]

# initialize poke matrix
# convention: [ pixel signal , cmd_number ]
poke_matrix = np.zeros((N_pixels,M_basis_set_cmds))

# possible other way of rearranging 2D detector response into the 'poke' matrix with one less dimension
# (i.e., x,y just becomes a single dimension of pixel ID)
#poke_matrix = test_response.reshape(20000,12, order='C')
import ipdb; ipdb.set_trace()
# loop over command-response pairs,
# put flattened 2D arrays into the 'poke' matrix
for i in range(0,np.shape(test_response)[0]):
    
    # show detector response going in
    #plt.imshow(detector_response_basis[i], origin="lower")
    #plt.show()
    
    # accumulate flattened responses
    flattened = test_response[i].flatten()
    poke_matrix[:,i] = flattened

# pseudoinverse: the instrument response matrix
response_matrix = np.linalg.pinv(poke_matrix)

### simple test: science signal S equals (approx) pseudoinverse(P) [response matrix] * V [pixel responses]
'''
S = np.matmul(response_matrix,poke_matrix[:,6]) # col of poke matrix here is same as detector_response_basis_flat[0]
# how do the calculated and true values compare?
plt.clf()
# plot the calculated science signal
plt.plot(S)
# plot the real science signal
plt.plot(np.add(test_cmds[6,:],1))
plt.xlim([0,20])
plt.show()
'''

# pickle stuff
data_list = [response_matrix, test_cmds, test_response]
file_name = "sample.pkl"
open_file = open(file_name, "wb")
pickle.dump(data_list, open_file)
open_file.close()
'''
object_1 = 1
object_2 = "A string"
object_3 = 5

sample_list = [object_1, object_2, object_3]
file_name = "sample.pkl"

open_file = open(file_name, "wb")
pickle.dump(sample_list, open_file)
open_file.close()
'''
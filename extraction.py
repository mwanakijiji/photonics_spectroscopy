#!/usr/bin/env python
# coding: utf-8

# Tests extract step itself, given known matrices

import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.sparse.linalg import lsmr
import time
from astropy.io import fits

from scipy.sparse import diags

def extract():

     # no noise, small footprint:
     '''
     file_name_empirical = "stellar_empirical_no_noise_small_footprint.pkl"
     file_name = "white_light_no_noise_small_footprint.pkl"
     '''
     # no noise, large footprint:
     file_name_empirical = "stellar_empirical_no_noise.pkl"
     file_name = "white_light_no_noise.pkl"

     # retrieve instrument response matrix
     open_file = open(file_name, "rb")

     # response matrix, test commands, 2D detector test response
     A_mat, test_cmds, test_response = pickle.load(open_file)
     open_file.close()

     # kludge: transpose
     A_mat = A_mat.T


     # retrieve 'empirical data'
     open_file = open(file_name_empirical, "rb")
     empirical_2d_array = pickle.load(open_file)[0]
     open_file.close()

     # add noise
     array_size = np.shape(np.random.normal(size=(np.shape(empirical_2d_array)[0],np.shape(empirical_2d_array)[1])))
     empirical_2d_array += (np.max(empirical_2d_array)/20.)*np.random.normal(size=array_size)

     '''
     # write to FITS to check

     hdu = fits.PrimaryHDU(empirical_2d_array)
     hdul = fits.HDUList([hdu])
     hdul.writeto('junk_empirical_2d_array.fits')

     hdu = fits.PrimaryHDU(test_response)
     hdul = fits.HDUList([hdu])
     hdul.writeto('junk_test_response.fits')
     '''

     '''
     plt.imshow(test_cmds)
     plt.show()
     '''

     # define variance
     #detector_variance = 0.1*np.random.normal(loc=0,size=(np.shape(empirical_2d_array)[0],np.shape(empirical_2d_array)[1]))
     detector_variance = 0.1*np.ones((np.shape(empirical_2d_array)[0],np.shape(empirical_2d_array)[1]))

     # make weight matrix
     w = 1./detector_variance.flatten()
     W = diags(w, 0)

     # define what we're decomposing
     detector_measured = empirical_2d_array

     # ---------- decompose and time it

     time_00 = time.time()

     time_0 = time.time()

     A = A_mat

     # compute matrices/vectors
     ATW = A.T@W # A^T . W

     time_1 = time.time()
     print("Time:",time_1-time_0)
     # -------------------------------------
     time_0 = time.time()

     #self.report(ATW, 'ATW')
     #ATWA = ATW.dot(A) # A^T . W . A
     ATWA = ATW@A # A^T . W . A

     time_1 = time.time()
     print("Time:",time_1-time_0)
     # -------------------------------------
     time_0 = time.time()

     #self.report(ATWA, 'ATWA')
     ATWx = ATW@detector_measured.flat

     time_1 = time.time()
     print("Time:",time_1-time_0)

     time_0 = time.time()

     thresh=3e-4

     # compute damping coefficient
     ATWAdiag = ATWA.diagonal()
     damp = thresh * ATWAdiag.max()

     time_1 = time.time()
     print("Time:",time_1-time_0)

     spec_lw, istop, itn, normr, normar, norma, conda, normx = \
               lsmr(ATWA, ATWx,
                    damp=damp,
                    )

     time_11 = time.time()

     print("Total time:", time_11-time_00)
     # ---------- 

     return spec_lw

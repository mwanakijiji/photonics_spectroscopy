#!/usr/bin/env python
# coding: utf-8

# Tests extract step itself, given known matrices

import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.sparse.linalg import lsmr
import time

from scipy.sparse import diags

# retrieve matrices
file_name = "./sample.pkl"
open_file = open(file_name, "rb")

# response matrix, text commands, test response
A_mat, test_cmds, test_response = pickle.load(open_file)
open_file.close()

# make fake empirical detector readout and noise (variance) array
detector_measured = np.random.normal(loc=0,size=(np.shape(test_response)[1],np.shape(test_response)[0]))
detector_variance = 0.1*np.random.normal(loc=0,size=(np.shape(test_response)[1],np.shape(test_response)[0]))

# make weight matrix
w = 1./detector_variance.flatten()
w = np.float32(w)
W = diags(w, 0)

# cosmetic
A = A_mat

import ipdb; ipdb.set_trace()

# convert all matrices from elements of np.float64 to 32
A = np.float32(A)
W = np.float32(W)
detector_measured = np.float32(detector_measured)
detector_variance = np.float32(detector_variance)
test_cmds = np.float32(test_cmds)
test_response = np.float32(test_response)

time_0 = time.time()
# compute matrices/vectors
ATW = A.T.dot(W) # A^T . W
time_1 = time.time()
print("Time 1:",time_1-time_0)

time_0 = time.time()
#self.report(ATW, 'ATW')
ATWA = ATW.dot(A) # A^T . W . A
time_1 = time.time()
print("Time 2:",time_1-time_0)

time_0 = time.time()
#self.report(ATWA, 'ATWA')
ATWx = ATW.dot(detector_measured.flat)
time_1 = time.time()
print("Time 3:",time_1-time_0)

time_0 = time.time()
thresh=3e-4
# compute damping coefficient
ATWAdiag = ATWA.diagonal()
damp = thresh * ATWAdiag.max()
time_1 = time.time()
print("Time 4:",time_1-time_0)


'''
def extract(self, im, variance=None, thresh=3e-4):
        "invert linear response to recover cube"
        _log.debug('extract called')
        # compute weight array
        from scipy.sparse import diags
        if variance is None:
            w = n.ones(self.sim.ndetpix, dtype=n.float) # uniform weights
        else:
            w = 1./variance.flatten()
        if self.badpix is not None:
            wb = n.nonzero(self.badpix.flat)[0]
            w[wb] = 0.
        if n.any(n.isnan(im)):
            wb = n.nonzero(n.isnan(im.flat))[0]
            im[n.isnan(im)] = 0. # TEMP
            w[wb] = 0.
        W = diags(w, 0)

        # compute matrices/vectors
        ATW = self.A.T.dot(W) # A^T . W
        self.report(ATW, 'ATW')
        ATWA = ATW.dot(self.A) # A^T . W . A
        self.report(ATWA, 'ATWA')
        ATWx = ATW.dot(im.flat)

        # compute damping coefficient
        ATWAdiag = ATWA.diagonal()
        damp = thresh * ATWAdiag.max()

        # get least-squares solution
        from scipy.sparse.linalg import lsmr
        _log.info('running least-squares solver ...')
        # solve (ATWA)S = ATWx for S, where S is science signal and x is image
        spec_lw, istop, itn, normr, normar, norma, conda, normx = \
          lsmr(ATWA, ATWx,
               damp=damp,
               )
        _log.info('done.')

              
        # reformat structure into datacube
        spectra = n.empty((self.sim.nll, self.nlam), dtype=n.float)
        spectra[:] = n.nan
        spectra[self.wlw] = spec_lw
        return spectra
'''


# In[ ]:





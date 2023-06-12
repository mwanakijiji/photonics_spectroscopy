#!/usr/bin/env python
# coding: utf-8

# Makes a 2D best fit to a white light scan

import glob as glob
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.sparse.linalg import lsmr
from utils2 import fake_19pl_data, basic_fcns

import cv2
import scipy

stem = '/Users/bandari/Documents/git.repos/photonics_spectroscopy/notebooks_for_development/data/19pl/'

# read in a test frame / dark
dark_apapane = stem + 'raw/apapanedark.fits'
test_array = fits.open(dark_apapane)[0].data
dark = test_array

x_extent = np.shape(test_array)[1]
y_extent = np.shape(test_array)[0]

# read in a broadband frame (already dark-subtracted)
raw_bb_frame_19pl = stem + 'dark_subted/19PL_bb_irnd1.0_optnd3.0.fits'
bb_array = fits.open(raw_bb_frame_19pl)[0].data
# dark subtract
#bb_array = np.subtract(bb_array,dark)

# translate the broadband frame to act like it's tonight's data
height, width = bb_array.shape[:2]
x_shift = 4.3 # +: translate to right
y_shift = 40.8 # +: translate up
T = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
tonight_frame = cv2.warpAffine(bb_array, T, (width, height))

# assign
ref_frame = bb_array

# find translation
## ## later: upsample, to get translations within a fraction of a pixel
corr = scipy.signal.correlate2d(ref_frame, tonight_frame, boundary='symm', mode='same')
y, x = np.unravel_index(np.argmax(corr), corr.shape)  # y, x: tonight's image is displaced by this much from the reference

# need to subtract the above (y,x)
import ipdb; ipdb.set_trace()

print(y)
print(x)

'''
plt.imshow(ref_frame, origin='lower')
plt.imshow(tonight_frame, origin='lower')
plt.show()
'''


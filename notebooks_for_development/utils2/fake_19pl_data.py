
import glob as glob
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import gaussian_filter

# fake stellar spectrum for testing

stem_fake_spec = '/Users/bandari/Documents/git.repos/rrlfe/src/model_spectra/rrmods_all/original_ascii_files'
spec_fake = pd.read_csv(stem_fake_spec + '/700020m30.smo', delim_whitespace=True, names=['wavel','flux','noise'])


'''
stem_fake_spec = '/Users/bandari/Documents/git.repos/photonics_spectroscopy/notebooks_for_development/'
spec_fake = pd.read_csv(stem_fake_spec + 'data/sdss_spectrum.dat', delim_whitespace=True, names=['wavel','flux'])
'''

# normalize flux 
spec_fake['flux_norm'] = np.divide(spec_fake['flux'],np.max(spec_fake['flux']))

def fake_injected(shape_pass, rel_pos_pass, sigma):

    # make test data
    test_spec_data = np.zeros(shape_pass)

    # loop over each spectrum starting position and inject a fake spectrum
    for key, coord_xy in rel_pos_pass.items():

        # stellar spectrum with some step offsets on the edges to check for offsets being introduced in the extraction
        test_spec_data[coord_xy[1],coord_xy[0]:coord_xy[0]+200] = 0.6*np.array(spec_fake['flux_norm'][0:200])

        # convolve with Gaussian
        test_spec_data_smoothed = gaussian_filter(test_spec_data, sigma=sigma)

        # just ones
        #test_data[coord_xy[1]-2,coord_xy[0]:coord_xy[0]+100] = np.ones((100))

    return test_spec_data_smoothed
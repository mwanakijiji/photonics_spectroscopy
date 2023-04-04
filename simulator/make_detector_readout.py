#!/usr/bin/env python
# coding: utf-8

# Reads in stellar spectrum, packs it on top of a 2D array to make it seem like it's real

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pickle
from astropy.io import fits
from scipy.ndimage import gaussian_filter

def fake_readout(file_name_empirical_write, x_size=1000, y_size=100):

    stem_spec = '/Users/bandari/Documents/git.repos/rrlfe/src/model_spectra/rrmods_all/original_ascii_files'

    spec_fake = pd.read_csv(stem_spec + '/700020m30.smo', delim_whitespace=True, names=['wavel','flux','noise'])

    # normalize flux somehow
    norm_val = 1.
    spec_fake['flux_norm'] = norm_val*np.divide(spec_fake['flux'],np.max(spec_fake['flux']))

    '''
    # option to plot
    plt.plot(spec_fake['wavel'],spec_fake['flux_norm'])
    plt.xlabel("Wavelength (angstr)")
    plt.ylabel("Flux")
    plt.show()
    #plt.savefig("truth.png")
    '''

    blank_2d = np.zeros((y_size,x_size))
    spec_perfect = blank_2d
    spec_perfect[int(0.5*y_size),:] = np.array(spec_fake['flux_norm'])

    # convolve with Gaussian
    spec_convolved = gaussian_filter(spec_perfect, sigma=5)

    # small rotation
    test_rotate = scipy.ndimage.rotate(spec_convolved, angle=1, reshape=False)

    array_2d_w_spec = test_rotate

    # pickle
    file_name_3 = file_name_empirical_write
    data_list = [array_2d_w_spec]
    open_file = open(file_name_3, "wb")
    pickle.dump(data_list, open_file)
    open_file.close()

    return array_2d_w_spec


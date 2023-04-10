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

def fake_readout_1_spec(file_name_empirical_write, angle=0, height=0.5, x_size=1000, y_size=100):

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
    spec_perfect[int(height*y_size),:] = np.array(spec_fake['flux_norm'])

    # convolve with Gaussian
    spec_convolved = gaussian_filter(spec_perfect, sigma=5)

    # small rotation
    test_rotate = scipy.ndimage.rotate(spec_convolved, angle=angle, reshape=False)

    array_2d_w_spec = test_rotate

    # pickle
    file_name_3 = file_name_empirical_write
    data_list = [array_2d_w_spec]
    open_file = open(file_name_3, "wb")
    pickle.dump(data_list, open_file)
    open_file.close()

    # write to FITS to check
    file_name = 'junk_fake_readout.fits'
    hdu = fits.PrimaryHDU(array_2d_w_spec)
    hdul = fits.HDUList([hdu])
    hdul.writeto(file_name, overwrite=True)
    print("Wrote",file_name)
    #print(wavel_array)

    return array_2d_w_spec


def DEFUNCT_FOR_NOW_white_light_scan(file_name_scan_write, N_cmd, angle=0, height=0.5, x_size=1000, y_size=100):
    '''
    angle: angle to rotate by
    height: 'height' in y on the detector the footprint lies at, before rotation (0 to 1)
    file_name_scan_write: cube of frames, one frame per response
    N_cmd: number of commands used to make that response (slices axis of cube will be this long)
    '''

    stem_spec = '/Users/bandari/Documents/git.repos/rrlfe/src/model_spectra/rrmods_all/original_ascii_files'

    # this is just read in to determine length and wavelengths
    spec_fake = pd.read_csv(stem_spec + '/700020m30.smo', delim_whitespace=True, names=['wavel','flux','noise'])

    # normalize flux somehow
    norm_val = 1.
    #spec_fake['flux_norm'] = norm_val*np.ones(len(spec_fake['flux_norm'])) #np.divide(spec_fake['flux'],np.max(spec_fake['flux']))

    '''
    # option to plot
    plt.plot(spec_fake['wavel'],spec_fake['flux_norm'])
    plt.xlabel("Wavelength (angstr)")
    plt.ylabel("Flux")
    plt.show()
    #plt.savefig("truth.png")
    '''

    blank_2d = np.zeros((N_cmd,y_size,x_size))
    spec_perfect = blank_2d
    #spec_perfect[int(0.5*y_size),:] = np.array(spec_fake['flux_norm'])

    step_wavel = int(len(spec_fake)/N_cmd)

    # get a discrete list of wavelengths to which each snapshot corresponds
    wavel_array = np.zeros((N_cmd))

    for t in range(0,N_cmd):
        # loop over slices/commands and leave a point response in the cube

        slice_this = np.zeros((y_size,x_size))

        slice_this[int(height*y_size),t*step_wavel:(t+1)*step_wavel] = norm_val*np.ones(len(spec_fake['flux']))[t*step_wavel:(t+1)*step_wavel]

        # convolve with Gaussian
        spec_convolved = gaussian_filter(slice_this, sigma=5)

        # small rotation
        test_rotate = scipy.ndimage.rotate(spec_convolved, angle=angle, reshape=False)

        # add to cube
        spec_perfect[t,:,:] = test_rotate

        # add wavelength
        wavel_array[t] = spec_fake['wavel'].loc[t*step_wavel]

    array_scan = np.copy(spec_perfect)

    # pickle
    file_name_3 = file_name_scan_write
    data_list = [array_scan, wavel_array]
    open_file = open(file_name_3, "wb")
    pickle.dump(data_list, open_file)
    open_file.close()
    print("Wrote",file_name_3)

    '''
    # write to FITS to check
    file_name = 'junk_white_light.fits'
    hdu = fits.PrimaryHDU(array_scan)
    hdul = fits.HDUList([hdu])
    hdul.writeto(file_name, overwrite=True)
    print("Wrote",file_name)
    #print(wavel_array)
    '''

    return array_scan

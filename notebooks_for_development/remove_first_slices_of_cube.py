#!/usr/bin/env python
# coding: utf-8

# Takes one of SCExAO's giant FITS cubes and saves the first few slices as 
# another FITS file, so as to do stuff locally with them

# Created 2023 Apr 23 by E.S.

from astropy.io import fits
import numpy as np
import glob

file_list = glob.glob('*.fits')

stem = '/Users/bandari/Documents/git.repos/photonics_spectroscopy/notebooks_for_development/'

n_slices = 20

for num_file in range(0,len(file_list)):

    file_name = file_list[num_file]
    hdul = fits.open(file_name)
    n = hdul[0].data[:n_slices]
    
    hdu2 = fits.PrimaryHDU(n)
    hdul2 = fits.HDUList([hdu2])
    hdul2.writeto(file_name.split('.')[0] + '_trunc.fits', overwrite=True)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pickle
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit


def apply_model(extracted_channels, file_name_wavel_model_read):
    # applies a pre-existing wavelength soln model to empirical data

    # retrieve wavelength model
    open_file = open(file_name_wavel_model_read, "rb")
    popt, pcov = pickle.load(open_file)
    open_file.close()

    a0,a1,a2 = popt

    wavel_soln = a0*np.power(extracted_channels,2.) + a1*extracted_channels + a2

    return wavel_soln


def gen_model(training_channels, training_wavel_steps, file_name_write):
    # Generates a wavelength solution from calibration frames

    # equivalent wavelength steps

    # polynomial fit
    def func(x, coeff_0, coeff_1, coeff_2):
        return coeff_0*x**2 + coeff_1*x + coeff_2
    
    # generate mapping
    popt, pcov = curve_fit(func, xdata=training_channels, ydata=training_wavel_steps, p0=np.array([0.01,30,3.]))

    # pickle
    data_list = [popt, pcov]
    open_file = open(file_name_write, "wb")
    pickle.dump(data_list, open_file)
    open_file.close()
    print('Wrote',file_name_write)

    return popt, pcov
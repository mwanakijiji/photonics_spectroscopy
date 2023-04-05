import scipy
import pickle
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from . import *

def fake_white_scan(file_name_responses_write, angle=0, height=0.5, N_cmd = 100, x_size=1000, y_size=100):

    # just for initializing a DataFrame column
    stem_spec = '/Users/bandari/Documents/git.repos/rrlfe/src/model_spectra/rrmods_all/original_ascii_files'
    spec_fake = pd.read_csv(stem_spec + '/700020m30.smo', delim_whitespace=True, names=['wavel','flux','noise'])

    # normalize flux somehow
    norm_val = 1.
    spec_fake['flux_norm'] = norm_val*np.divide(spec_fake['flux'],np.max(spec_fake['flux']))

    # make cubes of command-response pairs

    step_wavel = int(x_size/N_cmd) # step size of impulse on detector, in pixels
    x_detec_size = x_size
    y_detec_size = y_size

    N_pixels = x_detec_size*y_detec_size
    cube_w_step_impulse = np.zeros((100,y_detec_size,x_detec_size))
    cube_w_commands = np.zeros((N_cmd,N_cmd))

    M_basis_set_cmds = N_cmd
    poke_matrix = np.zeros((N_pixels,M_basis_set_cmds))

    # get a discrete list of wavelengths to which each snapshot corresponds
    wavel_array = np.zeros((N_cmd))


    for t in range(0,N_cmd):
        
        vec_command = np.zeros(N_cmd)
        
        # update command vector
        vec_command[t] = 1
        
        # make the fake responses
        blank_2d = np.zeros((y_detec_size,x_detec_size))
        spec_perfect = blank_2d
    
        #import ipdb; ipdb.set_trace()
        # a white light spectrum in y-middle
        spec_perfect[int(height*y_detec_size),t*step_wavel:(t+1)*step_wavel] = norm_val*np.ones(len(spec_fake['flux_norm']))[t*step_wavel:(t+1)*step_wavel]

        '''
        blank_2d = np.zeros((y_detec_size,x_detec_size)) #np.shape(flux_2d_perfect)
        spec_perfect = blank_2d

        spec_perfect[int(0.5*y_detec_size),:] = norm_val*np.ones(len(spec_fake['flux_norm']))
        
        spec_perfect[:,:t*(int(y_detec_size/N_cmd))] = 0
        spec_perfect[:,(t+1)*int(y_detec_size/N_cmd):] = 0
        '''
        
        spec_convolved = gaussian_filter(spec_perfect, sigma=5)

        test_rotate = scipy.ndimage.rotate(spec_convolved, angle=angle, reshape=False)

        '''
        plt.imshow(test_rotate)
        plt.show()
        '''
        
        '''
        # noisy background
        array_2d_substrate = 0.01*np.random.normal(size=(x_detec_size,y_detec_size))
        '''
        # no noise
        #array_2d_substrate = np.zeros((x_detec_size,y_detec_size))
        
        array_2d_w_spec = test_rotate
        #import ipdb; ipdb.set_trace()
        test_response = np.copy(array_2d_w_spec) # cosmetic
        
        # accumulate flattened responses
        flattened = test_response.flatten()

        # each column is one 'poke', and the elements are the response function
        poke_matrix[:,t] = flattened
        #poke_matrix[i,:] = flattened
        
        '''
        plt.imshow(array_2d_w_spec)
        plt.show()
        '''

        # add commands to cube
        cube_w_commands[t,:] = vec_command
        
        # add frame to cube
        cube_w_step_impulse[t,:,:] = array_2d_w_spec

        # add wavelength
        wavel_array[t] = spec_fake['wavel'].loc[t*step_wavel]

    # pseudoinverse: the instrument response matrix
    response_matrix = np.linalg.pinv(poke_matrix)

    # option to write to FITS
    '''
    hdu = fits.PrimaryHDU(cube_w_step_impulse)
    hdul = fits.HDUList([hdu])
    hdul.writeto('junk_poke_white_light_no_noise_small_footprint.fits', overwrite=True)
    '''

    '''
    # write to FITS to check
    file_name = 'junk_white_light.fits'
    hdu = fits.PrimaryHDU(array_scan)
    hdul = fits.HDUList([hdu])
    hdul.writeto(file_name, overwrite=True)
    print("Wrote",file_name)
    #print(wavel_array)
    '''

    test_response = cube_w_step_impulse
    test_cmds = cube_w_commands
    response_matrix = response_matrix

    # pickle
    data_list = [response_matrix, test_cmds, test_response, wavel_array]
    file_name = file_name_responses_write
    open_file = open(file_name, "wb")
    pickle.dump(data_list, open_file)
    open_file.close()

    return response_matrix, test_cmds, test_response



'''
# this is the old version
def toy_single_spectrum(num_responses):
    # generates cube of frames, each with a bit of a single spectrum

    # 2D detector dims
    x_size = 100
    y_size = 200
    n_slices = num_responses # how many frames

    #N_pixels = x_size*y_size
    detector_2D_basis = np.zeros((n_slices,x_size,y_size)) # initialize the plane of the detector

    for i_slice in range(0,n_slices-1):

        detector_2D_basis[i_slice,40:60,50+20*i_slice:60+20*i_slice] = 1.2

    # repeat the last response, to get a rectangular matrix
    detector_2D_basis[n_slices-1,:,:] = detector_2D_basis[n_slices-2,:,:]

    # convention:
    # [ slice , x pixel , y pixel ]
    return detector_2D_basis
'''
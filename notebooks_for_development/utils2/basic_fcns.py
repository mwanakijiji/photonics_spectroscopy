import pandas as pd
import os
from . import fake_19pl_data
import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.sparse.linalg import lsmr

def apply_wavel_solns(x_shift, y_shift, len_spec, rel_pos, canvas_array, wavel_soln_ports, eta_flux, wavel_solns_full):

    # dict to hold the wavelength solution for each extracted spectrum
    eta_all = {}

    # apply wavelength solutions
    for key, coord_xy in rel_pos.items():
        # note the (x,y) coordinates stretch over entire detector, not just the region sampled for the wavelength soln

        eta_all[str(key)] = pd.DataFrame()

        # full wavel soln is zeroed
        # eta_flux is NOT zeroed
        eta_all[str(key)]['x0'] = wavel_solns_full[str(key)]['x_full_0'] # zeroed
        eta_all[str(key)]['y0'] = wavel_solns_full[str(key)]['y_full_0'] # zeroed
        eta_all[str(key)]['x'] = wavel_solns_full[str(key)]['x_full_0'] + rel_pos[str(key)][0] # in detector coords
        eta_all[str(key)]['y'] = wavel_solns_full[str(key)]['y_full_0'] + rel_pos[str(key)][1]# in detector coords
        eta_all[str(key)]['wavel'] = wavel_solns_full[str(key)]['wavel']
        eta_all[str(key)]['flux'] = eta_flux[str(key)][coord_xy[0]:coord_xy[0]+len_spec]
        
    return eta_all
    

def extract_19pl(x_extent, y_extent, eta_flux, dict_profiles, D, array_variance):

    # loop over detector cols
    for col in range(0,x_extent): 
        
        # initialize matrices; we will solve for
        # c_mat.T * x.T = b_mat.T to get x
        c_mat = np.zeros((len(eta_flux),len(eta_flux)), dtype='float')
        b_mat = np.zeros((len(eta_flux)), dtype='float')

        # loop over pixels in col
        for pix_num in range(0,y_extent):
            
            # nested loop over matrix elements by (row, col)
            # there's doubtlessly a faster way to do this
            for mat_row in range(0,len(dict_profiles)):
                    
                # these are done all at once for speed; there must be a better way
                c_mat[mat_row][0] += dict_profiles[str(int(mat_row))][pix_num,col] * dict_profiles['0'][pix_num,col] / array_variance[pix_num,col]
                c_mat[mat_row][1] += dict_profiles[str(int(mat_row))][pix_num,col] * dict_profiles['1'][pix_num,col] / array_variance[pix_num,col]
                c_mat[mat_row][2] += dict_profiles[str(int(mat_row))][pix_num,col] * dict_profiles['2'][pix_num,col] / array_variance[pix_num,col]
                c_mat[mat_row][3] += dict_profiles[str(int(mat_row))][pix_num,col] * dict_profiles['3'][pix_num,col] / array_variance[pix_num,col]
                c_mat[mat_row][4] += dict_profiles[str(int(mat_row))][pix_num,col] * dict_profiles['4'][pix_num,col] / array_variance[pix_num,col]
                c_mat[mat_row][5] += dict_profiles[str(int(mat_row))][pix_num,col] * dict_profiles['5'][pix_num,col] / array_variance[pix_num,col]
                c_mat[mat_row][6] += dict_profiles[str(int(mat_row))][pix_num,col] * dict_profiles['6'][pix_num,col] / array_variance[pix_num,col]
                c_mat[mat_row][7] += dict_profiles[str(int(mat_row))][pix_num,col] * dict_profiles['7'][pix_num,col] / array_variance[pix_num,col]
                c_mat[mat_row][8] += dict_profiles[str(int(mat_row))][pix_num,col] * dict_profiles['8'][pix_num,col] / array_variance[pix_num,col]
                c_mat[mat_row][9] += dict_profiles[str(int(mat_row))][pix_num,col] * dict_profiles['9'][pix_num,col] / array_variance[pix_num,col]
                c_mat[mat_row][10] += dict_profiles[str(int(mat_row))][pix_num,col] * dict_profiles['10'][pix_num,col] / array_variance[pix_num,col]
                c_mat[mat_row][11] += dict_profiles[str(int(mat_row))][pix_num,col] * dict_profiles['11'][pix_num,col] / array_variance[pix_num,col]
                c_mat[mat_row][12] += dict_profiles[str(int(mat_row))][pix_num,col] * dict_profiles['12'][pix_num,col] / array_variance[pix_num,col]
                c_mat[mat_row][13] += dict_profiles[str(int(mat_row))][pix_num,col] * dict_profiles['13'][pix_num,col] / array_variance[pix_num,col]
                c_mat[mat_row][14] += dict_profiles[str(int(mat_row))][pix_num,col] * dict_profiles['14'][pix_num,col] / array_variance[pix_num,col]
                c_mat[mat_row][15] += dict_profiles[str(int(mat_row))][pix_num,col] * dict_profiles['15'][pix_num,col] / array_variance[pix_num,col]
                c_mat[mat_row][16] += dict_profiles[str(int(mat_row))][pix_num,col] * dict_profiles['16'][pix_num,col] / array_variance[pix_num,col]
                c_mat[mat_row][17] += dict_profiles[str(int(mat_row))][pix_num,col] * dict_profiles['17'][pix_num,col] / array_variance[pix_num,col]
                c_mat[mat_row][18] += dict_profiles[str(int(mat_row))][pix_num,col] * dict_profiles['18'][pix_num,col] / array_variance[pix_num,col]

                # b_mat is just 1D, so use mat_row as index
                b_mat[mat_row] += D[pix_num,col] * dict_profiles[str(int(mat_row))][pix_num,col] / array_variance[pix_num,col]
        
        # solve for the following transform:
        # x * c_mat = b_mat  -->  c_mat.T * x.T = b_mat.T
        eta_flux_mat_T, istop, itn, normr, normar, norma, conda, normx = \
                lsmr(c_mat.transpose(), b_mat.transpose())
        
        eta_flux_mat =  eta_flux_mat_T.transpose()
        
        for eta_flux_num in range(0,len(eta_flux)):
            eta_flux[str(eta_flux_num)][col] = eta_flux_mat[eta_flux_num]

    return eta_flux


def get_full_wavel_solns(x_extent, y_extent, len_spec, rel_pos, df_wavel_empirical_zeroed):

    # dict to contain a DataFrame for each spectrum wavelength solution (across all pixels of the array), each of which contains
    # x0: x values (with offset subtracted)
    # y0: y "  "  "
    # lambda: interpolated wavelength
    wavel_solns_full = {}

    # loop over each spectrum's starting position and 
    # calculate a wavelength solution
    for key, coord_xy in rel_pos.items():

        wavel_solns_full[str(key)] = pd.DataFrame()

        x_pix_locs = df_wavel_empirical_zeroed[str(key)]['x_shift_zeroed']
        y_pix_locs = df_wavel_empirical_zeroed[str(key)]['y_shift_zeroed']

        # fit coefficients based on (x,y) coords of given spectrum and the set of basis wavelengths
        fit_coeffs = find_coeffs(
            x_pix_locs,
            y_pix_locs,
            df_wavel_empirical_zeroed[str(key)]['wavel'].values.astype(float)
            )

        ## ## substitute a true spine of a spectrum here, and on an absolute (x,y) system
        wavel_solns_full[str(key)]['x_full_0'] = np.arange(len_spec)

        m_slope = (3.9-0)/(163+143)
        b_int = 143*m_slope
        x_p = len_spec*(np.arange(len(y_pix_locs))/len(y_pix_locs))
        y_p = m_slope*x_p # no y-int, because offset is subtracted # + b_int
        wavel_solns_full[str(key)]['y_full_0'] = np.interp(x=np.arange(len_spec), 
                                                           xp=x_p,
                                                           fp=y_p)

        wavel_solns_full[str(key)]['wavel'] = wavel_from_func((wavel_solns_full[str(key)]['x_full_0'],
                                                               wavel_solns_full[str(key)]['y_full_0']), 
                                                        fit_coeffs[0], fit_coeffs[1], fit_coeffs[2],
                                                        fit_coeffs[3], fit_coeffs[4])
        # sort stuff
        wavel_solns_full[str(key)] = wavel_solns_full[str(key)].sort_values(by='x_full_0', ignore_index=True)

    return wavel_solns_full


def gen_spec_profile(rel_pos, x_shift, y_shift, canvas_array, test_array, df_wavel_empirical_zeroed, len_spec, sigma):

    sigma_all = sigma
    dict_profiles = {} # dict which will hold all profiles

    # loop over each spectrum's starting position and 
    # 1. generate a full spectrum profile
    # 2. calculate a wavelength solution
    for key, coord_xy in rel_pos.items():

        spec_x_left = np.add(coord_xy[0],-x_shift)
        spec_y_left = np.add(coord_xy[1],-y_shift)

        # place profile on detector, while removing translation of frame relative to a reference frame
        profile_this_array = simple_profile(array_shape=np.shape(test_array), 
                                    x_left=spec_x_left, 
                                    y_left=spec_y_left, 
                                    len_spec=len_spec, 
                                    sigma_pass=sigma_all)
        
        # accumulate these onto an array that will let us look at the total footprint
        canvas_array += profile_this_array

        # save single profiles in an array
        dict_profiles[key] = profile_this_array
    
    return dict_profiles


def get_detector_array_and_shift(data_choice, bb_array, dark, canvas_array, rel_pos, sigma):

    sigma_all = sigma

    # define detector array D and any shift in (x,y) we have to account for for placing the profiles
    if data_choice == 'empirical':
        # real 19PL data
        D = bb_array 
        y_shift, x_shift = 0., 0.

    elif data_choice == 'empirical_translated':
        height, width = bb_array.shape[:2]
        T = np.float32([[1, 0, 4.3], [0, 1, 2.8]]) # third elements are sizes of translation, in pixels
        tonight_frame = cv2.warpAffine(bb_array-dark, T, (width, height)) # dark-subtn apparently necessary to avoid numerical issues
        D = tonight_frame
        # find shift relative to reference frame
        ref_frame = bb_array
        corr = scipy.signal.correlate2d(ref_frame, tonight_frame, boundary='symm', mode='same')
        y_abs, x_abs = np.unravel_index(np.argmax(corr), corr.shape)  # y, x: tonight's image is displaced by this much from the reference
        y_shift, x_shift = 0.5*height-y_abs,0.5*width-x_abs

    elif data_choice == 'fake_injected':
        # fake injected spectra
        test_spec_data = fake_19pl_data.fake_injected(shape_pass = np.shape(canvas_array), rel_pos_pass = rel_pos, sigma = sigma_all)
        D = test_spec_data
        y_shift, x_shift = 0., 0.

    elif data_choice == 'fake_profiles':
        # simple profiles themselves are treated as spectra
        # load a pickled frame
        #with open('fake_simple_profiles_width0pt25.pkl', 'rb') as handle: # don't forget to set sigma_pass in basic_fcns.simple_profile()
        #with open('fake_simple_profiles_width0pt5.pkl', 'rb') as handle:
        #with open('fake_simple_profiles_width1pt0.pkl', 'rb') as handle:
        #with open('fake_simple_profiles_width2pt0.pkl', 'rb') as handle:
        with open('fake_simple_profiles_width3pt0.pkl', 'rb') as handle:
            D = pickle.load(handle)
        y_shift, x_shift = 0., 0.

    return x_shift, y_shift, D


def init_frames(stem):

    # read in initial frames
    # read in a test frame / dark
    dark_apapane = stem + 'raw/apapanedark.fits'
    test_array = fits.open(dark_apapane)[0].data
    dark = test_array

    x_extent = np.shape(test_array)[1]
    y_extent = np.shape(test_array)[0]

    # read in a broadband frame (already dark-subtracted)
    # (this will only get used if applicable)
    bb_frame_19pl = stem + 'dark_subted/19PL_bb_irnd1.0_optnd3.0.fits'
    bb_array = fits.open(bb_frame_19pl)[0].data

    # read in a translated broadband frame (already dark-subtracted)
    # (this will only get used if applicable)
    bb_frame_19pl_translated = stem + 'dark_subted/19PL_bb_irnd1.0_optnd3.0_translated.fits'
    bb_array_translated = fits.open(bb_frame_19pl)[0].data

    # ersatz variance frame
    hdul = fits.open(stem + '../stacked.fits')
    test_data = hdul[0].data[0,:,:] # all 3 spectra
    array_variance = hdul[0].data[1,:,:]

    # replace zeros
    array_variance[array_variance == 0] = np.median(array_variance)

    return x_extent, y_extent, bb_array, bb_array_translated, dark, array_variance, test_array


# handy function to return basic initialization things
def infostack(x_extent, y_extent):

    # 19 PL starting positions (x,y) from one narrowband frame
    # (note these are RELATIVE (x,y) positions; the important thing 
    # is that they define the starting positions of the spectra relative to each other)
    rel_pos = {'0':(39,81),
            '1':(29,115),
            '2':(38,113),
            '3':(46,111),
            '4':(23,108),
            '5':(32,106),
            '6':(40,105),
            '7':(49,103),
            '8':(17,101),
            '9':(26,100),
            '10':(34,98),
            '11':(43,96),
            '12':(51,94),
            '13':(20,93),
            '14':(28,91),
            '15':(37,90),
            '16':(45,88),
            '17':(22,84),
            '18':(31,83)}
    
    # dict to hold the extracted spectra, 19PL
    eta_flux = {'0':np.zeros(x_extent),
        '1':np.zeros(x_extent),
        '2':np.zeros(x_extent),
        '3':np.zeros(x_extent),
        '4':np.zeros(x_extent),
        '5':np.zeros(x_extent),
        '6':np.zeros(x_extent),
        '7':np.zeros(x_extent),
        '8':np.zeros(x_extent),
        '9':np.zeros(x_extent),
        '10':np.zeros(x_extent),
        '11':np.zeros(x_extent),
        '12':np.zeros(x_extent),
        '13':np.zeros(x_extent),
        '14':np.zeros(x_extent),
        '15':np.zeros(x_extent),
        '16':np.zeros(x_extent),
        '17':np.zeros(x_extent),
        '18':np.zeros(x_extent)}

    # will hold wavelength solutions, one for each of the 19 ports
    wavel_soln_ports = {}

    return rel_pos, eta_flux, wavel_soln_ports


# return wavelength solutions: as measured empirically, and the best smooth fit to that data
def wavel_solns_empirical(rel_pos):
    '''
    RETURNS: 
    df_wavel_solns_empirical: DataFrame with keys
        x_shift:        x coord of wavelength solution
        y_shift:        y coord " " "
        x_shift_zeroed: x coord of wavelength solution, with offset removed (lowest value is zero)
        y_shift_zeroed: y coord " " " " (lowest value is zero)
        wavel:          wavelength (empirical sampling)
    '''

    # read in empirical wavelength solution as a function of displacement from an arbitrary point
    df_wavel_soln_single = pd.read_pickle('soln_wavelength_xy_shift_20230612.pkl') # valid in (1005, 1750)

    # sort wavelengths
    df_wavel_soln_single.sort_values(by='wavel', inplace=True)

    ## ## OVERWRITE ABOVE and make a cleaner wavelength soln made to imitate the one I found
    df_wavel_solns_empirical = {} # dict to hold DataFrames
    m_slope = (3.9-0)/(163+143)
    b_int = 143*m_slope

    for key, coord_xy in rel_pos.items():

        df_wavel_solns_empirical[str(key)] = pd.DataFrame()

        df_wavel_solns_empirical[str(key)]['wavel'] = df_wavel_soln_single['wavel']

        # x and y values of empirical measurements
        df_wavel_solns_empirical[str(key)]['x_shift'] = df_wavel_soln_single['x_shift'] + coord_xy[0]
        df_wavel_solns_empirical[str(key)]['y_shift'] = df_wavel_solns_empirical[str(key)]['x_shift']*m_slope + b_int  + coord_xy[1]

        # subtract offsets 
        df_wavel_solns_empirical[str(key)]['x_shift_zeroed'] = df_wavel_solns_empirical[str(key)]['x_shift'] - np.min(df_wavel_solns_empirical[str(key)]['x_shift'])
        df_wavel_solns_empirical[str(key)]['y_shift_zeroed'] = df_wavel_solns_empirical[str(key)]['y_shift'] - np.min(df_wavel_solns_empirical[str(key)]['y_shift'])

        # sort
        df_wavel_solns_empirical[str(key)] = df_wavel_solns_empirical[str(key)].sort_values(by='wavel', ignore_index=True)

    return df_wavel_solns_empirical


# return wavelength based on a pre-existing polynomial wavelength solution
def wavel_from_func(X, a_coeff, b_coeff, c_coeff, d_coeff, f_coeff):
    '''
    X: (x,y) array
    a,b,c,d,f: coefficients
    '''

    x_pass, y_pass = X
    
    return a_coeff*x_pass + b_coeff*y_pass + c_coeff*np.multiply(x_pass,y_pass) + d_coeff*np.power(x_pass,2.) + f_coeff*np.power(y_pass,2.)


# initial guesses for a,b,c:
def find_coeffs(x,y,z):
    '''
    x: x [pix]
    y: y [pix]
    z: wavel
    '''

    p0 = 1., 1., 1., 1., 1. # initial guess
    fit_coeffs, cov = curve_fit(wavel_from_func, (x,y), z, p0)

    return fit_coeffs


# gaussian profile (kind of confusing: coordinates are (lambda, x), instead of (x,y) )
def gaus1d(x_left, len_spec, x_pass, lambda_pass, mu_pass, sigma_pass=1):
    '''
    x_left: x coord of leftmost pixel of spectrum (y coord is assumed to be mu_pass)
    len_spec: length of spectrum [pix]
    x_pass: grid of y-coords in coord system of input
    lambda_pass: grid of x-coords in coord system of input
    mu_pass: profile center (in x_pass coords)
    sigma_pass: profile width (in x_pass coords)
    '''
    
    # condition for lambda axis to be inside footprint
    lambda_cond = np.logical_and(lambda_pass >= x_left, lambda_pass < x_left+len_spec)
    
    #plt.imshow(lambda_cond)
    #plt.show()
    
    # profile spanning entire array
    profile = (1./(sigma_pass*np.sqrt(2.*np.pi))) * np.exp(-0.5 * np.power((x_pass-mu_pass)/sigma_pass,2.) )
    
    # mask regions where there is zero signal
    profile *= lambda_cond
    
    # normalize columns of nonzero signal
    profile = np.divide(profile, np.nanmax(profile))
    
    # restore regions of zero signal as zeros (instead of False)
    profile[~lambda_cond] = 0.
    
    return profile


# wrapper to make the enclosing profile of a spectrum
def simple_profile(array_shape, x_left, y_left, len_spec, sigma_pass=1):
    # make simple 1D Gaussian profile in x-direction
    '''
    shape_array: shape of array
    x_left: x-coord of leftmost point of spectrum
    y_left: y-coord of leftmost point of spectrum
    len_spec: length of spectrum (in x-dir)
    sigma_pass: sigma width of profile
    '''

    x_left = int(x_left)
    y_left = int(y_left)

    array_profile = np.zeros(array_shape)

    xgrid, ygrid = np.meshgrid(np.arange(0,np.shape(array_profile)[1]),np.arange(0,np.shape(array_profile)[0]))
    array_profile = gaus1d(x_left=x_left, len_spec=len_spec, x_pass=ygrid, lambda_pass=xgrid, mu_pass=y_left, sigma_pass=sigma_pass)

    #plt.imshow(array_profile)
    #plt.show()
    
    # normalize it such that the marginalization in x (in (x,lambda) space) is 1
    # (with a perfect Gaussian profile in x this is redundant)
    array_profile[:,x_left:x_left+len_spec] = np.divide(array_profile[:,x_left:x_left+len_spec],np.sum(array_profile[:,x_left:x_left+len_spec], axis=0))
    
    return array_profile
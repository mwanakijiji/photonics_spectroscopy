# code for simulation of lenslet-based IFS spectra
#
# Michael Fitzgerald (mpfitz@ucla.edu)
# [ email from MF: There is the “extract” method which has the implementation. ]

import os, pickle
import numpy as n
import matplotlib as mpl
import pylab

import logging
_log = logging.getLogger('simulate')

#import ipdb

class LensletIFSSimulator(object):
    def __init__(self):
        _log.debug('initializing simulator')

        # specify geometry
        #

        # spacing bewteen lenslets on detector
        self.ld = 12. # [pix] displacement
        self.ldth = 30.*n.pi/180. # [rad] angle relative to detector x axis (+ccw)

        self.dlam = 1e-3 # [um/pix]  dispersion in detector frame

        # dispersion direction relative to detector x axis
        self.th_disp = -2.*n.pi/180. # [rad] (+ccw)

        # detector size
        self.ny, self.nx = 96, 128 # [pix], [pix]
        self.ndetpix = self.nx*self.ny

        # number of lenslets across yp and xp axes
        self.nllyp = int(n.round((self.ny*n.cos(self.ldth)+
                                  self.nx*n.sin(self.ldth))/self.ld)) + 2
        self.nllxp = int(n.round((self.nx*n.cos(self.ldth)+
                                  self.ny*n.sin(self.ldth))/self.ld)) + 2
        self.nll = self.nllyp*self.nllxp

        # PSF parameters
        self.pwd = 1./2.3548 # [pix]  PSF width, x axis (dispersion)
        self.pwp = 2./2.3548 # [pix]  PSF width, y axis (perp. to dispersion)
        self.psf_box_size = 10 # [pix]  width of subregion for evaluating PSF onto detector grid

        # shift on detector parameters
        self.shift_x = 0. # [pix]
        self.shift_y = 0. # [pix]

        import hashlib
        self.hasher = hashlib.md5()

    def lenslet_2d_to_1d(self, iy, ix):
        "get 1d lenslet index i given 2d indices iy, ix"
        return iy*self.nllxp + ix

    def lenslet_1d_to_2d(self, i):
        "get 2d lenslet indices ix, iy given 1d index i"
        iy = i / self.nllxp
        ix = i % self.nllxp
        return iy, ix

    def lenslet_to_det(self, i):
        "lenslet index i to detector y,x"

        iy, ix = self.lenslet_1d_to_2d(i)

        # transform from lenslet coords to detector
        y = self.ld * (iy*n.cos(self.ldth)+ix*n.sin(self.ldth)) - \
            (self.nx+1)*n.sin(self.ldth)*n.cos(self.ldth) + \
            self.shift_y
        x = self.ld * (-iy*n.sin(self.ldth)+ix*n.cos(self.ldth)) + \
            (self.nx+1)*n.sin(self.ldth)**2 + \
            self.shift_x

        return y, x


    def lenslet_lam_to_det(self, i, lam):
        "lenslet index i and wavelength lam to detector y,x"
        y0, x0 = self.lenslet_to_det(i)
        y = y0 + lam/self.dlam * n.sin(self.th_disp)
        x = x0 + lam/self.dlam * n.cos(self.th_disp)
        return y, x


    def psf(self, dx, dy, lenslet_ind=None, lam=None):
        "2D elliptical Gaussian, no rotation"
        assert self.psf_box_size > 3*self.pwp
        assert self.psf_box_size > 3*self.pwd
        return n.exp(-dy**2/2./self.pwp**2-dx**2/2./self.pwd**2)/2./n.pi/n.sqrt(self.pwd*self.pwp)


    def simulate_IFS_image(self, lams, spectra,
                           return_sparse=False,
                           multiprocess=False,
                           ):
        "given a spectrum for each lenslet, compute an IFS detector image"

        _log.debug('simulating image ...')

        if return_sparse:
            inds, imdata = [], []
        else:
            im = n.zeros((self.ny,self.nx), dtype=n.float) # blank image

        pad = 5. # [pix] padding around detector for where light centrally falls that might hit detector
        assert self.psf_box_size > pad

        def get_patch(i, y, x, s, lam):
            "relevant computation for a lenslet channel patch"
            # get coordinates for PSF patch
            by = n.max((int(y)-self.psf_box_size//2,0))
            ey = n.min((int(y)+self.psf_box_size//2,self.ny-1))
            bx = n.max((int(x)-self.psf_box_size//2,0))
            ex = n.min((int(x)+self.psf_box_size//2,self.nx-1))
            wy, wx = n.mgrid[by:ey,bx:ex]
            wy, wx = wy.flatten(), wx.flatten()
            wg = (wy, wx)
            # get PSF
            psf = s*self.psf(wx-x, wy-y, lenslet_ind=i, lam=lam)
            if return_sparse:
                # convert 2d inds to 1d index to flattened array
                w1 = wg[0]*self.nx+wg[1]
                ## inds.append(w1)
                ## imdata.append(psf)
                return w1, psf
            else:
                ## im[wg] += psf
                return wg, psf


        wlight = n.nonzero(n.any(spectra!=0., axis=1))[0] # skip if no light
        if multiprocess:

            def worker(input, output):
                for i in iter(input.get, 'STOP'):
                    output_dat = []

                    # positions for each wavelength channel
                    yy, xx = self.lenslet_lam_to_det(i, lams)

                    # place intensity in image given PSF and detector sampling
                    w = n.nonzero((yy >= -pad) & (yy < self.ny+pad) &
                                  (xx >= -pad) & (xx < self.nx+pad))[0]
                    for args in zip(yy[w], xx[w], spectra[i,w], lams[w]):
                        output_dat.append(get_patch(i, *args))

                    output.put(output_dat)

            def feeder(input):
                "place location/intensity data on input queue"
                # loop over lenslets
                for i in wlight:
                    input.put(i, True)
                _log.debug('feeder finished')

            import multiprocessing as mp
            n_process = mp.cpu_count()
            n_max = n_process*8
            _log.debug("{0} processes".format(n_process))
            inqueue = mp.Queue(n_max)
            outqueue = mp.Queue()

            # start worker processes
            for i in range(n_process):
                mp.Process(target=worker, args=(inqueue, outqueue)).start()

            # start feeder
            feedp = mp.Process(target=feeder, args=(inqueue,))
            feedp.start()

            # collect output
            for k in range(len(wlight)):
                for vals in outqueue.get():
                    if return_sparse:
                        w1, psf = vals
                        inds.append(w1)
                        imdata.append(psf)
                    else:
                        wg, psf = vals
                        im[wg] += psf

            # kill worker processes
            _log.debug('received all data; killing workers ...')
            for i in range(n_process):
                inqueue.put('STOP')

            # block until feeder finished
            _log.debug('waiting for feeder to finish ...')
            feedp.join(1.)
            _log.debug('... done.')

        else:
            # loop over lenslets
            wlight = n.nonzero(n.any(spectra!=0., axis=1))[0] # skip if no light
            for i in wlight:
                # positions for each wavelength channel
                yy, xx = self.lenslet_lam_to_det(i, lams)

                # place intensity in image given PSF and detector sampling
                w = n.nonzero((yy >= -pad) & (yy < self.ny+pad) &
                              (xx >= -pad) & (xx < self.nx+pad))[0]
                for y, x, s, lam in zip(yy[w], xx[w], spectra[i,w], lams[w]):
                    if return_sparse:
                        w1, psf = get_patch(i, y, x, s, lam)
                        inds.append(w1)
                        imdata.append(psf)
                    else:
                        wg, psf = get_patch(i, y, x, s, lam)
                        im[wg] += psf

        _log.debug('... done.')

        if return_sparse:
            return inds, imdata
        else:
            return im


    def simulate_WLS(self):
        "simulate a white-light scan"

        _log.info('simulating WLS ...')

        # construct spectrum for each lenslet
        nlam = 300 # num samples
        rlam = 10. # [pix] range
        lams = n.linspace(-rlam, rlam, nlam)*self.dlam # [um]
        spectrum = n.ones(nlam, dtype=n.float)

        # loop over lenslet x position
        ims = []
        for ix in range(self.nllxp):
            print('{0}/{1}'.format(ix,self.nllxp))

            # get 1d lenslet indices for this position
            iyy = n.arange(self.nllyp)
            ii = self.lenslet_2d_to_1d(iyy, ix)

            # get white spectra only at these lenslets
            spectra = n.zeros((self.nll, nlam), dtype=n.float)
            spectra[ii,:] = spectrum

            # simulate the image
            im = self.simulate_IFS_image(lams, spectra)
            ims.append(im)
        ims = n.array(ims)

        _log.info('... done.')

        return ims


def test_simulate_IFS_image(simulator, **kwargs):

    # construct spectrum for each lenslet
    nlam = 300 # num samples
    rlam = 10. # [pix] range
    per = 5. # [pix] period
    lams = n.linspace(-rlam, rlam, nlam)*simulator.dlam # [um]
    spectra = n.zeros((simulator.nll, nlam), dtype=n.float)
    spectra[:] = n.cos(2.*n.pi/per*lams/simulator.dlam)[n.newaxis,:]**2 + 1.

    # simulate
    im = simulator.simulate_IFS_image(lams, spectra, **kwargs)

    # show results
    fig = pylab.figure(0)
    fig.clear()
    ax = fig.add_subplot(111)
    ax.imshow(im,
              interpolation='nearest',
              )
    pylab.draw()
    pylab.show()

    return im

_ExtractorBase = object
class Extractor(_ExtractorBase):
    "class for extracting data from simulator"
    def __init__(self, sim, lams, use_wls=False, badpix=None,
                 recalculate_A=False):
        _log.debug('initializing extractor')
        self.sim = sim
        self.lams = lams # FIXME  need this?
        self.nlam = len(lams)
        thresh = 1e-3

        # ordered pairs of lenslet/wavelength that fall on detector
        lw_pairs = []
        if use_wls:
            pad = 0. # [pix]
        else:
            pad = 2. # [pix]
        for i in range(sim.nll):
            y, x = sim.lenslet_lam_to_det(i, lams)
            w = n.nonzero((y >= -pad) & (y < sim.ny+pad) &
                          (x >= -pad) & (x < sim.nx+pad))[0]
            for j in w:
                lw_pairs.append((i,j))
        nlw = len(lw_pairs)
        lw_pairs = n.array(lw_pairs)

        
        #f = open("lw_pairs.dat",'w')
        #for pair in lw_pairs:
        #    f.write(str(pair)+"\n")
        #f.close()
        #_log.debug("LW_PAIRS")

        
        # construct indexing array
        self.wlw = (lw_pairs[:,0], lw_pairs[:,1])

        _log.debug("{},{}".format(nlw,self.sim.nll*self.nlam))
      
        #f = open("wlw.dat",'w')
        #for pair in self.wlw:
        #    f.write(str(pair)+"\n")
        #f.close()
        #_log.debug("WLW")

        A_key = sim.hasher.hexdigest()
        A_fn = "data/cache/A-{}.dat".format(A_key)
        if recalculate_A or not os.access(A_fn, os.R_OK):

            # detector response for each ordered pair
            _log.info('constructing detector response ...')
            from scipy.sparse import lil_matrix
            if use_wls:
                # estimate linear response (from white light scans)
                A = lil_matrix((nlw, sim.ndetpix))
                yy, xx = n.mgrid[0:sim.ny,0:sim.nx]
                cur_l = -1 # current lenslet
                wlim = None # current white-light scan
                for ll, (i, j) in enumerate(lw_pairs):
                    if i != cur_l:
                        cur_l = i
                        # get white-light scan for this lenslet
                        spectra = n.zeros((sim.nll, self.nlam), dtype=n.float)
                        spectra[i,:] = 1.
                        try:
                            wlim = sim.simulate_IFS_image(lams, spectra)
                        except TypeError:
                            wlim = sim.simulate_IFS_image(spectra)
                    # extract column for this lenslet/wavelength
                    y, x = sim.lenslet_lam_to_det(i, lams[j])
                    ix = int(n.round(x))
                    if (ix < 0) or (ix >= sim.nx): continue
                    w = n.nonzero((wlim.flat > wlim[:,ix].max()*thresh) &
                                  (xx.flat==ix))[0]
                    A[ll,w] = wlim.flat[w]
            else:
                # do the proper linear response (from unobservable impulse
                # responses)
                import resource

                def worker(input, output):
                    for ll, (i,j) in iter(input.get, 'STOP'):
                        _log.debug("worker "+ str(ll)+' out of  '+str(len(lw_pairs)))
                        _log.debug("worker {} maxrss = {}".format(ll, resource.getrusage(resource.RUSAGE_SELF)[2]))
                        spectra = n.zeros((sim.nll, self.nlam), dtype=n.float)
                        spectra[i,j] = 1.
                        try:
                            inds, imdata = sim.simulate_IFS_image(lams, spectra, return_sparse=True)
                        except TypeError:
                            inds, imdata = sim.simulate_IFS_image(spectra, return_sparse=True)
                        # repackage
                        ww = n.unique(n.concatenate(inds)).tolist()
                        imd = n.zeros(len(ww), dtype=n.float)
                        for dind, (w, dat) in enumerate(zip(inds, imdata)):
                            if ~n.any(dat): continue
                            for ind, d in zip(w,dat):
                                i = ww.index(ind)
                                imd[i] += d
                        wg = n.nonzero(imd>imd.max()*thresh)[0]
                        # place on output queue
                        output.put((ll, n.array(ww)[wg], imd[wg]))

                def feeder(input):
                    "place locations on input queue"
                    for ll, (i, j) in enumerate(lw_pairs):
                        _log.debug("feeder " +str(ll)+' out of '+ str(len(lw_pairs)))
                        _log.debug("feeder {} maxrss = {}".format(ll, resource.getrusage(resource.RUSAGE_SELF)[2]))
                        inqueue.put((ll,(i,j)), True)
                    _log.debug('feeder finished')

                import multiprocessing as mp
                n_process = mp.cpu_count()
                n_max = n_process*8
                _log.debug("{0} processes".format(n_process))
                inqueue = mp.Queue(n_max)
                outqueue = mp.Queue()

                # start worker processes
                for i in range(n_process):
                    mp.Process(target=worker, args=(inqueue, outqueue)).start()

                # start feeder
                feedp = mp.Process(target=feeder, args=(inqueue,))
                feedp.start()

                # collect output
                A = lil_matrix((nlw, sim.ndetpix))
                for j in range(len(lw_pairs)):
                    _log.debug("collector {} maxrss = {}".format(j, resource.getrusage(resource.RUSAGE_SELF)[2]))
                    ll, w, imdat = outqueue.get()
                    A[ll,w] = imdat

                # kill worker processes
                _log.debug('received all data; killing workers ...')
                for i in range(n_process):
                    inqueue.put('STOP')

                # block until feeder finished
                _log.debug('waiting for feeder to finish ...')
                feedp.join(1.)
                _log.debug('... done.')

            A = A.T.tocsr() # convert from linked list to compressed sparse row

            with open(A_fn, 'wb') as f:
                pickle.dump(A, f)
        else:
            with open(A_fn, 'rb') as f:
                A = pickle.load(f)

        # now A is ndetpix by nlw
        self.A = A
        self.report(A, 'A')

        self.badpix = badpix

    @staticmethod
    def report(x, name):
        "report some statistics on sparse matrices"
        from scipy.sparse import issparse
        assert issparse(x)
        if len(x.shape)==2:
            nn = n.product(x.shape)
            _log.info("{0} matrix is {1} by {2} ({3} entries), with {4} stored values ({5:.2g}%)".format(name, x.shape[0], x.shape[1], nn, x.nnz, x.nnz/nn*100))
        elif len(x.shape)==1:
            _log.info("{0} vector is length {1}, with {2} stored values ({3:.2g}%)".format(name, x.shape[0], x.nnz, x.nnz/x.shape[0]*100))

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


def test_linear_rep(use_wls=False):
    cal_sim = LensletIFSSimulator()
    data_sim = LensletIFSSimulator()
    data_sim.shift_y = cal_sim.shift_y + 0. # [pix]
    data_sim.shift_x = cal_sim.shift_x + 0. # [pix]
    data_sim.pwd = cal_sim.pwd*1.0 # [pix]  PSF width, x axis (dispersion)
    data_sim.pwp = cal_sim.pwp*1.0 # [pix]  PSF width, y axis (perp. to dispersion)

    # construct wavelength grid
    npix = 38
    lams = (n.arange(npix, dtype=n.float)-(npix/2.))*cal_sim.dlam # [um]
    nlam = len(lams)

    extractor = Extractor(cal_sim, lams, use_wls=use_wls)

    # simulate a data cube
    input_spectra = n.zeros((data_sim.nll, nlam), dtype=n.float)
    per = 18.5 # [pix]  period
    input_spectra[:] = n.cos(2.*n.pi/per*lams/data_sim.dlam)[n.newaxis,:]**2 + 1.
    inspec_lw = input_spectra[extractor.wlw]

    # simulate noise-free detector image
    if use_wls:
        # don't use imperfect linear rep.
        det = data_sim.simulate_IFS_image(lams, input_spectra)
    else:
        # FIXME  maybe we want to use the data simulator here anyway
        det = extractor.A.dot(inspec_lw)
        det.shape = (data_sim.ny, data_sim.nx)

    def show_det(detector_im, fignum=0):
        "show result of detector sim"
        fig = pylab.figure(fignum)
        fig.clear()
        ax = fig.add_subplot(111)
        ax.imshow(detector_im,
                  interpolation='nearest',
                  )
        pylab.draw()
        pylab.show()

    # show noise-free detector image
    show_det(det)

    # get recovered spectra in noise-free image
    output_spectra = extractor.extract(det)

    def show_comparison(inspec, outspec, fignum=1):
        fig = pylab.figure(fignum)
        fig.clear()
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        kw = {'interpolation':'nearest',
              'vmin':0.,
              'vmax':input_spectra.max(),
              'aspect':'auto',
              }

        ax1.imshow(inspec, **kw)
        ax2.imshow(outspec, **kw)

        t = 0.01 # factor for scaling down from peak value to look at residuals
        ax3.imshow(inspec-outspec,
                   interpolation='nearest',
                   aspect='auto',
                   vmin=-t*kw['vmax'],
                   vmax=t*kw['vmax'],
                   cmap=mpl.cm.jet,
                   )

        ax1.set_ylabel('lenslet')
        ax1.set_xlabel('wavelength')

        pylab.draw()
        pylab.show()

    # compare noise-free input and recovered spectra
    show_comparison(input_spectra, output_spectra, fignum=1)

    # add some read noise across detector
    rn_sig = 0.03
    rs = n.random.RandomState(seed=28548)
    det_n = det + rs.randn(data_sim.ny, data_sim.nx)*rn_sig
    show_det(det_n, fignum=2)
    # recover extracted spectra from detector image
    output_spectra_n = extractor.extract(det_n)
    # compare noise-free input and (noisy) recovered spectra
    show_comparison(input_spectra, output_spectra_n, fignum=3)

    #ipdb.set_trace()
    return output_spectra_n

if __name__=='__main__':
    #logging.basicConfig(level=logging.INFO,
    logging.basicConfig(level=logging.DEBUG,
                        format='%(name)-12s: %(levelname)-8s %(message)s',
                        )

    sim = LensletIFSSimulator()
    im = test_simulate_IFS_image(sim)
    im_mp = test_simulate_IFS_image(sim, multiprocess=True)
    ## wls = sim.simulate_WLS()

    #test_linear_rep()
    #test_linear_rep(use_wls=True)
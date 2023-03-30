

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
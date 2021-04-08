import os
from tqdm import tqdm
from joblib import dump as jdump, load as jload

import json
import warnings

import numpy as np
import matplotlib.pyplot as plt
import sklearn.covariance as sklcov
import sklearn.decomposition as skldecomp

from fpca import fpca_pipeline

from scipy import interpolate

from astropy.io import fits

import glob


class MaNGASpecDiffs(object):
    def __init__(self, specs1, specs2, ivars1, ivars2):

        self.specs1, self.specs2 = specs1, specs2
        self.ivars1, self.ivars2 = ivars1, ivars2

        self.specs = np.stack([self.specs1, self.specs2])
        self.ivars = np.stack([self.ivars1, self.ivars2])

    def scaled_pairs(self):
        med = np.median(self.specs, axis=(0, 2), keepdims=True)
        qtls = np.percentile(self.specs, [2., 98.], axis=(0, 2), keepdims=True)
        iqr = qtls[1] - qtls[0]
        return (self.specs - med) / iqr

    def scaled_ivars(self):
        qtls = np.percentile(self.specs, [2., 98.], axis=(0, 2), keepdims=True)
        iqr = qtls[1] - qtls[0]
        return self.ivars * iqr**2.

    @classmethod
    def from_fits(cls, fname):
        with fits.open(fname) as hdulist:
            specs1, specs2 = hdulist['specs1'].data, hdulist['specs2'].data
            ivars1, ivars2 = hdulist['ivars1'].data, hdulist['ivars2'].data

        return cls(specs1, specs2, ivars1, ivars2)


def smooth_evecs(E, wt, x):
    q, n = E.shape

    def _smooth_evec(v, wt, x):
        spl = interpolate.splrep(x=x, y=v, w=wt)

        return interpolate.splev(x, spl)

    E_sm = np.empty_like(E)

    for i in range(q):
        E_sm[i, :] = _smooth_evec(E[i, :], wt, x)

    return E_sm


def differenced_spectra(fname):

    msd = MaNGASpecDiffs.from_fits(fname)

    specs = msd.scaled_pairs()
    ivars = msd.scaled_ivars()
    diffs = np.diff(specs, axis=0).squeeze()
    diffs_ivars = 1. / np.sum(1. / ivars, axis=0)
    diffs_ivars[~np.isfinite(diffs_ivars)] = 0.

    return diffs, diffs_ivars


def make_wavegrid(wavespec):
    wave0, nwave = wavespec
    dloglwave = 1.0e-4
    wave_grid = wave0 * \
        (10.**(np.linspace(0., nwave - 1., nwave) * dloglwave))

    return wave_grid


def batchwise_spline_predict(wave_grid, diffs_batch):
    gridsearch, X, Y = fpca_pipeline(
        wave_grid, diffs_batch, nx='all', scoring='explained_variance', n_jobs=-1)
    Y_splfitpred = gridsearch.predict(X)

    return gridsearch, X, Y, Y_splfitpred


if __name__ == '__main__':
    from conf import DATALOC, PAIRS_DATALOC

    with open(os.path.join(PAIRS_DATALOC, 'wavespecs.json'), 'r') as wavespecs_f:
        wavespecs = json.load(wavespecs_f)

    force_recompute = False

    for wavespec_ix in wavespecs.keys():

        pca_model_fname = 'ipca_{}.joblib'.format(wavespec_ix)

        # try to recover old model instance, or start a new one
        # there are three recovery states:
        #   - not recovering (or already achieved recovery) [0]
        #   - searching for correct file [1]
        #   - searching correct index within that file [2]
        if force_recompute:
            recover = 0
            pca = skldecomp.IncrementalPCA(n_components=50)
            pca.state = {'filename': None, 'subblock': None}
        if os.path.isfile(pca_model_fname):  # recover
            recover = 1
            pca = load(pca_model_fname)
        else:
            recover = 0
            pca = skldecomp.IncrementalPCA(n_components=50)
            pca.state = {'filename': None, 'subblock': None}

        print('STARTING {} in recovery mode {}'.format(wavespec_ix, recover))

        # generate wavelength grid
        wave_grid = make_wavegrid(wavespecs[wavespec_ix])

        specs_fnames = glob.glob(
            os.path.join(PAIRS_DATALOC, 'wavespec_{}/specs_*.fits'.format(wavespec_ix)))

        fn_progress = tqdm(specs_fnames, position=0)
        for fn in fn_progress:
            # if in a recovery state, try to track down the correct recovery point
            # which is the last filename and index *not completed*
            # if in file-finding mode, check current file against in-progress state
            if recover == 1:
                if pca.state['filename'] == fn:
                    recover == 2
                    print('resuming at filename {}'.format(fn))
                    print('recovery mode {}'.format(recover))
                else:
                    continue

            # update internal state
            pca.state.update(filename=fn, subblock=0)

            galname = os.path.basename(fn).split('.')[0].split('_')[1]
            fn_progress.set_postfix(galaxy=galname)
            # iterate  through galaxies

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                diffs = differenced_spectra(fn)[0]

            batchsize = 400
            nbatches = (diffs.shape[0] // batchsize) + 1

            if batchsize >= diffs.shape[0]:
                diffs_batches = [diffs]
            else:
                diffs_batches = np.array_split(diffs, nbatches)

            batch_progress = tqdm(diffs_batches, position=1, leave=False)
            for j, db in enumerate(batch_progress):
                # check recovery state
                # state 1 should not happen
                if recover == 1:
                    raise ValueError(
                        'access to blocks not permitted in recovery state 1 (file-finding)')
                # state 0 is fine (normal operation, non-recovery mode)
                elif recover == 0:
                    pass
                # state 2 is finding sub-block
                elif recover == 2:
                    # if current subblock matches model internal state,
                    # we've found our place and we emerge from recovery mode
                    if pca.state['subblock'] == j:
                        recover = 0
                        print('resuming at subblock {}'.format(j))
                        print('recovery mode {}'.format(recover))
                    # if current subblock does not match,
                    # skip to the next branch of the loop
                    else:
                        continue
                else:
                    raise ValueError('invalid recovery state: {}'.format(recover))
                
                # update internal state, then write out model
                # this ensures that model read-ins will proceed from next unused subblock
                pca.state.update(subblock=j)
                jdump(pca, pca_model_fname)
                
                # find best-fit spline smoothing for each galaxy,
                # and return predictions
                gridsearch, X, Y, Y_splfitpred = batchwise_spline_predict(
                    wave_grid, db)

                # update Incremental PCA with partial fit
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    pca.partial_fit(Y_splfitpred.T)



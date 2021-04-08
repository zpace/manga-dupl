import numpy as np

import os
import glob
import itertools

from marvin import config as mconf
from marvin.utils.general import downloadList as marvin_download

from astropy.utils.console import ProgressBar
from astropy import table as t
from astropy.io import fits


def load_mangadrp(plateifu, loglin='LOG', rsscube='CUBE', mpl_v=None):
    """retrieve local MaNGA logcube, downloading if necessary
    
    Parameters
    ----------
    plateifu : str
        plateifu designation of galaxy
    loglin : str, optional
        wavelength scale (LIN/LOG)
    rsscube : str, optional
        RSS or CUBE
    mpl_v : None or str, optional
        MPL version
    
    Returns
    -------
    fits.HDUList
        HDUList with logcube data
    """
    # resolve plateifu into filename
    if mpl_v is None:
        mpl_v = mconf.release

    drpver, dapver = mconf.lookUpVersions(mpl_v)
    sas_basedir = os.environ['SAS_BASE_DIR']

    plate, ifu = plateifu.split('-')
    fname = f'manga-{plateifu}-{loglin.upper()}{rsscube.upper()}.fits.gz'
    fpath_full = os.path.join(
        sas_basedir, 'mangawork/manga/spectro/redux',
        drpver, plate, 'stack', fname)

    if not os.path.isfile(fpath_full):
        marvin_download([plateifu], dltype=rsscube.lower(), release=mpl_v)

    return fits.open(fpath_full)


class WaveSpecSet(object):
    def __init__(self, ltol=0.5):
        """stores wavelength specifications for spectroscopic data
        """

        self.ltol = ltol
        self.wavespecs = dict()

    def contains(self, l0, n):
        if self.match(l0, n) is not None:
            return True
        else:
            return False

    def match(self, l0, n):
        for k in self.wavespecs:
            samestart = np.isclose(l0, k[0], atol=self.ltol)
            samelen = (n == k[1])
            if samestart and samelen:
                return self.wavespecs[k]
            else:
                pass
        else:
            return None

    def add(self, l0, n):
        if len(self.wavespecs) == 0:
            self.wavespecs[(l0, n)] = 0
        else:
            max_ix = np.array(list(self.wavespecs.values())).max()
            self.wavespecs[(l0, n)] = max_ix + 1


def wavespec(wave):
    return np.min(wave), len(wave)
        

class MultispecSubtractor(object):
    def __init__(self, cubes, ivars, locs):
        self.cubes = cubes
        self.ivars = ivars
        # assume that galaxies have already been grouped 
        # by wavelength specification
        self.locs = locs
        self.wavespec_ix = self.locs['wavespec_ix'][0]

        self.combos = list(itertools.combinations(range(len(self.locs)), 2))
        self.locs_combos = [((self.locs['plateifu'][i1], 
                              self.locs['i'][i1], self.locs['j'][i1]), 
                             (self.locs['plateifu'][i2], 
                              self.locs['i'][i2], self.locs['j'][i2]))
                            for i1, i2 in self.combos]

    def getflux(self, plateifu, i, j):
        return self.cubes[plateifu][:, i, j]

    def getivar(self, plateifu, i, j):
        return self.ivars[plateifu][:, i, j]

    def get_speccombos(self):
        specs = [(self.getflux(*locs1), self.getflux(*locs2))
                 for locs1, locs2 in self.locs_combos]
        specs1, specs2 = zip(*specs)
        return np.row_stack(specs1), np.row_stack(specs2)

    def get_ivarcombos(self):
        specs = [(self.getivar(*locs1), self.getivar(*locs2))
                 for locs1, locs2 in self.locs_combos]
        specs1, specs2 = zip(*specs)
        return np.row_stack(specs1), np.row_stack(specs2)

    def subtract_combos(self):
        diffs = np.row_stack(
            [self.getflux(*locs1) - self.getflux(*locs2)
             for locs1, locs2 in self.locs_combos])

        return diffs

    def loccombos_tab(self):
        loc_data = [(*s1l, *s2l) for s1l, s2l in self.locs_combos]
        tab = t.Table(
            rows=loc_data,
            names=['plateifu1', 'i1', 'j1', 'plateifu2', 'i2', 'j2'])
        tab['wavespec_ix'] = self.wavespec_ix

        return tab


def run_galaxy_multispec(tab, wavespecs):
    # group by bin assignment
    grouped = tab.group_by('assign')

    cubes = dict()
    ivars = dict()
    cubes_wavespecs = dict()

    # this is clumsy but what the hell
    # to avoid file-closing errors, preload flux and wavelength grid data
    for plateifu in np.unique(grouped['plateifu']):
        with load_mangadrp(plateifu) as cube_drp:
            cubes[plateifu] = cube_drp['FLUX'].data
            ivars[plateifu] = cube_drp['IVAR'].data
            cubes_wavespecs[plateifu] = wavespec(cube_drp['WAVE'].data)

    # make sure wavelength specifications are stored
    for ws in cubes_wavespecs.values():
        if wavespecs.contains(*ws):
            pass
        else:
            wavespecs.add(*ws)

    cube_wavespecs_ix = {k: wavespecs.match(*ws)
                         for (k, ws) in cubes_wavespecs.items()}
    tab['wavespec_ix'] = [cube_wavespecs_ix[c] for c in tab['plateifu']]

    # group by both assigned spatial bin and wavelength specification
    grouped_complete = tab.group_by(['wavespec_ix', 'assign'])

    # loop through groups
    subtractors = [MultispecSubtractor(cubes, ivars, locs=grp)
                   for grp in grouped_complete.groups]

    locs, specs, ivars = zip(*[(s.loccombos_tab(), s.get_speccombos(),
                                s.get_ivarcombos())
                               for s in subtractors])

    return locs, specs, ivars
    

if __name__ == '__main__':
    import json

    from conf import DATALOC, DUPLICATES_DATALOC, PAIRS_DATALOC

    # set up marvin
    mpl_v = 'MPL-10'
    mconf.access = 'collab'
    mconf.release = mpl_v
    mconf.download = True

    # find tables of duplicate spaxels
    dupl_tab_fnames = glob.glob(os.path.join(DUPLICATES_DATALOC, 'dupl_*.fits'))

    # set up wavelength specifications
    wavespecs = WaveSpecSet()

    # wipe directory of all spectra records and tabular records
    for fn in glob.iglob(os.path.join(PAIRS_DATALOC, 'wavespec-*/specs_*.fits')):
        os.remove(fn)

    for fn in glob.iglob(os.path.join(PAIRS_DATALOC, 'wavespec-*/spaxtab_*.fits')):
        os.remove(fn)

    # loop through spaxel-duplicates table for each manga-id
    for fn in ProgressBar(dupl_tab_fnames):
        galname = os.path.basename(fn).split('.')[0].split('_')[1]

        dupl_tab = t.Table.read(fn)

        # if table has no spaxels in it, skip to the next one
        if np.unique(dupl_tab['assign']).size < 1:
            continue
        
        # grab spectra and their ivars,
        # and keep track of what wavelength specification they're in
        # run_galaxy_multispec returns 3 lists: tables, specs, ivars
        spec_locs, specs, ivars = run_galaxy_multispec(dupl_tab, wavespecs)
        wsgrps = [loctab['wavespec_ix'][0] for loctab in spec_locs]

        # now group by wavespec_ix
        # each wavespec_ix group contains only entries with a single wavespec
        for wavespec_ix in np.unique(wsgrps):

            # within each group, grab the corresponding entries from specs
            # and unzip them into two groups of spectra            
            specs12 = [np.stack(s_, axis=0) for ws_, s_ in zip(wsgrps, specs)
                       if ws_ == wavespec_ix]
            ivars12 = [np.stack(s_, axis=0) for ws_, s_ in zip(wsgrps, ivars)
                       if ws_ == wavespec_ix]
            wsg = t.vstack([locs_ for ws_, locs_ in zip(wsgrps, spec_locs)
                            if ws_ == wavespec_ix])
            # stack each group of spectra like rows
            specs12 = np.concatenate(specs12, axis=1)
            ivars12 = np.concatenate(ivars12, axis=1)

            specs1, specs2 = specs12[0, ...], specs12[1, ...]
            ivars1, ivars2 = ivars12[0, ...], ivars12[1, ...]

            # now append to existing records on disk
            specs_fname = os.path.join(
                PAIRS_DATALOC, 
                'wavespec_{}/specs_{}.fits'.format(wavespec_ix, galname))
            spaxtab_fname = os.path.join(
                PAIRS_DATALOC,
                'wavespec_{}/spaxtab_{}.fits'.format(wavespec_ix, galname))

            os.makedirs(os.path.dirname(specs_fname), exist_ok=True)

            specs1_hdu = fits.ImageHDU(
                data=specs1, 
                header=fits.Header(cards={'EXTNAME': 'specs1'}))
            specs2_hdu = fits.ImageHDU(
                data=specs2, 
                header=fits.Header(cards={'EXTNAME': 'specs2'}))
            ivars1_hdu = fits.ImageHDU(
                data=ivars1, 
                header=fits.Header(cards={'EXTNAME': 'ivars1'}))
            ivars2_hdu = fits.ImageHDU(
                data=ivars2, 
                header=fits.Header(cards={'EXTNAME': 'ivars2'}))
            specs_hdulist = fits.HDUList(
                [fits.PrimaryHDU(), specs1_hdu, specs2_hdu, 
                 ivars1_hdu, ivars2_hdu])
            specs_hdulist.writeto(specs_fname, overwrite=True)
            
            wsg.write(spaxtab_fname, overwrite=True)

    # finally dump all wavespecs into json
    # we're actualyl storing the inverse mapping
    with open(os.path.join(PAIRS_DATALOC, 'wavespecs.json'), 'w') as wavespecs_f:
        inverted_wavespecs = {v: k for k, v in wavespecs.wavespecs.items()}
        wavespecs_f.write(json.dumps(inverted_wavespecs))

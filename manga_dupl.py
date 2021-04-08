import numpy as np
import onehotclustering as ohc
from marvin import config as mconf
from marvin import MarvinError
from marvin.utils.general import get_drpall_table
from marvin.tools import Maps
from astropy import table as t
from astropy.utils.console import ProgressBar
import os


non_unique_group = lambda tab, _: len(tab) > 1


def get_drpall_mangaid_grouped():
    """Get drpall file and return groups based on MaNGA-ID
    """

    drpall = get_drpall_table()
    # filter acc'g to which groups have more than one entry
    drpall_mangaid = drpall.group_by('mangaid').groups.filter(
        non_unique_group)

    return drpall_mangaid


def get_plateifu_offsetmap(plateifu):
    """Retrieve sky coordinate offset map wrt object center

    Parameters
    ----------
    plateifu : str
        plate-ifu code for galaxy
    """

    dap_maps = Maps(plateifu)
    x, y = dap_maps.spx_skycoo_on_sky_x, dap_maps.spx_skycoo_on_sky_y
    # inds = np.indices(x.shape)

    return x, y


def match_plateifu_spaxels(*plateifus):
    """match spaxels between two plateifus, based on their sky position

    Parameters
    ----------
    *plateifus
        plateifus to run

    Returns
    -------
    OneHotClustering
        OneHotClustering object fitted to spaxel positions
    np.ndarray
        indices of spaxel (3 columns: IX, SPAXI, SPAXJ)
    """
    ndim = 2
    clusterer = ohc.OneHotClustering(ndim=ndim, maxdist=0.1, metric='euclidean')
    spaxel_inds = np.zeros((0, ndim + 1))

    for i, plateifu in enumerate(plateifus):
        dap_maps = Maps(plateifu)
        # coordinate and index maps
        xmap, ymap = dap_maps.spx_skycoo_on_sky_x, dap_maps.spx_skycoo_on_sky_y
        indmaps = np.indices(xmap.shape)

        # use only spaxels with good signal
        goodsnr = dap_maps.spax_snr.value >= 0.5

        # in column format
        coords = np.column_stack([xmap.value[goodsnr], ymap.value[goodsnr]])
        inds = np.column_stack([ind[goodsnr] for ind in indmaps])
        # prepend column signifying which plateifu
        inds = np.column_stack([np.full(len(inds), i), inds])

        # add new galaxy to clusterer
        clusterer.add_solve(coords)

        # append this galaxy's spaxel coordinates to the global list
        spaxel_inds = np.row_stack([spaxel_inds, inds]).astype(int)

    return clusterer, spaxel_inds


def group_plateifu_spaxels(*plateifus):
    """groups spaxels in multiple plateifus corresponding to the same galaxy

    Parameters
    ----------
    *plateifus
        plateifus to run

    Returns
    -------
    astropy.table.Table
        table containing each spaxels' parent plateifu, group assignment,
        and coords
    """
    clusterer, spaxel_inds = match_plateifu_spaxels(*plateifus)

    coordtab = t.Table(clusterer.coords, names=['dx', 'dy', 'plateifu_ix'])
    coordtab['plateifu_ix'] = coordtab['plateifu_ix'].astype(int)
    coordtab['assign'] = clusterer.assign
    coordtab['plateifu'] = (np.array(plateifus)[coordtab['plateifu_ix']])
    coordtab['plateifu'] = coordtab['plateifu'].astype('str')

    spaxtab = t.Table(spaxel_inds[:, 1:], names=['i', 'j'])

    # merge tables, group, and filter out singleton groups
    spaxassign_groups = t.hstack([coordtab, spaxtab])[
        'plateifu', 'assign', 'i', 'j'].group_by('assign').groups.filter(
        non_unique_group)

    return spaxassign_groups


if __name__ == '__main__':
    from conf import DATALOC, DUPLICATES_DATALOC

    # set up marvin
    mpl_v = 'MPL-10'
    mconf.access = 'collab'
    mconf.release = mpl_v
    mconf.download = True

    # grab drpall and group by mangaid
    drpall_mangaid_grouped = get_drpall_mangaid_grouped()

    # construct and write out table for each unique mangaid
    for grp in ProgressBar(drpall_mangaid_grouped.groups):
        dupl_fname = os.path.join(DUPLICATES_DATALOC, 'dupl_{}.fits'.format(grp['mangaid'][0]))

        if os.path.isfile(dupl_fname):
            continue
        try:
            spaxassign_groups = group_plateifu_spaxels(
                *grp['plateifu'])
        except MarvinError:
            pass
        else:
            spaxassign_groups.write(
                dupl_fname, format='fits', overwrite=True)

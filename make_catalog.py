#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from photutils import CircularAperture

os.environ['FLARE'] = '/cosma7/data/dp004/dc-wilk2/flare'

matplotlib.use('Agg')
warnings.filterwarnings('ignore')
import seaborn as sns
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from scipy.stats import binned_statistic
import phot_modules as phot
import utilities as util
import flare
from flare.photom import lum_to_M, M_to_lum
from astropy.cosmology import Planck13 as cosmo
from photutils.segmentation import SourceCatalog
import h5py
import photutils as phut
from astropy.convolution import Gaussian2DKernel
from scipy import signal
import sys
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
import time
import eritlux.simulations.imagesim as imagesim
import flare.surveys
import flare.plots.image

sns.set_context("paper")
sns.set_style('whitegrid')


regions = []
for reg in range(0, 40):
    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))

snaps = ['000_z015p000', '001_z014p000', '002_z013p000',
         '003_z012p000', '004_z011p000', '005_z010p000',
         '006_z009p000', '007_z008p000', '008_z007p000',
         '009_z006p000', '010_z005p000', '011_z004p770']

reg_snaps = []
for reg in reversed(regions):

    for snap in snaps:

        reg_snaps.append((reg, snap))


ind = int(sys.argv[1])

# Set orientation
orientation = sys.argv[2]

# Define flux and dust model types
Type = sys.argv[3]
extinction = 'default'

reg, tag = reg_snaps[ind]
snap = tag
print("Computing HLRs with orientation {o}, type {t}, and extinction {e}"
      " for region {x} and snapshot {u}".format(o=orientation, t=Type,
                                               e=extinction, x=reg, u=tag))

# Define filter
filters = [f'Hubble.ACS.{f}'
           for f in ['f435w', 'f606w', 'f775w', 'f814w', 'f850lp']] \
          + [f'Hubble.WFC3.{f}' for f in ['f105w', 'f125w', 'f140w', 'f160w']]

depths = [0.1, 1, 5, 10, 20]

# Define radii
radii_fracs = (0.2, 0.5, 0.8)

# Set mass limit
masslim = 700

z_str = tag.split('z')[1].split('p')
z = float(z_str[0] + '.' + z_str[1])

survey_id = 'XDF'  # the XDF (updated HUDF)
field_id = 'dXDF'  # deepest sub-region of XDF (defined by a mask)

# --- get field info object. This contains the filters, depths,
# image location etc. (if real image)
field = flare.surveys.surveys[survey_id].fields[field_id]

if z <= 2.8:
    csoft = 0.000474390 / 0.6777 * 1e3
else:
    csoft = 0.001802390 / (0.6777 * (1 + z)) * 1e3
    
# Define smoothing kernel for deblending
kernel_sigma = 8 / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # FWHM = 3
kernel = Gaussian2DKernel(kernel_sigma)
kernel.normalize()

arcsec_per_kpc_proper = cosmo.arcsec_per_kpc_proper(z).value

# Define width
ini_width_pkpc = 500
ini_width = ini_width_pkpc * arcsec_per_kpc_proper

thresh = 2.5

hdf = h5py.File("mock_data/flares_mock_cat_{}_{}_{}_{}.hdf5"
                .format(reg, tag, Type, orientation), "w")
print("Creating File...")

for f in filters:

    for num, depth in enumerate(depths):

        # segm_flux = {}
        # segm_flux_err = {}
        # kron_flux = {}
        # kron_flux_err = {}
        # kron_rad = {}
        # kron_hlr = {}
        # gini = {}
        # xs = {}
        # ys = {}
        # labels = {}
        obs_data = {}
        obs_img_num = {}
        obs_begin = {}
        obs_len = {}

        subf_flux = {}
        subf_img_num = {}
        subf_begin = {}
        subf_len = {}

        print("Getting SUBFIND occupancy with orientation {o}, type {t}, "
              "and extinction {e} for region {x} and "
              "snapshot {u}".format(o=orientation, t=Type, e=extinction,
                                    x=reg, u=snap))
        try:
            hdf = h5py.File("mock_data/flares_segm_{}_{}_{}_{}_{}.hdf5"
                            .format(reg, snap, Type, orientation, f), "r")
        except OSError as e:
            print(e)
            continue

        try:

            fluxes = hdf["Fluxes"][:]
            subgrpids = hdf["Part_subgrpids"][:]
            begin = hdf["Start_Index"][:]
            group_len = hdf["Group_Length"][:]
            gal_ids = set(hdf["Subgroup_IDs"][:])

            hdf.close()
        except KeyError as e:
            hdf.close()
            continue

        for depth in depths:

            if depth == depths[0]:

                flux_subfind = []

                for (img_num, beg), img_len in zip(enumerate(begin), group_len):

                    this_subgrpids = subgrpids[beg: beg + img_len]

                    subgrps, inverse_inds = np.unique(this_subgrpids,
                                                      return_inverse=True)

                    this_flux = np.zeros(subgrps.size)

                    for flux, i, subgrpid in zip(fluxes[beg: beg + img_len],
                                                 inverse_inds, this_subgrpids):
                        this_flux[i] += flux

                    this_flux = this_flux[this_flux > 0]
                    flux_subfind.extend(this_flux)

                    subf_flux.setdefault(f + "." + str(depth), []).extend(this_flux)
                    subf_img_num.setdefault(f + "." + str(depth), []).extend(
                        np.full_like(this_flux, img_num))

            hdf = h5py.File("mock_data/flares_segm_{}_{}_{}_{}.hdf5"
                            .format(reg, snap, Type, orientation),
                            "r")

            try:
                fdepth_group = hdf[str(depth)]

                imgs = fdepth_group["Images"]
                sigs = fdepth_group["Significance_Images"]

            except KeyError as e:
                print(e)
                hdf.close()
                continue

            flux_segm = []
            flux_segm_err = []

            if sigs.shape[0] == 0:
                continue

            if sigs[:].max() < thresh:
                continue

            for ind in range(imgs.shape[0]):

                sig = sigs[ind, :, :]
                img = imgs[ind, :, :]

                if sig.max() < thresh:
                    continue

                try:
                    segm = phut.detect_sources(sig, thresh, npixels=5)
                    segm = phut.deblend_sources(img, segm, npixels=5,
                                                nlevels=32, contrast=0.001)
                except TypeError as e:
                    continue

                source_cat = SourceCatalog(img, segm, error=None, mask=None,
                                           kernel=None, background=None,
                                           wcs=None, localbkg_width=0,
                                           apermask_method='correct',
                                           kron_params=(2.5, 0.0),
                                           detection_cat=None)

                tab = source_cat.to_table()
                print(tab.colnames)

                flux_segm.extend(source_cat.kron_fluxs)
                flux_segm_err.extend(source_cat.kron_fluxerr)

            hdf.close()

            flux_segm_dict.setdefault(f + "." + str(depth), []).extend(
                flux_segm)
            flux_segmerr_dict.setdefault(f + "." + str(depth), []).extend(
                flux_segm_err)

        fdepth_group = f_group.create_group(str(depth))

        dset = fdepth_group.create_dataset("Images", data=imgs,
                                      dtype=imgs.dtype,
                                      shape=imgs.shape,
                                      compression="gzip")
        dset.attrs["units"] = "$nJy$"

        # dset = fdepth_group.create_dataset("Segmentation_Maps", data=segms,
        #                               dtype=segms.dtype,
        #                               shape=segms.shape,
        #                               compression="gzip")
        # dset.attrs["units"] = "None"

        dset = fdepth_group.create_dataset("Significance_Images", data=sigs,
                                      dtype=sigs.dtype,
                                      shape=sigs.shape,
                                      compression="gzip")
        dset.attrs["units"] = "None"

    fluxes = np.array(fluxes)
    subgrpids = np.array(subgrpids)
    smls = np.array(smls)
    star_pos = np.array(star_pos)

    dset = f_group.create_dataset("Subgroup_IDs",
                                            data=gal_haloids,
                                            dtype=gal_haloids.dtype,
                                            shape=gal_haloids.shape,
                                            compression="gzip")
    dset.attrs["units"] = "None"

    dset = f_group.create_dataset("Galaxy Mass",
                                  data=gal_mass,
                                  dtype=gal_mass.dtype,
                                  shape=gal_mass.shape,
                                  compression="gzip")
    dset.attrs["units"] = "$M_\odot$"

    dset = f_group.create_dataset("Start_Index", data=begin,
                                  dtype=begin.dtype,
                                  shape=begin.shape,
                                  compression="gzip")
    dset.attrs["units"] = "None"

    dset = f_group.create_dataset("Group_Mass", data=grp_mass,
                                  dtype=grp_mass.dtype,
                                  shape=grp_mass.shape,
                                  compression="gzip")
    dset.attrs["units"] = "$M_\odot$"

    dset = f_group.create_dataset("Smoothing_Length", data=smls,
                                  dtype=smls.dtype,
                                  shape=smls.shape,
                                  compression="gzip")
    dset.attrs["units"] = "Mpc"

    dset = f_group.create_dataset("Star_Pos", data=star_pos,
                                  dtype=star_pos.dtype,
                                  shape=star_pos.shape,
                                  compression="gzip")
    dset.attrs["units"] = "kpc"

    dset = f_group.create_dataset("Fluxes", data=fluxes,
                                  dtype=fluxes.dtype,
                                  shape=fluxes.shape,
                                  compression="gzip")
    dset.attrs["units"] = "nJy"

    dset = f_group.create_dataset("Part_subgrpids", data=subgrpids,
                                  dtype=subgrpids.dtype,
                                  shape=subgrpids.shape,
                                  compression="gzip")
    dset.attrs["units"] = "None"

    dset = f_group.create_dataset("Group_Length", data=Slen,
                                  dtype=Slen.dtype,
                                  shape=Slen.shape,
                                  compression="gzip")
    dset.attrs["units"] = "None"

hdf.close()

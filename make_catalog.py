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
import gc

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

z_str = snap.split('z')[1].split('p')
z = float(z_str[0] + '.' + z_str[1])

# Define filter
filters = [f'Hubble.ACS.{f}'
           for f in ['f435w', 'f606w', 'f775w', 'f814w', 'f850lp']] \
          + [f'Hubble.WFC3.{f}' for f in ['f105w', 'f125w', 'f140w', 'f160w']]

depths = [0.1, 1, 5, 10, 20, "SUBFIND"]

# Remove filters beyond the lyman break
detect_filters = []
for f in filters:
    f_split = f.split(".")
    inst = f.split(".")[1]
    filt = f.split(".")[-1]
    if len(filt) == 5:
        wl = int(filt[1:-1])
    else:
        wl = int(filt[1:-2])
    if inst == "ACS":
        if wl * 10 > (912 * (1 + z)):
            detect_filters.append(f)
    else:
        if wl * 100 > (912 * (1 + z)):
            detect_filters.append(f)

print("Lyman break at", 912 * (1 + z), "A")
print(detect_filters)

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

quantities = ('label', 'xcentroid', 'ycentroid', 'area',
              "eccentricity", "ellipticity", "gini",
              'segment_flux', 'segment_fluxerr', 'kron_flux', 'kron_fluxerr',
              "inertia_tensor", "kron_radius")
units = {'label': "None", 'xcentroid': "pixels", 'ycentroid': "pixels",
         'area': "pixels$^2$", "eccentricity": "None", "ellipticity": "None",
         "gini": "None", 'segment_flux': "nJy", 'segment_fluxerr': "nJy",
         'kron_flux': "nJy", 'kron_fluxerr': "nJy",
         "inertia_tensor": "None", "kron_radius": "pixels",
         "Kron_HLR": "pkpc", 'Fluxes': "nJy", "Start_Index": "None",
         "Image_ID": "None", "Image_Length": "None", "SNR_segm": "None",
         "SNR_Kron": "None"}

hdf_cat = h5py.File("mock_data/flares_mock_cat_{}_{}_{}_{}.hdf5"
                    .format(reg, tag, Type, orientation), "w")
print("Creating File...")

obs_data = {}

subf_data = {}

subf_flux = {}
subf_img_num = {}
subf_begin = {}
subf_len = {}

for num, depth in enumerate(depths):

    if depth == "SUBFIND":

        for f in filters:

            # --- initialise ImageCreator object
            image_creator = imagesim.Idealised(f, field)

            arc_res = image_creator.pixel_scale
            kpc_res = arc_res / arcsec_per_kpc_proper

            print("Getting sources with orientation {o}, type {t}, "
                  "and extinction {e} for region {x}, "
                  "snapshot {u}, filter {i}, and depth {d}"
                  .format(o=orientation, t=Type, e=extinction, x=reg,
                          u=snap, i=f, d=depth))

            try:
                hdf = h5py.File("mock_data/flares_segm_{}_{}_{}_{}_{}.hdf5"
                                .format(reg, snap, Type, orientation, f), "r")
            except OSError as e:
                print(e)
                continue

            try:
                fdepth_group = hdf[str(depth)]

                fluxes = fdepth_group["Fluxes"][:]
                subgrpids = fdepth_group["Part_subgrpids"][:]
                begin = fdepth_group["Start_Index"][:]
                group_len = fdepth_group["Image_Length"][:]
                gal_ids = set(fdepth_group["Subgroup_IDs"][:])

                hdf.close()
            except KeyError as e:
                print(e)
                hdf.close()
                continue

            for (img_num, beg), img_len in zip(enumerate(begin), group_len):

                this_subgrpids = subgrpids[beg: beg + img_len]

                subgrps, inverse_inds = np.unique(this_subgrpids,
                                                  return_inverse=True)

                this_flux = np.zeros(subgrps.size)

                for flux, i, subgrpid in zip(fluxes[beg: beg + img_len],
                                             inverse_inds, this_subgrpids):
                    this_flux[i] += flux

                this_flux = this_flux[this_flux > 0]

                subf_data.setdefault(f + "." + str(depth), {}).setdefault("Fluxes", []).extend(this_flux)
                subf_data[f + "." + str(depth)].setdefault("Start_Index", []).append(len(subf_data[f + "." + str(depth)]["Fluxes"]))
                subf_data[f + "." + str(depth)].setdefault("Image_Length", []).append(len(this_flux))
                subf_data[f + "." + str(depth)].setdefault("Image_ID", []).extend(np.full_like(this_flux, img_num))

    else:

        for f in detect_filters:

            # --- initialise ImageCreator object
            image_creator = imagesim.Idealised(f, field)

            arc_res = image_creator.pixel_scale
            kpc_res = arc_res / arcsec_per_kpc_proper

            print("Creating detection image for filter {i}, and depth {d}"
                  .format(i=f, d=depth))

            hdf = h5py.File("mock_data/flares_segm_{}_{}_{}_{}_{}.hdf5"
                            .format(reg, snap, Type, orientation, f), "r")

            try:
                fdepth_group = hdf[str(depth)]

                imgs = fdepth_group["Images"][...]
                noises = fdepth_group["Noise_value"][...]

                if f == detect_filters[0]:

                    detection_img = np.zeros_like(imgs)
                    weight_img = np.zeros_like(noises)
                    noise_img = np.zeros_like(noises)

                detection_img += (imgs / noises[:, None, None]**2)
                weight_img += (1 / noises**2)
                noise_img += (1 / noises)

                hdf.close()

            except KeyError as e:
                print(e)
                hdf.close()
                detection_img = None
                weight_img = None
                noise_img = None

                continue

        detection_img /= weight_img[:, None, None]
        noise_img /= weight_img

        sig = detection_img / noise_img[:, None, None]

        del noise_img
        del weight_img
        gc.collect()

        for f in filters:

            # --- initialise ImageCreator object
            image_creator = imagesim.Idealised(f, field)

            arc_res = image_creator.pixel_scale
            kpc_res = arc_res / arcsec_per_kpc_proper

            print("Getting sources with orientation {o}, type {t}, "
                  "and extinction {e} for region {x}, "
                  "snapshot {u}, filter {i}, and depth {d}"
                  .format(o=orientation, t=Type, e=extinction, x=reg,
                          u=snap, i=f, d=depth))

            hdf = h5py.File("mock_data/flares_segm_{}_{}_{}_{}_{}.hdf5"
                            .format(reg, snap, Type, orientation, f), "r")

            try:
                ijk = hdf["Cell_Image_Number"][:]

                fdepth_group = hdf[str(depth)]

                imgs = fdepth_group["Images"]
                img_ids = fdepth_group["Image_ID"][...]
                noises = fdepth_group["Noise_value"]

            except KeyError as e:
                print(e)
                hdf.close()
                continue

            res = imgs.shape[-1]

            for img_id in img_ids:
                img = imgs[img_id, :, :]
                det_img = detection_img[img_id, :, :]
                sig_img = sig[img_id, :, :]

            # for k in range(ijk.shape[2]):
            #     img = np.zeros((ijk.shape[0] * res, ijk.shape[1] * res),
            #                    dtype=np.float32)
            #     det_img = np.zeros((ijk.shape[0] * res, ijk.shape[1] * res),
            #                        dtype=np.float32)
            #     sig_img = np.zeros((ijk.shape[0] * res, ijk.shape[1] * res),
            #                        dtype=np.float32)
            #     for i in range(ijk.shape[0]):
            #         for j in range(ijk.shape[1]):
            #             img_id = ijk[i, j, k]
            #             if img_id >= 0:
            #                 det_img[i * res: (i + 1) * res, j * res: (j + 1) * res] += detection_img[img_id, :, :]
            #                 sig_img[i * res: (i + 1) * res, j * res: (j + 1) * res] += sig[img_id, :, :]
            #                 img[i * res: (i + 1) * res, j * res: (j + 1) * res] += imgs[img_id, :, :]

                try:
                    segm = phut.detect_sources(sig_img, thresh, npixels=5)
                    segm = phut.deblend_sources(det_img, segm,
                                                npixels=5, nlevels=32,
                                                contrast=0.001)
                except TypeError as e:
                    print(e)
                    continue

                n = noises[0]

                source_cat = SourceCatalog(img, segm,
                                           error=None, mask=None,
                                           kernel=None, background=None,
                                           wcs=None, localbkg_width=0,
                                           apermask_method='correct',
                                           kron_params=(2.5, 0.0),
                                           detection_cat=None)

                try:
                    obs_data.setdefault(f + "." + str(depth), {}).setdefault("Kron_HLR", []).extend(source_cat.fluxfrac_radius(0.5) * kpc_res)
                except ValueError as e:
                    print(e)
                    continue

                tab = source_cat.to_table(columns=quantities)
                for key in tab.colnames:
                    obs_data[f + "." + str(depth)].setdefault(key, []).extend(tab[key])
                obs_data[f + "." + str(depth)].setdefault("SNR_segm", []).extend(tab['segment_flux'] / n)
                obs_data[f + "." + str(depth)].setdefault("SNR_Kron", []).extend(tab['kron_flux'] / n)
                obs_data[f + "." + str(depth)].setdefault("Image_ID", []).extend(np.full(tab["label"].size, img_id))
                obs_data[f + "." + str(depth)].setdefault("Start_Index", []).append(len(obs_data[f + "." + str(depth)]["Image_ID"]))
                obs_data[f + "." + str(depth)].setdefault("Image_Length", []).append(tab["label"].size)

            hdf.close()

for f in filters:
    f_cat_group = hdf_cat.create_group(f)
    for num, depth in enumerate(depths):

        fdepth_cat_group = f_cat_group.create_group(str(depth))
        if f + "." + str(depth) in obs_data.keys():
            for key, val in obs_data[f + "." + str(depth)].items():

                print("Writing out", key, "for", f, depth)

                try:
                    val = np.array(val)
                except TypeError:
                    val = np.array([i.value for i in val])

                dset = fdepth_cat_group.create_dataset(key,
                                                        data=val,
                                                        dtype=val.dtype,
                                                        shape=val.shape,
                                                        compression="gzip")
                dset.attrs["units"] = units[key]

        if depth == "SUBFIND":

            if f + "." + str(depth) in subf_data.keys():
                for key, val in subf_data[f + "." + str(depth)].items():

                    print("Writing out", key, "for", f, depth)

                    val = np.array(val)

                    dset = fdepth_cat_group.create_dataset(key,
                                                           data=val,
                                                           dtype=val.dtype,
                                                           shape=val.shape,
                                                           compression="gzip")
                    dset.attrs["units"] = units[key]


hdf_cat.close()

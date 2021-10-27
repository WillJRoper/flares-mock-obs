#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import os
import warnings

import matplotlib
import numpy as np

os.environ['FLARE'] = '/cosma7/data/dp004/dc-wilk2/flare'

matplotlib.use('Agg')
warnings.filterwarnings('ignore')
import seaborn as sns
from astropy.cosmology import Planck13 as cosmo
from photutils.segmentation import SourceCatalog
import h5py
import photutils as phut
from astropy.convolution import Gaussian2DKernel
import sys
import utilities as util
from flare.photom import m_to_flux, flux_to_m
import eritlux.simulations.imagesim as imagesim
import flare.surveys as survey
import gc
import mpi4py
from mpi4py import MPI

mpi4py.rc.recv_mprobe = False

# Initializations and preliminaries
comm = MPI.COMM_WORLD  # get MPI communicator object
size = comm.size  # total number of processes
rank = comm.rank  # rank of this process

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
if rank == 0:
    print("Making catalog with orientation {o}, type {t}, and extinction {e}"
          " for region {x} and snapshot {u}".format(o=orientation, t=Type,
                                                    e=extinction, x=reg,
                                                    u=tag))

z_str = snap.split('z')[1].split('p')
z = float(z_str[0] + '.' + z_str[1])

# Define filter
filters = [f'Hubble.ACS.{f}'
           for f in ['f435w', 'f606w', 'f775w', 'f814w', 'f850lp']] \
          + [f'Hubble.WFC3.{f}' for f in ['f105w', 'f125w', 'f140w', 'f160w']]

# Set up depths relative to the Hubble Xtreme deep field
XDF_depth_m = 31.2
XDF_depth_flux = m_to_flux(XDF_depth_m)
depths = [XDF_depth_flux * 0.01, XDF_depth_flux * 0.1,
          XDF_depth_flux, 10 * XDF_depth_flux, 100 * XDF_depth_flux]
depths_m = [flux_to_m(d) for d in depths]

depths.append("SUBFIND")
depths_m.append("SUBFIND")

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
if rank == 0:
    print("Lyman break at", 912 * (1 + z), "A")
    print("Filters redder then the Lyman break:", detect_filters)

# Define radii
radii_fracs = (0.2, 0.5, 0.8)

z_str = tag.split('z')[1].split('p')
z = float(z_str[0] + '.' + z_str[1])

survey_id = 'XDF'  # the XDF (updated HUDF)
field_id = 'dXDF'  # deepest sub-region of XDF (defined by a mask)

# Get field info object. This contains the filters, depths,
# image location etc. (if real image)
field = survey.surveys[survey_id].fields[field_id]

# Define smoothing kernel for deblending
kernel_sigma = 8 / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # FWHM = 3
kernel = Gaussian2DKernel(kernel_sigma)
kernel.normalize()

# Get the conversion between arcseconds and pkpc at this redshift
arcsec_per_kpc_proper = cosmo.arcsec_per_kpc_proper(z).value

# Define widths
full_ini_width_kpc = 30000
full_ini_width = full_ini_width_kpc * arcsec_per_kpc_proper
ini_width = 20
ini_width_pkpc = ini_width / arcsec_per_kpc_proper

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
         "SNR_Kron": "None", "Kron_HMR": "M_\odot"}

if rank == 0:
    hdf_cat = h5py.File("mock_data/flares_mock_cat_{}_{}_{}_{}.hdf5"
                        .format(reg, tag, Type, orientation), "w")
    print("Creating File...")
else:
    hdf_cat = None

obs_data = {}

subf_data = {}

exists = os.path.isfile("mock_data/flares_segm_{}_{}_{}_{}_{}.hdf5"
                        .format(reg, snap, Type, orientation,
                            detect_filters[0]))

if exists:
    hdf = h5py.File("mock_data/flares_segm_{}_{}_{}_{}_{}.hdf5"
                    .format(reg, snap, Type, orientation,
                            detect_filters[0]), "r")

    fdepth_group = hdf[str(depths[0])]

    noises = fdepth_group["Noise_value"]

    nimg = noises.shape[0]

    hdf.close()

    img_ids = np.linspace(0, nimg - 1, nimg)

    if nimg < size:
        if rank < nimg:
            my_img_ids = np.array([rank, ], dtype=int)
        else:
            print("Rank", rank, "has no images")
            my_img_ids = np.array([], dtype=int)
    else:
        rank_img_bins = np.linspace(0, nimg, size + 1)
        my_img_ids = np.arange(rank_img_bins[rank], rank_img_bins[rank + 1], 1,
                               dtype=int)
else:
    print("Input file does not exist")
    my_img_ids = np.array([], dtype=int)

if len(my_img_ids) > 0:

    print("Rank:", rank, "of", size - 1,
          "My Images: (" + str(my_img_ids[0]), "->",
          str(my_img_ids[-1]) + ")",
          "Total:", nimg)

    for num, depth in enumerate(depths):

        if depth == "SUBFIND":

            for f in filters:

                # --- initialise ImageCreator object
                image_creator = imagesim.Idealised(f, field)

                arc_res = image_creator.pixel_scale
                kpc_res = arc_res / arcsec_per_kpc_proper

                try:
                    hdf = h5py.File(
                        "mock_data/flares_segm_{}_{}_{}_{}_{}.hdf5"
                            .format(reg, snap, Type, orientation, f), "r")
                except OSError as e:
                    # print(e)
                    continue

                try:
                    fdepth_group = hdf[str(depth)]

                    fluxes = fdepth_group["Fluxes"][:]
                    subgrpids = fdepth_group["Part_subgrpids"][:]
                    star_pos = fdepth_group["Star_Pos"][:]
                    begin = fdepth_group["Start_Index"][:]
                    group_len = fdepth_group["Image_Length"][:]
                    gal_ids = set(fdepth_group["Subgroup_IDs"][:])

                    hdf.close()
                except KeyError as e:
                    print(e)
                    hdf.close()
                    continue

                begins = begin[my_img_ids]
                lens = group_len[my_img_ids]

                for img_num, beg, img_len in zip(my_img_ids, begins, lens):

                    this_subgrpids = subgrpids[beg: beg + img_len]
                    this_pos = star_pos[beg: beg + img_len, :]

                    subgrps, inverse_inds = np.unique(this_subgrpids,
                                                      return_inverse=True)

                    this_flux = np.zeros(subgrps.size)
                    this_poss = {}
                    this_fluxs = {}

                    for flux, i, subgrpid, pos in zip(
                            fluxes[beg: beg + img_len],
                            inverse_inds, this_subgrpids, this_pos):

                        this_flux[i] += flux
                        this_poss.setdefault(i, []).append(pos)
                        this_fluxs.setdefault(i, []).append(flux)

                    for i in this_poss:
                        com = np.average(this_poss[i], weights=this_fluxs[i],
                                         axis=0)
                        this_poss[i] = np.array(this_poss[i]) - com
                        this_fluxs[i] = np.array(this_fluxs[i])

                    this_rads = {i: np.sqrt(this_poss[i][:, 0]**2
                                            + this_poss[i][:, 1]**2
                                            + this_poss[i][:, 2]**2)
                                 for i in this_poss.keys()}

                    this_hlrs = [util.calc_light_mass_rad(this_rads[i],
                                                          this_fluxs[i],
                                                          radii_frac=0.5)
                                 for i in this_fluxs.keys()]

                    subf_data.setdefault(f + "." + str(depth),
                                         {}).setdefault(
                        "Fluxes", []).extend(this_flux)
                    subf_data.setdefault(f + "." + str(depth),
                                         {}).setdefault(
                        "HLRs", []).extend(this_hlrs)
                    subf_data[f + "." + str(depth)].setdefault(
                        "Start_Index",
                        []).append(
                        len(subf_data[f + "." + str(depth)]["Fluxes"]))
                    subf_data[f + "." + str(depth)].setdefault(
                        "Image_Length",
                        []).append(
                        len(this_flux))
                    subf_data[f + "." + str(depth)].setdefault("Image_ID",
                                                               []).extend(
                        np.full_like(this_flux, img_num))

        else:

            for f in detect_filters:

                # --- initialise ImageCreator object
                image_creator = imagesim.Idealised(f, field)

                arc_res = image_creator.pixel_scale
                kpc_res = arc_res / arcsec_per_kpc_proper
                if rank == 0:
                    print("Creating detection image for "
                          "filter {i}, and depth {d}"
                            .format(i=f, d=depth))

                hdf = h5py.File("mock_data/flares_segm_{}_{}_{}_{}_{}.hdf5"
                                .format(reg, snap, Type, orientation, f),
                                "r")

                try:
                    fdepth_group = hdf[str(depth)]

                    imgs = fdepth_group["Images"][...]
                    noises = fdepth_group["Noise_value"][...]

                    if f == detect_filters[0]:
                        detection_img = np.zeros_like(imgs)
                        weight_img = np.zeros_like(noises)
                        noise_img = np.zeros_like(noises)

                    detection_img += (imgs / noises[:, None, None] ** 2)
                    weight_img += (1 / noises ** 2)
                    noise_img += (1 / noises)

                    hdf.close()

                except KeyError as e:
                    # print(e)
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

            for img_id in my_img_ids:
                det_img = detection_img[img_id, :, :]
                sig_img = sig[img_id, :, :]

                try:
                    segm = phut.detect_sources(sig_img, thresh, npixels=5,
                                               kernel=kernel)
                    segm = phut.deblend_sources(det_img, segm,
                                                npixels=5, nlevels=16,
                                                contrast=0.01,
                                                kernel=kernel)
                except TypeError as e:
                    # print(e)
                    for f in filters:
                        obs_data.setdefault(f + "." + str(depth), {})
                    continue

                for f in filters:

                    obs_data.setdefault(f + "." + str(depth), {})

                    # --- initialise ImageCreator object
                    image_creator = imagesim.Idealised(f, field)

                    arc_res = image_creator.pixel_scale
                    kpc_res = arc_res / arcsec_per_kpc_proper

                    single_pix_area = kpc_res**2

                    hdf = h5py.File(
                        "mock_data/flares_segm_{}_{}_{}_{}_{}.hdf5"
                            .format(reg, snap, Type, orientation, f), "r")

                    try:

                        fdepth_group = hdf[str(depth)]

                        imgs = fdepth_group["Images"]
                        mimgs = fdepth_group["Mass_Images"]
                        noises = fdepth_group["Noise_value"]

                    except KeyError as e:
                        # print(e)
                        hdf.close()
                        continue

                    n = np.max(
                        noises[img_id])  # noise images have 1 unique value
                    res = imgs.shape[-1]

                    # Get images
                    img = imgs[img_id, :, :]
                    mimg = mimgs[img_id, :, :]

                    # ================= Flux =================

                    source_cat = SourceCatalog(img, segm,
                                               error=None, mask=None,
                                               kernel=kernel,
                                               background=None,
                                               wcs=None, localbkg_width=0,
                                               apermask_method='correct',
                                               kron_params=(2.5, 0.0),
                                               detection_cat=None)

                    tab = source_cat.to_table(columns=quantities)
                    for key in tab.colnames:
                        obs_data[f + "." + str(depth)].setdefault(key,
                                                                  []).extend(
                            tab[key])
                    obs_data[f + "." + str(depth)].setdefault("SNR_segm",
                                                              []).extend(
                        tab['segment_flux'] / n)
                    obs_data[f + "." + str(depth)].setdefault("SNR_Kron",
                                                              []).extend(
                        tab['kron_flux'] / n)
                    obs_data[f + "." + str(depth)].setdefault("Image_ID",
                                                              []).extend(
                        np.full(tab["label"].size, img_id))
                    obs_data[f + "." + str(depth)].setdefault(
                        "Start_Index",
                        []).append(
                        len(obs_data[f + "." + str(depth)]["Image_ID"]))
                    obs_data[f + "." + str(depth)].setdefault(
                        "Image_Length",
                        []).append(
                        tab["label"].size)

                    this_pix_hlr = []
                    for lab in tab["label"]:
                        this_pix_hlr.append(
                            util.get_pixel_hlr(img[segm.data == lab],
                                               single_pix_area,
                                               radii_frac=0.5))

                    obs_data[f + "." + str(depth)].setdefault(
                        "Pix_HLR", []).extend(this_pix_hlr)

                    try:
                        obs_data[f + "." + str(depth)].setdefault(
                            "Kron_HLR", []).extend(
                            source_cat.fluxfrac_radius(0.5) * kpc_res)
                    except (ValueError, TypeError) as e:
                        obs_data[f + "." + str(depth)].setdefault(
                            "Kron_HLR", []).extend(
                            np.zeros_like(tab['kron_flux']))
                        # print(e)
                        hdf.close()

                    # ================= Mass =================

                    source_cat = SourceCatalog(mimg, segm,
                                               error=None, mask=None,
                                               kernel=kernel,
                                               background=None,
                                               wcs=None, localbkg_width=0,
                                               apermask_method='correct',
                                               kron_params=(2.5, 0.0),
                                               detection_cat=None)

                    tab = source_cat.to_table(columns=quantities)
                    for key in tab.colnames:
                        obs_data[f + "." + str(depth)].setdefault(
                            "Mass_" + key,
                            []).extend(
                            tab[key])

                    hdf.close()

                    try:
                        obs_data[f + "." + str(depth)].setdefault(
                            "Kron_HMR", []).extend(
                            source_cat.fluxfrac_radius(0.5) * kpc_res)
                    except (ValueError, TypeError) as e:
                        obs_data[f + "." + str(depth)].setdefault(
                            "Kron_HMR", []).extend(np.zeros_like(tab['kron_flux']))
                        continue

collected_subf_data = comm.gather(subf_data, root=0)
collected_obs_data = comm.gather(obs_data, root=0)
if rank == 0 and exists:

    nres = 0
    for i in collected_subf_data:
        try:
            nres += len(
                i[filters[0] + "." + "SUBFIND"]["Start_Index"])
        except KeyError:
            continue
    print("Collected", nres, "Subfind results")
    nres = 0
    for i in collected_obs_data:
        try:
            nres += len(
                i[filters[0] + "." + str(depths[0])]["Start_Index"])
        except KeyError:
            continue
    print("Collected", nres, "Observational results")

    out_subf_data = {}
    for f in filters:
        d = "SUBFIND"
        out_subf_data.setdefault(f + "." + str(d), {})
        for key in collected_subf_data[0][f + "." + str(d)]:
            for res in collected_subf_data:
                if len(res) == 0:
                    continue
                if key == "Start_Index":
                    out_subf_data[f + "." + str(d)].setdefault(key, []).extend(
                        np.array(res[f + "." + str(d)][key]) + len(
                            out_subf_data[f + "." + str(d)][
                                "Start_Index"]))
                else:
                    out_subf_data[f + "." + str(d)].setdefault(key,
                                                                   []).extend(
                        res[f + "." + str(d)][key])

    out_obs_data = {}
    for f in filters:
        for d in depths:
            if d == "SUBFIND":
                continue
            out_obs_data.setdefault(f + "." + str(d), {})
            for key in collected_obs_data[0][f + "." + str(d)]:
                for res in collected_obs_data:
                    if len(res) == 0 or not 'Kron_HLR' in res[f + "." + str(d)]:
                        continue
                    if key == "Start_Index":
                        out_obs_data[f + "." + str(d)].setdefault(key,
                                                                      []).extend(
                            np.array(res[f + "." + str(d)][key]) + len(
                                out_obs_data[f + "." + str(d)][
                                    "Start_Index"]))
                    else:
                        out_obs_data[f + "." + str(d)].setdefault(key,
                                                                      []).extend(
                            res[f + "." + str(d)][key])

    for f in filters:
        f_cat_group = hdf_cat.create_group(f)
        for num, depth in enumerate(depths):

            fdepth_cat_group = f_cat_group.create_group(str(depth))
            if f + "." + str(depth) in out_obs_data.keys():
                for key, val in out_obs_data[f + "." + str(depth)].items():

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
                    # dset.attrs["units"] = units[key]

            if depth == "SUBFIND":

                if f + "." + str(depth) in out_subf_data.keys():
                    for key, val in out_subf_data[
                        f + "." + str(depth)].items():
                        print("Writing out", key, "for", f, depth)

                        val = np.array(val)

                        dset = fdepth_cat_group.create_dataset(key,
                                                               data=val,
                                                               dtype=val.dtype,
                                                               shape=val.shape,
                                                               compression="gzip")
                        # dset.attrs["units"] = units[key]

    hdf_cat.close()

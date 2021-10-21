#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import os
import warnings

import matplotlib
import numpy as np

os.environ['FLARE'] = '/cosma7/data/dp004/dc-wilk2/flare'

matplotlib.use('Agg')
warnings.filterwarnings('ignore')
import seaborn as sns
import phot_modules as phot
import utilities as util
from astropy.cosmology import Planck13 as cosmo
import h5py
from scipy import signal
from flare.photom import m_to_flux, flux_to_m
import sys
from scipy.spatial import cKDTree
import eritlux.simulations.imagesim as imagesim
import flare.surveys as survey

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

filter_ind = int(sys.argv[4])

reg, tag = reg_snaps[ind]
print("Creating images with orientation {o}, type {t}, and extinction {e}"
      " for region {x} and snapshot {u}".format(o=orientation, t=Type,
                                                e=extinction, x=reg, u=tag))

# Define filter
filters = [f'Hubble.ACS.{f}'
           for f in ['f435w', 'f606w', 'f775w', 'f814w', 'f850lp']] \
          + [f'HST.WFC3.{f}' for f in ['f105w', 'f125w', 'f140w', 'f160w']]

# Set up depths relative to the Xtreme deep field
XDF_depth_m = 31.2
XDF_depth_flux = m_to_flux(XDF_depth_m)
depths = [XDF_depth_flux * 0.1, XDF_depth_flux,
          2 * XDF_depth_flux, 10 * XDF_depth_flux]
depths_m = [flux_to_m(XDF_depth_flux * 0.01), flux_to_m(XDF_depth_flux * 0.1),
            flux_to_m(XDF_depth_flux), flux_to_m(10 * XDF_depth_flux),
            flux_to_m(100 * XDF_depth_flux)]

z_str = tag.split('z')[1].split('p')
z = float(z_str[0] + '.' + z_str[1])

survey_id = 'XDF'  # the XDF (updated HUDF)
field_id = 'dXDF'  # deepest sub-region of XDF (defined by a mask)

# Get field info object. This contains the filters, depths,
# image location etc. (if real image)
field = survey.surveys[survey_id].fields[field_id]

# Get the conversion between arcseconds and pkpc at this redshift
arcsec_per_kpc_proper = cosmo.arcsec_per_kpc_proper(z).value

# Define widths
full_ini_width_kpc = 30000
full_ini_width = full_ini_width_kpc * arcsec_per_kpc_proper
ini_width = 15
ini_width_pkpc = ini_width / arcsec_per_kpc_proper

f = filters[filter_ind]

hdf = h5py.File("mock_data/flares_segm_{}_{}_{}_{}_{}.hdf5"
                .format(reg, tag, Type, orientation, f), "w")
print("Creating File...")

# --- initialise ImageCreator object
image_creator = imagesim.Idealised(f, field)

arc_res = image_creator.pixel_scale
arc_res_kpc = arc_res / arcsec_per_kpc_proper

# Compute the resolution
ini_npix = ini_width / arc_res
ini_full_npix = full_ini_width / arc_res
npix = int(np.ceil(ini_npix))
full_npix = int(np.ceil(ini_full_npix))

if npix % 2 == 0:
    npix += 1

# Compute the new widths
width = arc_res * npix
full_width = arc_res * full_npix
width_pkpc = width / arcsec_per_kpc_proper
full_width_pkpc = full_width / arcsec_per_kpc_proper

print("Filter:", f)
print("Image width and resolution (in arcseconds):", width, arc_res)
print("Image width and resolution (in pkpc):", width_pkpc, arc_res_kpc)
print("Image width (in pixels):", npix)
print("Region width (in arcseconds):", full_width)
print("Region width (in pkpc):", full_width_pkpc)
print("Region width (in pixels):", full_npix)

# Kappa with DTM 0.0795, BC_fac=1., without 0.0063 BC_fac=1.25
reg_dict = phot.flux(reg, kappa=0.0795, tag=tag, BC_fac=1,
                     IMF='Chabrier_300',
                     filters=(f,), Type=Type, log10t_BC=7.,
                     extinction=extinction, orientation=orientation,
                     width=width_pkpc / 1000)

print("Got the dictionary for the region's groups:",
      len(reg_dict) - 5, "groups to  test")

# Extract region centre and radius
region_cent, region_rad = reg_dict["region_cent"], reg_dict["region_radius"]

# Define pixel area in pkpc
single_pixel_area = arc_res * arc_res \
                    / (arcsec_per_kpc_proper * arcsec_per_kpc_proper)

# Define range and extent for the images in arc seconds
imgrange = ((-width / 2, width / 2), (-width / 2, width / 2))
imgextent = [-width / 2, width / 2, -width / 2, width / 2]
region_extent = [region_cent[0] - region_rad, region_cent[0] + region_rad,
                 region_cent[1] - region_rad, region_cent[1] + region_rad,
                 region_cent[2] - region_rad, region_cent[2] + region_rad]

# # Set up aperture objects
# positions = [(npix / 2, npix / 2)]
# app_radii = np.linspace(0.001, npix / 4, 100)
# apertures = [CircularAperture(positions, r=r) for r in app_radii]

# Define x and y positions of pixels
X, Y = np.meshgrid(np.linspace(imgrange[0][0], imgrange[0][1], npix),
                   np.linspace(imgrange[1][0], imgrange[1][1], npix))

# Define pixel position array for the KDTree
pix_pos = np.zeros((X.size, 2))
pix_pos[:, 0] = X.ravel()
pix_pos[:, 1] = Y.ravel()

# Build KDTree
tree = cKDTree(pix_pos)

print("Pixel tree built")

psf = imagesim.create_PSF(f, field, npix)

image_keys = [k for k in reg_dict.keys() if type(k) is int]

gal_mass = reg_dict["gal_ms"]
gal_haloids = reg_dict["gal_haloids"]

# hdf.create_dataset("Image_Centres", data=cents, dtype=cents.dtype,
#                    shape=cents.shape, compression="gzip")
# hdf.create_dataset("Cell_Image_Number", data=ijk, dtype=ijk.dtype,
#                    shape=ijk.shape, compression="gzip")
hdf.create_dataset("Galaxy_Mass", data=gal_mass, dtype=gal_mass.dtype,
                   shape=gal_mass.shape, compression="gzip")
hdf.create_dataset("Galaxy_IDs", data=gal_haloids, dtype=gal_haloids.dtype,
                   shape=gal_haloids.shape, compression="gzip")
hdf.attrs["npix"] = npix
hdf.attrs["region_npix"] = full_npix
hdf.attrs["region_extent"] = region_extent

for num, depth in enumerate(depths):

    field.depths[f] = depth

    # --- initialise ImageCreator object
    image_creator = imagesim.Idealised(f, field)

    print("Creating image for Filter, Pixel noise, Depth:",
          f, image_creator.pixel.noise, field.depths[f])

    fdepth = f + "." + str(depth)

    imgs = []
    noise = []
    img_cop = []

    begin = np.full(len(image_keys), np.nan, dtype=np.int32)
    Slen = np.full(len(image_keys), np.nan, dtype=np.int32)
    ijk = np.full((len(image_keys), 3), np.nan, dtype=np.int32)
    smls = []
    fluxes = []
    subgrpids = []
    star_pos = []

    failed = 0
    segm_sources = 0

    for key in image_keys:

        if "coords" not in reg_dict[key].keys():
            continue

        ind = int(key)
        print(ind)
        this_cop = reg_dict[key]["cent"] * 10 ** 3

        # Find the pixel in the region image this occupies
        i = (this_cop[0] - region_extent[0]) * arc_res_kpc
        j = (this_cop[1] - region_extent[2]) * arc_res_kpc
        k = (this_cop[2] - region_extent[4]) * arc_res_kpc

        shift = (np.array([i - int(i),
                           j - int(j),
                           k - int(k)]) + 0.5) * arc_res

        # Convert the region indices to integers
        ijk[ind] = np.array([int(i), int(j), int(k)])

        this_pos = (reg_dict[key]["coords"]
                    * 10 ** 3 * arcsec_per_kpc_proper + shift)
        this_smls = reg_dict[key]["smls"] * 10 ** 3 * arcsec_per_kpc_proper
        this_subgrpids = reg_dict[key]["part_subgrpids"]

        xcond = np.logical_and(this_pos[:, 0] < imgextent[1],
                               this_pos[:, 0] > imgextent[0])
        ycond = np.logical_and(this_pos[:, 1] < imgextent[1],
                               this_pos[:, 1] > imgextent[0])
        zcond = np.logical_and(this_pos[:, 2] < imgextent[1],
                               this_pos[:, 2] > imgextent[0])
        okinds = np.logical_and(np.logical_and(xcond, ycond), zcond)

        this_flux = reg_dict[key][f][okinds]
        this_pos = this_pos[okinds]
        this_smls = this_smls[okinds]
        this_subgrpids = this_subgrpids[okinds]

        subfind_ids = np.unique(this_subgrpids)

        if np.nansum(this_flux) == 0:
            continue

        if orientation == "sim" or orientation == "face-on":

            img = util.make_spline_img(this_pos, npix, 0, 1, tree,
                                       this_flux, this_smls)

            if Type != "Intrinsic":
                img = signal.fftconvolve(img, psf, mode="same")

            img, img_obj = util.noisy_img(img, image_creator)

        else:

            # # Centre positions on fluxosity weighted centre
            # flux_cent = util.flux_weighted_centre(this_pos,
            #                                         this_flux,
            #                                         i=2, j=0)
            # this_pos[:, (2, 0)] -= flux_cent

            this_radii = util.calc_rad(this_pos, i=2, j=0)

            img = util.make_spline_img(this_pos, npix, 2, 0, tree,
                                       this_flux, this_smls)

            if Type != "Intrinsic":
                img = signal.fftconvolve(img, psf, mode="same")

            img, img_obj = util.noisy_img(img, image_creator)
        print(ind)
        imgs[ind, :, :] = img
        noise[ind] = image_creator.pixel.noise

        begin[ind] = len(fluxes)
        Slen[ind] = len(this_smls)

        star_pos.extend(this_pos)
        smls.extend(this_smls)
        fluxes.extend(this_flux)
        subgrpids.extend(this_subgrpids)

    print("There are", imgs.shape[0], "images")

    fdepth_group = hdf.create_group(str(depth))

    dset = fdepth_group.create_dataset("Images", data=imgs,
                                       dtype=imgs.dtype,
                                       shape=imgs.shape,
                                       compression="gzip")
    dset.attrs["units"] = "$nJy$"

    dset = fdepth_group.create_dataset("IJK", data=ijk,
                                       dtype=ijk.dtype,
                                       shape=ijk.shape,
                                       compression="gzip")
    dset.attrs["units"] = "None"

    dset = fdepth_group.create_dataset("Noise_value", data=noise,
                                       dtype=noise.dtype,
                                       shape=noise.shape,
                                       compression="gzip")
    dset.attrs["units"] = "None"

    fluxes = np.array(fluxes)
    subgrpids = np.array(subgrpids)
    smls = np.array(smls)
    star_pos = np.array(star_pos)

    dset = fdepth_group.create_dataset("Start_Index", data=begin,
                                       dtype=begin.dtype,
                                       shape=begin.shape,
                                       compression="gzip")
    dset.attrs["units"] = "None"

    dset = fdepth_group.create_dataset("Smoothing_Length", data=smls,
                                       dtype=smls.dtype,
                                       shape=smls.shape,
                                       compression="gzip")
    dset.attrs["units"] = "Mpc"

    dset = fdepth_group.create_dataset("Star_Pos", data=star_pos,
                                       dtype=star_pos.dtype,
                                       shape=star_pos.shape,
                                       compression="gzip")
    dset.attrs["units"] = "kpc"

    dset = fdepth_group.create_dataset("Fluxes", data=fluxes,
                                       dtype=fluxes.dtype,
                                       shape=fluxes.shape,
                                       compression="gzip")
    dset.attrs["units"] = "nJy"

    dset = fdepth_group.create_dataset("Part_subgrpids", data=subgrpids,
                                       dtype=subgrpids.dtype,
                                       shape=subgrpids.shape,
                                       compression="gzip")
    dset.attrs["units"] = "None"

    dset = fdepth_group.create_dataset("Image_Length", data=Slen,
                                       dtype=Slen.dtype,
                                       shape=Slen.shape,
                                       compression="gzip")
    dset.attrs["units"] = "None"

# depth = "SUBFIND"
#
# # --- initialise ImageCreator object
# image_creator = imagesim.Idealised(f, field)
#
# print("Creating image for Filter, Pixel noise, Depth:",
#       f, image_creator.pixel.noise, field.depths[f])
#
# fdepth = f + "." + str(depth)
#
# img_num = []
#
# begin = []
# Slen = []
# grp_mass = []
# smls = []
# fluxes = []
# subgrpids = []
# star_pos = []
#
# pre_cut = 0
# post_cut = 0
#
# for key in image_keys:
#
#     ind = key
#
#     this_pos = reg_dict[key]["coords"] * 10 ** 3 * arcsec_per_kpc_proper
#     this_smls = reg_dict[key]["smls"] * 10 ** 3 * arcsec_per_kpc_proper
#     this_subgrpids = reg_dict[key]["part_subgrpids"]
#
#     pre_cut += len(this_smls)
#
#     xcond = np.logical_and(this_pos[:, 0] < imgextent[1],
#                            this_pos[:, 0] > imgextent[0])
#     ycond = np.logical_and(this_pos[:, 1] < imgextent[1],
#                            this_pos[:, 1] > imgextent[0])
#     zcond = np.logical_and(this_pos[:, 2] < imgextent[1],
#                            this_pos[:, 2] > imgextent[0])
#     okinds = np.logical_and(np.logical_and(xcond, ycond), zcond)
#
#     this_flux = reg_dict[key][f][okinds]
#     this_pos = this_pos[okinds]
#     this_smls = this_smls[okinds]
#     this_subgrpids = this_subgrpids[okinds]
#
#     post_cut += len(this_smls)
#
#     img_num.append(key)
#
#     begin.append(len(fluxes))
#     Slen.append(len(this_smls))
#
#     star_pos.extend(this_pos)
#     smls.extend(this_smls)
#     fluxes.extend(this_flux)
#     subgrpids.extend(this_subgrpids)
#
# fdepth_group = hdf.create_group(str(depth))
#
# img_num = np.array(img_num)
# fluxes = np.array(fluxes)
# subgrpids = np.array(subgrpids)
# smls = np.array(smls)
# star_pos = np.array(star_pos)
# begin = np.array(begin)
# Slen = np.array(Slen)
#
# dset = fdepth_group.create_dataset("Image_ID", data=img_num,
#                                    dtype=img_num.dtype,
#                                    shape=img_num.shape,
#                                    compression="gzip")
# dset.attrs["units"] = "None"
#
# dset = fdepth_group.create_dataset("Subgroup_IDs",
#                                    data=gal_haloids,
#                                    dtype=gal_haloids.dtype,
#                                    shape=gal_haloids.shape,
#                                    compression="gzip")
# dset.attrs["units"] = "None"
#
# dset = fdepth_group.create_dataset("Galaxy Mass",
#                                    data=gal_mass,
#                                    dtype=gal_mass.dtype,
#                                    shape=gal_mass.shape,
#                                    compression="gzip")
# dset.attrs["units"] = "$M_\odot$"
#
# dset = fdepth_group.create_dataset("Start_Index", data=begin,
#                                    dtype=begin.dtype,
#                                    shape=begin.shape,
#                                    compression="gzip")
# dset.attrs["units"] = "None"
#
# dset = fdepth_group.create_dataset("Smoothing_Length", data=smls,
#                                    dtype=smls.dtype,
#                                    shape=smls.shape,
#                                    compression="gzip")
# dset.attrs["units"] = "Mpc"
#
# dset = fdepth_group.create_dataset("Star_Pos", data=star_pos,
#                                    dtype=star_pos.dtype,
#                                    shape=star_pos.shape,
#                                    compression="gzip")
# dset.attrs["units"] = "kpc"
#
# dset = fdepth_group.create_dataset("Fluxes", data=fluxes,
#                                    dtype=fluxes.dtype,
#                                    shape=fluxes.shape,
#                                    compression="gzip")
# dset.attrs["units"] = "nJy"
#
# dset = fdepth_group.create_dataset("Part_subgrpids", data=subgrpids,
#                                    dtype=subgrpids.dtype,
#                                    shape=subgrpids.shape,
#                                    compression="gzip")
# dset.attrs["units"] = "None"
#
# dset = fdepth_group.create_dataset("Image_Length", data=Slen,
#                                    dtype=Slen.dtype,
#                                    shape=Slen.shape,
#                                    compression="gzip")
# dset.attrs["units"] = "None"
#
hdf.close()
#
# print(pre_cut, post_cut)

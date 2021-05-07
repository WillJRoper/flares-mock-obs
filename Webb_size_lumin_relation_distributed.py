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
from FLARE.photom import lum_to_M, M_to_lum
from astropy.cosmology import Planck13 as cosmo
import h5py
import photutils as phut
from astropy.convolution import Gaussian2DKernel, convolve_fft
import sys
import time

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

# Define luminosity and dust model types
Type = sys.argv[3]
extinction = 'default'

reg, tag = reg_snaps[ind]
print("Computing HLRs with orientation {o}, type {t}, and extinction {e}"
      "for region {x} and snapshot {u}".format(o=orientation, t=Type,
                                               e=extinction, x=reg, u=tag))

# Define filter
filters = ('JWST.NIRCAM.F150W', )

# Define radii
radii_fracs = (0.2, 0.5, 0.8)

# Define dictionaries for results
hlr_app_dict = {}
hlr_pix_dict = {}
lumin_dict = {}
img_dict = {}
segm_dict = {}

# Set mass limit
masslim = 700

z_str = tag.split('z')[1].split('p')
z = float(z_str[0] + '.' + z_str[1])

hlr_app_dict.setdefault(tag, {})
hlr_pix_dict.setdefault(tag, {})
lumin_dict.setdefault(tag, {})
img_dict.setdefault(tag, {})
segm_dict.setdefault(tag, {})

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
ini_width = 2000 * arcsec_per_kpc_proper

# Define arc_second resolution
if int(filters[0].split(".")[-1][1:4]) < 230:
    arc_res = 0.031
else:
    arc_res = 0.063

# Compute the resolution
ini_res = ini_width / arc_res
res = int(np.ceil(ini_res))

# Compute the new width
width = arc_res * res

print("Image width and resolution (in arcseconds):", width, arc_res)
print("Image width and resolution (in pkpc):",
      width / arcsec_per_kpc_proper,
      arc_res / arcsec_per_kpc_proper)
print("Image width (in pixels):", res)

# Define pixel area in pkpc
single_pixel_area = arc_res * arc_res \
                    / (arcsec_per_kpc_proper * arcsec_per_kpc_proper)

# Define range and extent for the images in arc seconds
imgrange = ((-width / 2, width / 2), (-width / 2, width / 2))
imgextent = [-width / 2, width / 2, -width / 2, width / 2]

# Set up aperture objects
positions = [(res / 2, res / 2)]
app_radii = np.linspace(0.001, res / 4, 100)
apertures = [CircularAperture(positions, r=r) for r in app_radii]
app_radii *= csoft

# Kappa with DTM 0.0795, BC_fac=1., without 0.0063 BC_fac=1.25
reg_dict = phot.flux(reg, kappa=0.0795, tag=tag, BC_fac=1, IMF='Chabrier_300',
                     filters=filters, Type=Type, log10t_BC=7.,
                     extinction=extinction, orientation=orientation,
                     masslim=masslim)

print("Got the dictionary for the region's groups:",
      len(reg_dict), "groups to  test")

for f in filters:

    hlr_app_dict[tag].setdefault(f, {})
    hlr_pix_dict[tag].setdefault(f, {})

    for r in radii_fracs:
        hlr_app_dict[tag][f].setdefault(r, [])
        hlr_pix_dict[tag][f].setdefault(r, [])

    lumin_dict[tag].setdefault(f, [])
    img_dict[tag].setdefault(f, [])
    segm_dict[tag].setdefault(f, [])

    for ind in reg_dict:

        print(ind)

        this_pos = reg_dict[ind]["coords"] * 10 ** 3 * arcsec_per_kpc_proper
        this_smls = reg_dict[ind]["smls"] * 10 ** 3 * arcsec_per_kpc_proper

        this_lumin = reg_dict[ind][f]

        if np.nansum(this_lumin) == 0:
            continue

        if orientation == "sim" or orientation == "face-on":

            this_radii = util.calc_rad(this_pos, i=0, j=1)

            print("Got radii")

            img = util.make_soft_img(this_pos, res, 0, 1, imgrange,
                                     this_lumin,
                                     this_smls)

            print("Got image")

        else:

            # # Centre positions on luminosity weighted centre
            # lumin_cent = util.lumin_weighted_centre(this_pos,
            #                                         this_lumin,
            #                                         i=2, j=0)
            # this_pos[:, (2, 0)] -= lumin_cent

            this_radii = util.calc_rad(this_pos, i=2, j=0)

            img = util.make_soft_img(this_pos, res, 2, 0, imgrange,
                                     this_lumin,
                                     this_smls)

        # img[img < 10**21] = 0

        threshold = phut.detect_threshold(img, nsigma=5)
        print("Threshold:", np.median(img))
        # threshold = np.median(img)

        segm = phut.detect_sources(img, threshold, npixels=10,
                                   filter_kernel=kernel)
        segm = phut.deblend_sources(img, segm, npixels=10,
                                    filter_kernel=kernel,
                                    nlevels=32, contrast=0.001)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        ax1.grid(False)
        ax2.grid(False)
        ax1.imshow(np.log10(img), extent=imgextent, cmap="Greys_r")
        ax2.imshow(segm.data, extent=imgextent)
        fig.savefig("plots/gal_img_log_" + f + "_%.1f.png"
                    % np.log10(np.sum(img)), dpi=300)
        plt.close(fig)

        for i in range(1, np.max(segm.data) + 1):
            if np.sum(img[segm.data == i]) < np.median(img):
                continue
            print(np.sum(img[segm.data == i]))
            for r in radii_fracs:
                img_segm = np.zeros_like(img)
                img_segm[segm.data == i] = img[segm.data == i]
                hlr_pix_dict[tag][f][r].append(
                    util.get_pixel_hlr(img_segm, single_pixel_area,
                                       radii_frac=r))
                # hlr_app_dict[tag][f][r].append(
                #     util.get_img_hlr(img, apertures, app_radii, res,
                #                      arc_res / arcsec_per_kpc_proper, r))
            lumin_dict[tag][f].append(np.sum(img[segm.data == i]))

        img_dict[tag][f].append(img)
        segm_dict[tag][f].append(segm.data)


try:
    hdf = h5py.File("mock_data/"
                    "flares_sizes_{}_{}_Webb.hdf5".format(reg, tag), "r+")
except OSError:
    hdf = h5py.File("mock_data/"
                    "flares_sizes_{}_{}_Webb.hdf5".format(reg, tag), "w")

try:
    type_group = hdf[Type]
except KeyError:
    print(Type, "Doesn't exists: Creating...")
    type_group = hdf.create_group(Type)

try:
    orientation_group = type_group[orientation]
except KeyError:
    print(orientation, "Doesn't exists: Creating...")
    orientation_group = type_group.create_group(orientation)

for f in filters:

    fluxes = np.array(lumin_dict[tag][f])
    imgs = np.array(img_dict[tag][f])
    segms = np.array(segm_dict[tag][f])

    print(imgs.shape)

    try:
        f_group = orientation_group[f]
    except KeyError:
        print(f, "Doesn't exists: Creating...")
        f_group = orientation_group.create_group(f)

    try:
        dset = f_group.create_dataset("Flux", data=fluxes,
                                      dtype=fluxes.dtype,
                                      shape=fluxes.shape,
                                      compression="gzip")
        dset.attrs["units"] = "$nJy$"
    except RuntimeError:
        print("Flux already exists: Overwriting...")
        del f_group["Flux"]
        dset = f_group.create_dataset("Flux", data=fluxes,
                                      dtype=fluxes.dtype,
                                      shape=fluxes.shape,
                                      compression="gzip")
        dset.attrs["units"] = "$nJy$"

    try:
        dset = f_group.create_dataset("Images", data=imgs,
                                      dtype=imgs.dtype,
                                      shape=imgs.shape,
                                      compression="gzip")
        dset.attrs["units"] = "$nJy$"
    except RuntimeError:
        print("Images already exists: Overwriting...")
        del f_group["Images"]
        dset = f_group.create_dataset("Images", data=imgs,
                                      dtype=imgs.dtype,
                                      shape=imgs.shape,
                                      compression="gzip")
        dset.attrs["units"] = "$nJy$"

    try:
        dset = f_group.create_dataset("Segmentation_Maps", data=segms,
                                      dtype=segms.dtype,
                                      shape=segms.shape,
                                      compression="gzip")
        dset.attrs["units"] = "None"
    except RuntimeError:
        print("Images already exists: Overwriting...")
        del f_group["Images"]
        dset = f_group.create_dataset("Segmentation_Maps", data=segms,
                                      dtype=segms.dtype,
                                      shape=segms.shape,
                                      compression="gzip")
        dset.attrs["units"] = "None"

    for r in radii_fracs:

        hlrs_app = np.array(hlr_app_dict[tag][f][r])
        hlrs_pix = np.array(hlr_pix_dict[tag][f][r])

        try:
            dset = f_group.create_dataset("HLR_Aperture_%.1f" % r,
                                          data=hlrs_app,
                                          dtype=hlrs_app.dtype,
                                          shape=hlrs_app.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$\mathrm{pkpc}$"
        except RuntimeError:
            print("HLR_Aperture_%.1f" % r, "already exists: Overwriting...")
            del f_group["HLR_Aperture_%.1f" % r]
            dset = f_group.create_dataset("HLR_Aperture_%.1f" % r,
                                          data=hlrs_app,
                                          dtype=hlrs_app.dtype,
                                          shape=hlrs_app.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$\mathrm{pkpc}$"

        try:
            dset = f_group.create_dataset("HLR_Pixel_%.1f" % r,
                                          data=hlrs_pix,
                                          dtype=hlrs_pix.dtype,
                                          shape=hlrs_pix.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$\mathrm{pkpc}$"
        except RuntimeError:
            print("HLR_Pixel_%.1f" % r, "already exists: Overwriting...")
            del f_group["HLR_Pixel_%.1f" % r]
            dset = f_group.create_dataset("HLR_Pixel_%.1f" % r,
                                          data=hlrs_pix,
                                          dtype=hlrs_pix.dtype,
                                          shape=hlrs_pix.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$\mathrm{pkpc}$"

hdf.close()

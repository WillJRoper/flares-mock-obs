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

filter_ind = int(sys.argv[4])

reg, tag = reg_snaps[ind]
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

hdf = h5py.File("mock_data/flares_segm_{}_{}_{}_{}.hdf5"
                .format(reg, tag, Type, orientation), "w")
print("Creating File...")

f = filters[filter_ind]

f_group = hdf.create_group(f)

# --- initialise ImageCreator object
image_creator = imagesim.Idealised(f, field)

arc_res = image_creator.pixel_scale

# Compute the resolution
ini_res = ini_width / arc_res
res = int(np.ceil(ini_res))

# Compute the new width
width = arc_res * res

print("Filter:", f)
print("Image width and resolution (in arcseconds):", width, arc_res)
print("Image width and resolution (in pkpc):",
      width / arcsec_per_kpc_proper,
      arc_res / arcsec_per_kpc_proper)
print("Image width (in pixels):", res)

# Kappa with DTM 0.0795, BC_fac=1., without 0.0063 BC_fac=1.25
reg_dict = phot.flux(reg, kappa=0.0795, tag=tag, BC_fac=1,
                     IMF='Chabrier_300',
                     filters=(f, ), Type=Type, log10t_BC=7.,
                     extinction=extinction, orientation=orientation,
                     masslim=masslim,
                     r=width / arcsec_per_kpc_proper / 1000 / 2)

print("Got the dictionary for the region's groups:",
      len(reg_dict) - 4, "groups to  test")

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

# Define x and y positions of pixels
X, Y = np.meshgrid(np.linspace(imgrange[0][0], imgrange[0][1], res),
                   np.linspace(imgrange[1][0], imgrange[1][1], res))

# Define pixel position array for the KDTree
pix_pos = np.zeros((X.size, 2))
pix_pos[:, 0] = X.ravel()
pix_pos[:, 1] = Y.ravel()

# Build KDTree
tree = cKDTree(pix_pos)

print("Pixel tree built")

psf = imagesim.create_PSF(f, field, res)

image_keys = [k for k in reg_dict.keys() if type(k) is int]

gal_mass = reg_dict["gal_ms"]
gal_haloids = reg_dict["gal_haloids"]

if len(image_keys) > 0:
    nimg = np.max(image_keys) + 1

    begin = np.zeros(nimg, dtype=int)
    Slen = np.zeros(nimg, dtype=int)
    grp_mass  = np.zeros(nimg)

else:
    nimg = 0
    begin = np.zeros(nimg, dtype=int)
    Slen  = np.zeros(nimg, dtype=int)
    grp_mass  = np.zeros(nimg)

smls = []
fluxes = []
subgrpids  = []
star_pos = []

for num, depth in enumerate(depths):

    field.depths[f] = depth

    # --- initialise ImageCreator object
    image_creator = imagesim.Idealised(f, field)

    print("Creating image for Filter, Pixel noise, Depth:",
          f, image_creator.pixel.noise, field.depths[f])

    fdepth = f + "." + str(depth)

    imgs = np.full((nimg, res, res), np.nan)
    # segms = np.zeros((nimg, res, res))
    sigs = np.zeros((nimg, res, res))

    failed = 0
    segm_sources = 0

    for key in image_keys:

        ind = key

        this_pos = reg_dict[key]["coords"] * 10 ** 3 * arcsec_per_kpc_proper
        this_smls = reg_dict[key]["smls"] * 10 ** 3 * arcsec_per_kpc_proper
        this_subgrpids = reg_dict[key]["part_subgrpids"]
        this_groupmass = reg_dict[key]["group_mass"]

        this_flux = reg_dict[key][f]

        subfind_ids = np.unique(this_subgrpids)

        if np.nansum(this_flux) == 0:

            if num == 0:
                begin[ind] = -1
                Slen[ind] = 0

            failed += 1

            print(failed, ind)

            continue

        if orientation == "sim" or orientation == "face-on":

            this_radii = util.calc_rad(this_pos, i=0, j=1)

            img = util.make_spline_img(this_pos, res, 0, 1, tree,
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

            img = util.make_spline_img(this_pos, res, 2, 0, tree,
                                       this_flux, this_smls)

            if Type != "Intrinsic":
                img = signal.fftconvolve(img, psf, mode="same")

            img, img_obj = util.noisy_img(img, image_creator)

        significance_image = img / img_obj.noise
        significance_image[significance_image < 0] = 0

        try:
            # segm = phut.detect_sources(significance_image, 2.5, npixels=5)
            # segm = phut.deblend_sources(img, segm, npixels=5,
            #                             nlevels=32, contrast=0.001)

            imgs[ind, :, :] = img
            # segms[ind, :, :] = segm.data
            sigs[ind, :, :] = significance_image

            # util.plot_images(img, segm.data, significance_image, reg,
            #                  f, depth, tag, ind, imgextent, ini_width_pkpc,
            #                  cutout_halfsize=int(0.1 * res))

            # segm_sources += np.unique(segm.data).size - 1

        except TypeError as e:
            print(e)
            print(img.min(), img.max(),
                  significance_image.min(), significance_image.max())

            failed += 1

            imgs[ind, :, :] = img
            # segms[ind, :, :] = np.zeros((res, res))
            sigs[ind, :, :] = np.zeros((res, res))

        if num == 0:

            begin[ind] = len(fluxes)
            Slen[ind] = len(this_smls)
            grp_mass[ind] = this_groupmass

            star_pos.extend(this_pos)
            smls.extend(this_smls)
            fluxes.extend(this_flux)
            subgrpids.extend(this_subgrpids)

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

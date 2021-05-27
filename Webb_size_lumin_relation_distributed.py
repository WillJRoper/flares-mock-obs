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

reg, tag = reg_snaps[ind]
print("Computing HLRs with orientation {o}, type {t}, and extinction {e}"
      " for region {x} and snapshot {u}".format(o=orientation, t=Type,
                                               e=extinction, x=reg, u=tag))

# Define filter
filters = ('Hubble.WFC3.f160w', )

depths = [1, 5, 10, 20, 50]

# Define radii
radii_fracs = (0.2, 0.5, 0.8)

# Define dictionaries and lists for results
hlr_app_dict = {}
hlr_pix_dict = {}
flux_dict = {}
img_dict = {}
segm_dict = {}
sig_dict = {}
ngal_dict = {}
sf_ngal_dict = {}
grp_dict = {}
star_pos = {}
begin = {}
Slen = {}
smls = {}
fluxes = {}
subgrpids = {}
grp_mass = {}
gal_mass = {}
gal_haloids = {}

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
ini_width_pkpc = 100
ini_width = ini_width_pkpc * arcsec_per_kpc_proper

for f in filters:

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

    nimg = max(image_keys) + 1

    gal_mass[f] = reg_dict["gal_ms"]
    gal_haloids[f] = reg_dict["gal_haloids"]

    begin.setdefault(f, np.zeros(nimg, dtype=int))
    Slen.setdefault(f, np.zeros(nimg, dtype=int))
    grp_mass.setdefault(f, np.zeros(nimg))

    smls.setdefault(f, [])
    fluxes.setdefault(f, [])
    subgrpids.setdefault(f, [])
    star_pos.setdefault(f, [])

    for num, depth in enumerate(depths):

        field.depths[f] = depth

        print("Creating image for Filter, Noise, Depth:",
              f, image_creator.pixel.noise, field.depths[f])

        # --- initialise ImageCreator object
        image_creator = imagesim.Idealised(f, field)

        fdepth = f + "." + str(depth)

        img_dict.setdefault(fdepth, np.full((nimg, res, res), np.nan))
        segm_dict.setdefault(fdepth, np.zeros((nimg, res, res)))
        sig_dict.setdefault(fdepth, np.zeros((nimg, res, res)))

        for key in image_keys:

            ind = key

            this_pos = reg_dict[key]["coords"] * 10 ** 3 * arcsec_per_kpc_proper
            this_smls = reg_dict[key]["smls"] * 10 ** 3 * arcsec_per_kpc_proper
            this_subgrpids = reg_dict[key]["part_subgrpids"]
            this_groupmass = reg_dict[key]["group_mass"]

            this_flux = reg_dict[key][f]

            subfind_ids = np.unique(this_subgrpids)

            if np.nansum(this_flux) == 0:

                img_dict[fdepth][ind, :, :] = np.full((res, res), np.nan)
                segm_dict[fdepth][ind, :, :] = np.zeros((res, res))
                sig_dict[fdepth][ind, :, :] = np.zeros((res, res))

                if num == 0:
                    begin[f][ind] = -1
                    Slen[f][ind] = 0

                continue

            if orientation == "sim" or orientation == "face-on":

                this_radii = util.calc_rad(this_pos, i=0, j=1)

                img = util.make_spline_img(this_pos, res, 0, 1, tree,
                                           this_flux, this_smls)

                img_psf = signal.fftconvolve(img, psf, mode="same")

                img, noise = util.noisy_img(img_psf, image_creator)

            else:

                # # Centre positions on fluxosity weighted centre
                # flux_cent = util.flux_weighted_centre(this_pos,
                #                                         this_flux,
                #                                         i=2, j=0)
                # this_pos[:, (2, 0)] -= flux_cent

                this_radii = util.calc_rad(this_pos, i=2, j=0)

                img = util.make_spline_img(this_pos, res, 2, 0, tree,
                                           this_flux, this_smls)

                img_psf = signal.fftconvolve(img, psf, mode="same")

                img, noise = util.noisy_img(img_psf, image_creator)

            significance_image = img_psf / noise

            try:
                segm = phut.detect_sources(significance_image, 2.0, npixels=5)
                segm = phut.deblend_sources(img, segm, npixels=5,
                                            nlevels=32, contrast=0.001)

                img_dict[fdepth][ind, :, :] = img
                segm_dict[fdepth][ind, :, :] = segm.data
                sig_dict[fdepth][ind, :, :] = significance_image

            except TypeError:

                print(ind, "had no sources above noise with a depth of",
                      depth, "nJy")

                img_dict[fdepth][ind, :, :] = img
                segm_dict[fdepth][ind, :, :] = np.zeros((res, res))
                sig_dict[fdepth][ind, :, :] = np.zeros((res, res))

            if num == 0:

                begin[f][ind] = len(fluxes[f])
                Slen[f][ind] = len(this_smls)
                grp_mass[f][ind] = this_groupmass

                star_pos[f].extend(this_pos)
                smls[f].extend(this_smls)
                fluxes[f].extend(this_flux)
                subgrpids[f].extend(this_subgrpids)


try:
    hdf = h5py.File("mock_data/"
                    "flares_segm_{}_{}_Webb.hdf5".format(reg, tag), "r+")
except OSError:
    hdf = h5py.File("mock_data/"
                    "flares_segm_{}_{}_Webb.hdf5".format(reg, tag), "w")
    print("Creating File...")

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

    try:
        f_group = orientation_group[f]
    except KeyError:
        print(f, "Doesn't exists: Creating...")
        f_group = orientation_group.create_group(f)

    begin = begin[f]
    grp_mass = grp_mass[f]
    Slen = Slen[f]
    fluxes = np.array(fluxes[f])
    subgrpids = np.array(subgrpids[f])
    smls = np.array(smls[f])
    star_pos = np.array(star_pos[f])

    try:
        dset = f_group.create_dataset("Subgroup_IDs",
                                                data=gal_haloids[f],
                                                dtype=gal_haloids[f].dtype,
                                                shape=gal_haloids[f].shape,
                                                compression="gzip")
        dset.attrs["units"] = "None"
    except OSError:
        print("Subgroup_IDs already exists: Overwriting...")
        del f_group["Subgroup_IDs"]
        dset = f_group.create_dataset("Subgroup_IDs",
                                                data=gal_haloids[f],
                                                dtype=gal_haloids[f].dtype,
                                                shape=gal_haloids[f].shape,
                                                compression="gzip")
        dset.attrs["units"] = "None"

    try:
        dset = f_group.create_dataset("Galaxy Mass",
                                                data=gal_mass[f],
                                                dtype=gal_mass[f].dtype,
                                                shape=gal_mass[f].shape,
                                                compression="gzip")
        dset.attrs["units"] = "$M_\odot$"
    except OSError:
        print("Galaxy Mass already exists: Overwriting...")
        del f_group["Galaxy Mass"]
        dset = f_group.create_dataset("Galaxy Mass",
                                                data=gal_mass[f],
                                                dtype=gal_mass[f].dtype,
                                                shape=gal_mass[f].shape,
                                                compression="gzip")
        dset.attrs["units"] = "$M_\odot$"

    try:
        dset = f_group.create_dataset("Start_Index", data=begin,
                                        dtype=begin.dtype,
                                        shape=begin.shape,
                                        compression="gzip")
        dset.attrs["units"] = "None"
    except OSError:
        print("Start_Index already exists: Overwriting...")
        del f_group["Start_Index"]
        dset = f_group.create_dataset("Start_Index", data=begin,
                                        dtype=begin.dtype,
                                        shape=begin.shape,
                                        compression="gzip")
        dset.attrs["units"] = "None"

    try:
        dset = f_group.create_dataset("Group_Mass", data=grp_mass,
                                        dtype=grp_mass.dtype,
                                        shape=grp_mass.shape,
                                        compression="gzip")
        dset.attrs["units"] = "$M_\odot$"
    except OSError:
        print("Group_Mass already exists: Overwriting...")
        del f_group["Group_Mass"]
        dset = f_group.create_dataset("Group_Mass", data=grp_mass,
                                        dtype=grp_mass.dtype,
                                        shape=grp_mass.shape,
                                        compression="gzip")
        dset.attrs["units"] = "$M_\odot$"

    try:
        dset = f_group.create_dataset("Smoothing_Length", data=smls,
                                        dtype=smls.dtype,
                                        shape=smls.shape,
                                        compression="gzip")
        dset.attrs["units"] = "Mpc"
    except OSError:
        print("Smoothing_Length already exists: Overwriting...")
        del f_group["Smoothing_Length"]
        dset = f_group.create_dataset("Smoothing_Length", data=smls,
                                        dtype=smls.dtype,
                                        shape=smls.shape,
                                        compression="gzip")
        dset.attrs["units"] = "Mpc"

    try:
        dset = f_group.create_dataset("Star_Pos", data=star_pos,
                                        dtype=star_pos.dtype,
                                        shape=star_pos.shape,
                                        compression="gzip")
        dset.attrs["units"] = "kpc"
    except OSError:
        print("Star_Pos already exists: Overwriting...")
        del f_group["Star_Pos"]
        dset = f_group.create_dataset("Star_Pos", data=star_pos,
                                        dtype=star_pos.dtype,
                                        shape=star_pos.shape,
                                        compression="gzip")
        dset.attrs["units"] = "kpc"

    try:
        dset = f_group.create_dataset("Fluxes", data=fluxes,
                                        dtype=fluxes.dtype,
                                        shape=fluxes.shape,
                                        compression="gzip")
        dset.attrs["units"] = "nJy"
    except OSError:
        print("Fluxes already exists: Overwriting...")
        del f_group["Fluxes"]
        dset = f_group.create_dataset("Fluxes", data=fluxes,
                                        dtype=fluxes.dtype,
                                        shape=fluxes.shape,
                                        compression="gzip")
        dset.attrs["units"] = "nJy"

    try:
        dset = f_group.create_dataset("Part_subgrpids", data=subgrpids,
                                        dtype=subgrpids.dtype,
                                        shape=subgrpids.shape,
                                        compression="gzip")
        dset.attrs["units"] = "None"
    except OSError:
        print("Part_subgrpids already exists: Overwriting...")
        del f_group["Part_subgrpids"]
        dset = f_group.create_dataset("Part_subgrpids", data=subgrpids,
                                        dtype=subgrpids.dtype,
                                        shape=subgrpids.shape,
                                        compression="gzip")
        dset.attrs["units"] = "None"

    try:
        dset = f_group.create_dataset("Group_Length", data=Slen,
                                        dtype=Slen.dtype,
                                        shape=Slen.shape,
                                        compression="gzip")
        dset.attrs["units"] = "None"
    except OSError:
        print("Group_Length already exists: Overwriting...")
        del f_group["Group_Length"]
        dset = f_group.create_dataset("Group_Length", data=Slen,
                                        dtype=Slen.dtype,
                                        shape=Slen.shape,
                                        compression="gzip")
        dset.attrs["units"] = "None"

    for num, depth in enumerate(depths):

        fdepth = f + "." + str(depth)
        imgs = np.array(img_dict[fdepth])
        segms = np.array(segm_dict[fdepth])
        sigs = np.array(sig_dict[fdepth])

        print(imgs.shape)

        try:
            fdepth_group = f_group[str(depth)]
        except KeyError:
            print(str(depth), "Doesn't exists: Creating...")
            fdepth_group = f_group.create_group(str(depth))

        try:
            dset = fdepth_group.create_dataset("Images", data=imgs,
                                          dtype=imgs.dtype,
                                          shape=imgs.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$nJy$"
        except OSError:
            print("Images already exists: Overwriting...")
            del fdepth_group["Images"]
            dset = fdepth_group.create_dataset("Images", data=imgs,
                                          dtype=imgs.dtype,
                                          shape=imgs.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$nJy$"

        try:
            dset = fdepth_group.create_dataset("Segmentation_Maps", data=segms,
                                          dtype=segms.dtype,
                                          shape=segms.shape,
                                          compression="gzip")
            dset.attrs["units"] = "None"
        except OSError:
            print("Segmentation_Maps already exists: Overwriting...")
            del fdepth_group["Segmentation_Maps"]
            dset = fdepth_group.create_dataset("Segmentation_Maps", data=segms,
                                          dtype=segms.dtype,
                                          shape=segms.shape,
                                          compression="gzip")
            dset.attrs["units"] = "None"


        try:
            dset = fdepth_group.create_dataset("Significance_Images", data=sigs,
                                          dtype=sigs.dtype,
                                          shape=sigs.shape,
                                          compression="gzip")
            dset.attrs["units"] = "None"
        except OSError:
            print("Significance_Images already exists: Overwriting...")
            del fdepth_group["Significance_Images"]
            dset = fdepth_group.create_dataset("Significance_Images", data=sigs,
                                          dtype=sigs.dtype,
                                          shape=sigs.shape,
                                          compression="gzip")
            dset.attrs["units"] = "None"

hdf.close()

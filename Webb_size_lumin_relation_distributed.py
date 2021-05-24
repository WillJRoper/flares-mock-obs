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
      "for region {x} and snapshot {u}".format(o=orientation, t=Type,
                                               e=extinction, x=reg, u=tag))

# Define filter
filters = ('Hubble.WFC3.f160w.10','Hubble.WFC3.f160w.5', 'Hubble.WFC3.f160w.1',
           'Hubble.WFC3.f160w.20', 'Hubble.WFC3.f160w.50')

# Define radii
radii_fracs = (0.2, 0.5, 0.8)

# Define dictionaries and lists for results
hlr_app_dict = {}
hlr_pix_dict = {}
flux_dict = {}
img_dict = {}
segm_dict = {}
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

# Set mass limit
masslim = 700

z_str = tag.split('z')[1].split('p')
z = float(z_str[0] + '.' + z_str[1])

hlr_app_dict.setdefault(tag, {})
hlr_pix_dict.setdefault(tag, {})
flux_dict.setdefault(tag, {})
img_dict.setdefault(tag, {})
segm_dict.setdefault(tag, {})
ngal_dict.setdefault(tag, {})
sf_ngal_dict.setdefault(tag, {})
grp_dict.setdefault(tag, {})
begin.setdefault(tag, {})
Slen.setdefault(tag, {})
smls.setdefault(tag, {})
fluxes.setdefault(tag, {})
subgrpids.setdefault(tag, {})
grp_mass.setdefault(tag, {})

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
ini_width = 500 * arcsec_per_kpc_proper

for fdepth in filters:

    f_split = fdepth.split(".")
    f = f_split[0] + "." + f_split[1] + "." + f_split[2]
    depth = float(f_split[-1])

    field.depths[f] = depth

    # --- initialise ImageCreator object
    image_creator = imagesim.Idealised(f, field)

    print("Noise, Depth:", image_creator.pixel.noise, field.depths[f])

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
                         filters=filters, Type=Type, log10t_BC=7.,
                         extinction=extinction, orientation=orientation,
                         masslim=masslim,
                         r=width / arcsec_per_kpc_proper / 1000 / 2)

    print("Got the dictionary for the region's groups:",
          len(reg_dict) - 3, "groups to  test")

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

    hlr_app_dict[tag].setdefault(f, {})
    hlr_pix_dict[tag].setdefault(f, {})

    for r in radii_fracs:
        hlr_app_dict[tag][f].setdefault(r, [])
        hlr_pix_dict[tag][f].setdefault(r, [])

    flux_dict[tag].setdefault(fdepth, [])
    img_dict[tag].setdefault(fdepth, [])
    segm_dict[tag].setdefault(fdepth, [])
    ngal_dict[tag].setdefault(fdepth, [])
    grp_dict[tag].setdefault(fdepth, [])
    sf_ngal_dict[tag].setdefault(fdepth, [])
    begin[tag].setdefault(fdepth, [])
    Slen[tag].setdefault(fdepth, [])
    smls[tag].setdefault(fdepth, [])
    fluxes[tag].setdefault(fdepth, [])
    subgrpids[tag].setdefault(fdepth, [])
    grp_mass[tag].setdefault(fdepth, [])

    for ind in reg_dict:

        if not type(ind) is int:
            continue

        this_pos = reg_dict[ind]["coords"] * 10 ** 3 * arcsec_per_kpc_proper
        this_smls = reg_dict[ind]["smls"] * 10 ** 3 * arcsec_per_kpc_proper
        this_subgrpids = reg_dict[ind]["part_subgrpids"]
        this_groupmass = reg_dict[ind]["group_mass"]

        this_flux = reg_dict[ind][f]

        subfind_ids = np.unique(this_subgrpids)

        if np.nansum(this_flux) == 0:
            continue

        if orientation == "sim" or orientation == "face-on":

            this_radii = util.calc_rad(this_pos, i=0, j=1)

            img = util.make_spline_img(this_pos, res, 0, 1, tree,
                                       this_flux, this_smls)

            img = signal.fftconvolve(img, psf, mode="same")

            img, noise = util.noisy_img(img, image_creator)

        else:

            # # Centre positions on fluxosity weighted centre
            # flux_cent = util.flux_weighted_centre(this_pos,
            #                                         this_flux,
            #                                         i=2, j=0)
            # this_pos[:, (2, 0)] -= flux_cent

            this_radii = util.calc_rad(this_pos, i=2, j=0)

            img = util.make_soft_img(this_pos, res, 2, 0, imgrange,
                                     this_flux,
                                     this_smls)

        threshold = phut.detect_threshold(img, nsigma=5)

        try:
            segm = phut.detect_sources(img, threshold, npixels=5)
            segm = phut.deblend_sources(img, segm, npixels=5,
                                        nlevels=32, contrast=0.001)
        except TypeError:
            print(ind, "had no sources above noise")
            continue

        subfind_img = util.make_subfind_spline_img(this_pos, res, 0, 1, tree,
                                                   this_subgrpids,
                                                   this_smls,
                                                   spline_cut_off=5 / 2)

        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        # ax1.grid(False)
        # ax2.grid(False)
        # plt_img = np.zeros_like(img)
        # plt_img[img > 0] = np.log10(img[img > 0])
        # ax1.imshow(plt_img, extent=imgextent, cmap="Greys_r")
        # cmap = segm.make_cmap()
        # ax2.imshow(segm.data, extent=imgextent, cmap=cmap)
        # fig.savefig("plots/gal_img_log_" + f + "_%d.png"
        #             % int(ind), dpi=300)
        # plt.close(fig)

        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        # ax1.grid(False)
        # ax2.grid(False)
        # plt_img = np.zeros_like(img)
        # plt_img[img > 0] = np.log10(img[img > 0])
        # max_ind = np.unravel_index(np.argmax(plt_img), plt_img.shape)
        # ax1.imshow(plt_img[max_ind[0] - 100: max_ind[0] + 100,
        #            max_ind[1] - 100: max_ind[1] + 100],
        #            extent=imgextent, cmap="Greys_r")
        # cmap = segm.make_cmap()
        # ax2.imshow(segm.data[max_ind[0] - 100: max_ind[0] + 100,
        #            max_ind[1] - 100: max_ind[1] + 100], extent=imgextent,
        #            cmap=cmap)
        # fig.savefig("plots/max_gal_img_log_" + f + "_%.1f.png"
        #             % np.log10(np.sum(img)), dpi=300)
        # plt.close(fig)

        # for i in range(1, np.max(segm.data) + 1):
        #     if np.sum(img[segm.data == i]) < np.median(img):
        #         continue
        #     print(np.sum(img[segm.data == i]))
        #     for r in radii_fracs:
        #         img_segm = np.zeros_like(img)
        #         img_segm[segm.data == i] = img[segm.data == i]
        #         hlr_pix_dict[tag][f][r].append(
        #             util.get_pixel_hlr(img_segm, single_pixel_area,
        #                                radii_frac=r))
        #         # hlr_app_dict[tag][f][r].append(
        #         #     util.get_img_hlr(img, apertures, app_radii, res,
        #         #                      arc_res / arcsec_per_kpc_proper, r))
        #     flux_dict[tag][f].append(np.sum(img[segm.data == i]))
        #
        img_dict[tag][fdepth].append(img)
        segm_dict[tag][fdepth].append(segm.data)
        grp_dict[tag][fdepth].append(ind)

        begin[tag][fdepth].append(len(fluxes))
        grp_mass[tag][fdepth].append(this_groupmass)
        star_pos[tag][fdepth].extend(this_pos)
        smls[tag][fdepth].extend(this_smls)
        fluxes[tag][fdepth].extend(this_flux)
        subgrpids[tag][fdepth].extend(this_subgrpids)
        Slen[tag][fdepth].append(len(this_smls))

        ngal = 0
        for gal in np.unique(segm.data):
            if np.sum(img[segm.data == gal]) > image_creator.aperture.background and gal > 0:
                ngal += 1

        ngal_dict[tag][fdepth].append(ngal)

        sub_ngal = np.unique(subfind_img[segm.data != 0]).size

        sf_ngal_dict[tag][fdepth].append(sub_ngal)


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

    begin = np.array(begin[tag][f])
    grp_mass = np.array(grp_mass[tag][f])
    fluxes = np.array(fluxes[tag][f])
    subgrpids = np.array(subgrpids[tag][f])
    Slen = np.array(Slen[tag][f])
    smls = np.array(smls[tag][f])
    star_pos = np.array(star_pos[tag][f])

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

    # fluxes = np.array(flux_dict[tag][f][snr])
    imgs = np.array(img_dict[tag][f])
    segms = np.array(segm_dict[tag][f])
    ngals = np.array(ngal_dict[tag][f])
    sf_ngals = np.array(sf_ngal_dict[tag][f])
    grps = np.array(grp_dict[tag][f])

    print(imgs.shape)

    try:
        dset = f_group.create_dataset("Group_ID", data=grps,
                                      dtype=grps.dtype,
                                      shape=grps.shape,
                                      compression="gzip")
        dset.attrs["units"] = "None"
    except OSError:
        print("Group_ID already exists: Overwriting...")
        del f_group["Group_ID"]
        dset = f_group.create_dataset("Group_ID", data=grps,
                                      dtype=grps.dtype,
                                      shape=grps.shape,
                                      compression="gzip")
        dset.attrs["units"] = "None"

    try:
        dset = f_group.create_dataset("Images", data=imgs,
                                      dtype=imgs.dtype,
                                      shape=imgs.shape,
                                      compression="gzip")
        dset.attrs["units"] = "$nJy$"
    except OSError:
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
    except OSError:
        print("Segmentation_Maps already exists: Overwriting...")
        del f_group["Segmentation_Maps"]
        dset = f_group.create_dataset("Segmentation_Maps", data=segms,
                                      dtype=segms.dtype,
                                      shape=segms.shape,
                                      compression="gzip")
        dset.attrs["units"] = "None"

    try:
        dset = f_group.create_dataset("NGalaxy", data=ngals,
                                      dtype=ngals.dtype,
                                      shape=ngals.shape,
                                      compression="gzip")
        dset.attrs["units"] = "None"
    except OSError:
        print("NGalaxy already exists: Overwriting...")
        del f_group["NGalaxy"]
        dset = f_group.create_dataset("NGalaxy", data=ngals,
                                      dtype=ngals.dtype,
                                      shape=ngals.shape,
                                      compression="gzip")
        dset.attrs["units"] = "None"

    try:
        dset = f_group.create_dataset("SUBFIND_NGalaxy", data=sf_ngals,
                                      dtype=sf_ngals.dtype,
                                      shape=sf_ngals.shape,
                                      compression="gzip")
        dset.attrs["units"] = "None"
    except OSError:
        print("SUBFIND_NGalaxy already exists: Overwriting...")
        del f_group["SUBFIND_NGalaxy"]
        dset = f_group.create_dataset("SUBFIND_NGalaxy", data=sf_ngals,
                                      dtype=sf_ngals.dtype,
                                      shape=sf_ngals.shape,
                                      compression="gzip")
        dset.attrs["units"] = "None"

        # for r in radii_fracs:
        #
        #     hlrs_app = np.array(hlr_app_dict[tag][f][r])
        #     hlrs_pix = np.array(hlr_pix_dict[tag][f][r])
        #
        #     try:
        #         dset = f_group.create_dataset("HLR_Aperture_%.1f" % r,
        #                                       data=hlrs_app,
        #                                       dtype=hlrs_app.dtype,
        #                                       shape=hlrs_app.shape,
        #                                       compression="gzip")
        #         dset.attrs["units"] = "$\mathrm{pkpc}$"
        #     except OSError:
        #         print("HLR_Aperture_%.1f" % r, "already exists: Overwriting...")
        #         del f_group["HLR_Aperture_%.1f" % r]
        #         dset = f_group.create_dataset("HLR_Aperture_%.1f" % r,
        #                                       data=hlrs_app,
        #                                       dtype=hlrs_app.dtype,
        #                                       shape=hlrs_app.shape,
        #                                       compression="gzip")
        #         dset.attrs["units"] = "$\mathrm{pkpc}$"
        #
        #     try:
        #         dset = f_group.create_dataset("HLR_Pixel_%.1f" % r,
        #                                       data=hlrs_pix,
        #                                       dtype=hlrs_pix.dtype,
        #                                       shape=hlrs_pix.shape,
        #                                       compression="gzip")
        #         dset.attrs["units"] = "$\mathrm{pkpc}$"
        #     except OSError:
        #         print("HLR_Pixel_%.1f" % r, "already exists: Overwriting...")
        #         del f_group["HLR_Pixel_%.1f" % r]
        #         dset = f_group.create_dataset("HLR_Pixel_%.1f" % r,
        #                                       data=hlrs_pix,
        #                                       dtype=hlrs_pix.dtype,
        #                                       shape=hlrs_pix.shape,
        #                                       compression="gzip")
        #         dset.attrs["units"] = "$\mathrm{pkpc}$"
        #
hdf.close()

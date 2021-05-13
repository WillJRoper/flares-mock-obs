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
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
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

# Define fluxosity and dust model types
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

# Define dictionaries and lists for results
hlr_app_dict = {}
hlr_pix_dict = {}
flux_dict = {}
img_dict = {}
segm_dict = {}
ngal_dict = {}
sf_ngal_dict = {}
grp_dict = {}
star_pos = []
begin = []
Slen = []
smls = []
fluxes = []
subgrpids = []
grp_mass = []

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
                     masslim=masslim,
                     r=width / arcsec_per_kpc_proper / 1000 / 2)

print("Got the dictionary for the region's groups:",
      len(reg_dict), "groups to  test")

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

snrs = [20,]

for f in filters:

    hlr_app_dict[tag].setdefault(f, {})
    hlr_pix_dict[tag].setdefault(f, {})

    for r in radii_fracs:
        hlr_app_dict[tag][f].setdefault(r, [])
        hlr_pix_dict[tag][f].setdefault(r, [])

    flux_dict[tag].setdefault(f, {})
    img_dict[tag].setdefault(f, {})
    segm_dict[tag].setdefault(f, {})
    ngal_dict[tag].setdefault(f, {})
    grp_dict[tag].setdefault(f, {})
    sf_ngal_dict[tag].setdefault(f, {})

    for snr in snrs:

        flux_dict[tag][f].setdefault(snr, [])
        img_dict[tag][f].setdefault(snr, [])
        segm_dict[tag][f].setdefault(snr, [])
        ngal_dict[tag][f].setdefault(snr, [])
        sf_ngal_dict[tag][f].setdefault(snr, [])
        grp_dict[tag][f].setdefault(snr, [])

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

                # img = util.make_soft_img(this_pos, res, 0, 1, imgrange,
                #                          this_flux,
                #                          this_smls)
                img = util.make_spline_img(this_pos, res, 0, 1, tree,
                                           this_flux, this_smls)

                img = gaussian_filter(img, 3)

                img = util.noisy_img(img, snr=snr, seed=10000)

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

            try:
                # img[img < 10**21] = 0
                threshold = phut.detect_threshold(img, nsigma=5)
                # threshold = np.median(img)

                segm = phut.detect_sources(img, threshold, npixels=5)
                segm = phut.deblend_sources(img, segm, npixels=5,
                                            nlevels=32, contrast=0.001)
            except:
                continue

            # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
            # ax1.grid(False)
            # ax2.grid(False)
            # plt_img = np.zeros_like(img)
            # plt_img[img > 0] = np.log10(img[img > 0])
            # ax1.imshow(plt_img, extent=imgextent, cmap="Greys_r")
            # cmap = segm.make_cmap()
            # ax2.imshow(segm.data, extent=imgextent, cmap=cmap)
            # fig.savefig("plots/gal_img_log_" + f + "_%.1f.png"
            #             % np.log10(np.sum(img)), dpi=300)
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
            img_dict[tag][f][snr].append(img)
            segm_dict[tag][f][snr].append(segm.data)
            grp_dict[tag][f][snr].append(ind)

            begin.append(len(fluxes))
            grp_mass.append(this_groupmass)
            star_pos.extend(this_pos)
            smls.extend(this_smls)
            fluxes.extend(this_flux)
            subgrpids.extend(this_subgrpids)
            Slen.append(len(this_smls))

            ngal = 0
            for gal in np.unique(segm.data):
                if np.sum(img[segm.data == gal]) > 10 and gal > 0:
                    ngal += 1

            ngal_dict[tag][f][snr].append(ngal)

            ngal = 0
            for gal in subfind_ids:
                if np.sum(this_flux[this_subgrpids == gal]) > 10:
                    ngal += 1

            sf_ngal_dict[tag][f][snr].append(ngal)


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

    begin = np.array(begin)
    grp_mass = np.array(grp_mass)
    fluxes = np.array(fluxes)
    subgrpids = np.array(subgrpids)
    Slen = np.array(Slen)
    smls = np.array(smls)
    star_pos = np.array(star_pos)

    try:
        dset = f_group.create_dataset("Start_Index", data=begin,
                                        dtype=begin.dtype,
                                        shape=begin.shape,
                                        compression="gzip")
        dset.attrs["units"] = "None"
    except ValueError:
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
    except ValueError:
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
    except ValueError:
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
    except ValueError:
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
    except ValueError:
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
    except ValueError:
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
    except ValueError:
        print("Group_Length already exists: Overwriting...")
        del f_group["Group_Length"]
        dset = f_group.create_dataset("Group_Length", data=Slen,
                                        dtype=Slen.dtype,
                                        shape=Slen.shape,
                                        compression="gzip")
        dset.attrs["units"] = "None"

    for snr in snrs:

        # fluxes = np.array(flux_dict[tag][f][snr])
        imgs = np.array(img_dict[tag][f][snr])
        segms = np.array(segm_dict[tag][f][snr])
        ngals = np.array(ngal_dict[tag][f][snr])
        sf_ngals = np.array(sf_ngal_dict[tag][f][snr])
        grps = np.array(grp_dict[tag][f][snr])

        print(imgs.shape)
            
        try:
            snr_group = f_group[str(snr)]
        except KeyError:
            print(f, "Doesn't exists: Creating...")
            snr_group = f_group.create_group(str(snr))

        try:
            dset = snr_group.create_dataset("Group_ID", data=grps,
                                          dtype=grps.dtype,
                                          shape=grps.shape,
                                          compression="gzip")
            dset.attrs["units"] = "None"
        except ValueError:
            print("Group_ID already exists: Overwriting...")
            del snr_group["Group_ID"]
            dset = snr_group.create_dataset("Group_ID", data=grps,
                                          dtype=grps.dtype,
                                          shape=grps.shape,
                                          compression="gzip")
            dset.attrs["units"] = "None"

        try:
            dset = snr_group.create_dataset("Images", data=imgs,
                                          dtype=imgs.dtype,
                                          shape=imgs.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$nJy$"
        except ValueError:
            print("Images already exists: Overwriting...")
            del snr_group["Images"]
            dset = snr_group.create_dataset("Images", data=imgs,
                                          dtype=imgs.dtype,
                                          shape=imgs.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$nJy$"

        try:
            dset = snr_group.create_dataset("Segmentation_Maps", data=segms,
                                          dtype=segms.dtype,
                                          shape=segms.shape,
                                          compression="gzip")
            dset.attrs["units"] = "None"
        except ValueError:
            print("Segmentation_Maps already exists: Overwriting...")
            del snr_group["Segmentation_Maps"]
            dset = snr_group.create_dataset("Segmentation_Maps", data=segms,
                                          dtype=segms.dtype,
                                          shape=segms.shape,
                                          compression="gzip")
            dset.attrs["units"] = "None"

        try:
            dset = snr_group.create_dataset("NGalaxy", data=ngals,
                                          dtype=ngals.dtype,
                                          shape=ngals.shape,
                                          compression="gzip")
            dset.attrs["units"] = "None"
        except ValueError:
            print("NGalaxy already exists: Overwriting...")
            del snr_group["NGalaxy"]
            dset = snr_group.create_dataset("NGalaxy", data=ngals,
                                          dtype=ngals.dtype,
                                          shape=ngals.shape,
                                          compression="gzip")
            dset.attrs["units"] = "None"

        try:
            dset = snr_group.create_dataset("SUBFIND_NGalaxy", data=sf_ngals,
                                          dtype=sf_ngals.dtype,
                                          shape=sf_ngals.shape,
                                          compression="gzip")
            dset.attrs["units"] = "None"
        except ValueError:
            print("SUBFIND_NGalaxy already exists: Overwriting...")
            del snr_group["SUBFIND_NGalaxy"]
            dset = snr_group.create_dataset("SUBFIND_NGalaxy", data=sf_ngals,
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
        #     except ValueError:
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
        #     except ValueError:
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

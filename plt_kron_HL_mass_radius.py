#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

os.environ['FLARE'] = '/cosma7/data/dp004/dc-wilk2/flare'

matplotlib.use('Agg')
warnings.filterwarnings('ignore')
import seaborn as sns
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
import photutils as phut
from photutils.segmentation import SourceCatalog
import h5py
import sys
from astropy.cosmology import Planck13 as cosmo
import eritlux.simulations.imagesim as imagesim
import flare.surveys
import flare.plots.image
import utilities as util
from scipy.spatial import cKDTree

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

# Set orientation
orientation = sys.argv[1]

# Define luminosity and dust model types
Type = sys.argv[2]
extinction = 'default'

# Define filter
filters = ('Hubble.WFC3.f160w', )

depths = [0.1, 1, 5, 10, 20]

thresh = 2.5

survey_id = 'XDF'  # the XDF (updated HUDF)
field_id = 'dXDF'  # deepest sub-region of XDF (defined by a mask)

# --- get field info object. This contains the filters, depths,
# image location etc. (if real image)
field = flare.surveys.surveys[survey_id].fields[field_id]

# Define width
ini_width_pkpc = 500

for n_z in range(len(snaps)):

    if len(sys.argv) > 3:
        if n_z != int(sys.argv[3]):
            continue

    kron_radii_dict = {}
    kron_flux_dict = {}

    for f in filters:

        snap = snaps[n_z]

        z_str = snap.split('z')[1].split('p')
        z = float(z_str[0] + '.' + z_str[1])

        arcsec_per_kpc_proper = cosmo.arcsec_per_kpc_proper(z).value

        ini_width = ini_width_pkpc * arcsec_per_kpc_proper

        # --- initialise ImageCreator object
        image_creator = imagesim.Idealised(f, field)

        arc_res = image_creator.pixel_scale
        kpc_res = arc_res / arcsec_per_kpc_proper

        # Compute the resolution
        ini_res = ini_width / arc_res
        res = int(np.ceil(ini_res))
        cutout_halfsize = int(res * 0.1)

        # Compute the new width
        width = arc_res * res

        print("Filter:", f)
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

        # Define x and y positions of pixels
        X, Y = np.meshgrid(np.linspace(imgrange[0][0], imgrange[0][1], res),
                           np.linspace(imgrange[1][0], imgrange[1][1], res))

        # Define pixel position array for the KDTree
        pix_pos = np.zeros((X.size, 2))
        pix_pos[:, 0] = X.ravel()
        pix_pos[:, 1] = Y.ravel()

        # Build KDTree
        tree = cKDTree(pix_pos)

        for reg in regions:

            print("Getting SUBFIND occupancy with orientation {o}, type {t}, "
                  "and extinction {e} for region {x} and "
                  "snapshot {u}".format(o=orientation, t=Type, e=extinction,
                                        x=reg, u=snap))

            for depth in depths:

                try:
                    hdf = h5py.File("mock_data/flares_segm_{}_{}_{}_{}.hdf5"
                                    .format(reg, snap, Type, orientation), "r")
                except OSError as e:
                    print(e)
                    continue

                try:
                    f_group = hdf[f]
                    fdepth_group = f_group[str(depth)]

                    imgs = fdepth_group["Images"]
                    sigs = fdepth_group["Significance_Images"]
                    subfind_spos = f_group["Star_Pos"][:]
                    all_smls = f_group["Smoothing_Length"][:]
                    subgrpids = f_group["Part_subgrpids"][:]
                    begin = f_group["Start_Index"][:]
                    group_len = f_group["Group_Length"][:]
                    gal_ids = f_group["Subgroup_IDs"][:]
                    gal_ms = f_group["Galaxy Mass"][:]

                except KeyError as e:
                    print(e)
                    hdf.close()
                    continue

                kron_radii = []
                fluxes = []

                if sigs.shape[0] == 0:
                    continue

                if sigs[:].max() < thresh:
                    continue

                for ind in range(imgs.shape[0]):

                    poss = subfind_spos[
                           begin[ind]: begin[ind] + group_len[ind], (0, 1)]
                    subgrp = subgrpids[begin[ind]: begin[ind] + group_len[ind]]
                    smooth = all_smls[begin[ind]: begin[ind] + group_len[ind]]
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

                    subfind_img = util.make_uni_subfind_spline_img(poss, res, 0, 1,
                                                               tree, subgrp,
                                                               smooth, gal_ids, subgrp,
                                                               spline_cut_off=5 / 2)

                    source_cat = SourceCatalog(img, segm, error=None,
                                               mask=None,  kernel=None,
                                               background=None, wcs=None,
                                               localbkg_width=0,
                                               apermask_method='correct',
                                               kron_params=(2.5, 0.0),
                                               detection_cat=None)

                    try:
                        labels = source_cat.labels
                        radii = source_cat.fluxfrac_radius(0.5) * kpc_res
                    except ValueError:
                        continue

                    for i, r in zip(labels, radii):
                        this_ids = np.unique(subfind_img[segm.data == i])

                        for ii in this_ids:
                            kron_radii.append(r)
                            fluxes.append(gal_ms[gal_ids == ii])

                hdf.close()

                kron_radii_dict.setdefault(f + "." + str(depth), []).extend(kron_radii)
                kron_flux_dict.setdefault(f + "." + str(depth), []).extend(
                    fluxes)

        fig = plt.figure(figsize=(4, 10))
        gs = gridspec.GridSpec(len(depths), 1)
        gs.update(wspace=0.0, hspace=0.0)
        axes = []
        for i in range(len(depths)):
            axes.append(fig.add_subplot(gs[i, 0]))

        for ax, depth in zip(axes, depths):

            fdepth = f + "." + str(depth)

            if not fdepth in kron_radii_dict.keys():
                continue

            try:
                print(kron_flux_dict[fdepth].shape, kron_radii_dict[fdepth].shape)
                print(kron_flux_dict[fdepth])
                print(kron_radii_dict[fdepth])
                cbar = ax.hexbin(kron_flux_dict[fdepth],
                                 kron_radii_dict[fdepth],
                                 gridsize=50, mincnt=1,
                                 xscale='log', yscale='log',
                                 norm=LogNorm(), linewidths=0.2,
                                 cmap='viridis')
            except ValueError as e:
                print(e)
                continue

        axes[0].text(0.95, 0.05, f'$z={z}$',
                     bbox=dict(boxstyle="round,pad=0.3", fc='w',
                               ec="k", lw=1, alpha=0.8),
                     transform=ax.transAxes, horizontalalignment='right',
                     fontsize=8)

        # Label axes
        axes[-1].set_xlabel(r'$M_\star/$ M_\odot')
        for ax in axes:
            ax.set_ylabel('$R_{1/2}/ [pkpc]$')
            ax.set_ylim(10**-1.5, 10**2)
            ax.set_xlim(10 ** -3, 10 ** 4)

        for ax in axes[:-1]:
            ax.tick_params(axis='x', top=False, bottom=False,
                           labeltop=False, labelbottom=False)

        axes[-1].tick_params(axis='x', which='minor', bottom=True)

        fig.savefig("plots/HalfLightRadiusvsMass_Filter-" + f + "_Orientation-" + orientation + "_Type-" + Type + "_Snap-" + snap + ".png", bbox_inches="tight")

        plt.close(fig)


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
import matplotlib as mpl
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from scipy.stats import binned_statistic
from matplotlib.lines import Line2D
from astropy.cosmology import Planck13 as cosmo
import astropy.units as u
from flare.photom import lum_to_M, M_to_lum, lum_to_flux, m_to_flux
import flare.photom as photconv
from scipy.spatial import cKDTree
import h5py
import sys
import pandas as pd
import utilities as util
import phot_modules as phot
import utilities as util
import eritlux.simulations.imagesim as imagesim
import flare.surveys
import flare.plots.image
import photutils as phut

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

# Set orientation
orientation = sys.argv[3]

# Define luminosity and dust model types
Type = sys.argv[4]
extinction = 'default'

# Define filter
filters = [f'Hubble.WFC3.{f}' for f in ['f105w', 'f125w', 'f140w', 'f160w']] \
          + [f'Hubble.ACS.{f}'
             for f in ['f435w', 'f606w', 'f775w', 'f814w', 'f850lp']]

depths = [0.1, 1, 5, 10, 20, "SUBFIND"]

reg_ind = int(sys.argv[1])
snap_ind = int(sys.argv[2])

reg, snap = regions[reg_ind], snaps[snap_ind]

z_str = snap.split('z')[1].split('p')
z = float(z_str[0] + '.' + z_str[1])

print("Making images for with orientation {o}, type {t}, "
      "and extinction {e} for region {x} and "
      "snapshot {u}".format(o=orientation, t=Type, e=extinction,
                            x=reg, u=snap))

arcsec_per_kpc_proper = cosmo.arcsec_per_kpc_proper(z).value

# Define width
ini_width_pkpc = 500
ini_width = ini_width_pkpc * arcsec_per_kpc_proper

survey_id = 'XDF'  # the XDF (updated HUDF)
field_id = 'dXDF'  # deepest sub-region of XDF (defined by a mask)

# --- get field info object. This contains the filters, depths,
# image location etc. (if real image)
field = flare.surveys.surveys[survey_id].fields[field_id]

thresh = 2.5

for f in filters:

    # --- initialise ImageCreator object
    image_creator = imagesim.Idealised(f, field)

    arc_res = image_creator.pixel_scale

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

    print("Pixel tree built")

    hdf = h5py.File("mock_data/flares_segm_{}_{}_{}_{}_{}.hdf5"
                    .format(reg, snap, Type, orientation, f),
                    "r")
    print(list(hdf.keys()))
    fdepth_group = hdf["0.1"]

    max_imgs = fdepth_group["Images"][:]
    group_len = fdepth_group["Image_Length"][:]
    img_ids = fdepth_group["Image_ID"][:]

    hdf.close()

    if max_imgs.size == 0:
        continue

    sinds = np.argsort(group_len)[::-1]
    create_img_ids = img_ids[sinds][:10]
    max_imgs = max_imgs[sinds]

    for img_ind in range(create_img_ids.size):

        img_id = create_img_ids[img_ind]

        hdf = h5py.File("mock_data/flares_segm_{}_{}_{}_{}_{}.hdf5"
                        .format(reg, snap, Type, orientation, f),
                        "r")

        norm_imgs = []
        norm_sigs = []
        for i in depths[:-1]:

            fdepth_group = hdf[str(i)]

            norm_imgs.append(fdepth_group["Images"][img_id, :, :])
            norm_sigs.append(np.max(fdepth_group["Images"][img_id, :, :]
                             / fdepth_group["Noise_value"][img_id]))

        hdf.close()

        img_norm = mpl.colors.Normalize(vmin=np.log10(-np.percentile(norm_imgs, 33.175)),
                                        vmax=np.log10(np.percentile(norm_imgs, 99)))
        sig_norm = mpl.colors.TwoSlopeNorm(vmin=0., vcenter=2.5,
                                           vmax=np.max(norm_sigs))
        bi_cmap = matplotlib.colors.ListedColormap(['k', 'w'])
        bounds = [0, 0.5, 1]
        bi_norm = matplotlib.colors.BoundaryNorm(bounds, bi_cmap.N)

        fig = plt.figure(figsize=(10, 11))
        gs = gridspec.GridSpec(5, 7, width_ratios=[10, 10, 10, 10, 10, 10, 1])
        gs.update(wspace=0.0, hspace=0.0)
        axes = np.empty((7, 5), dtype=object)
        for i in range(7):
            for j in range(5):
                axes[i, j] = fig.add_subplot(gs[j, i])

        for i in range(7):
            for j in range(5):
                ax = axes[i, j]
                ax.grid(False)

                if j < axes.shape[1]:
                    ax.tick_params(axis='x', top=False, bottom=False,
                                   labeltop=False, labelbottom=False)
                if i > 0:
                    ax.tick_params(axis='y', left=False, right=False,
                                   labelleft=False, labelright=False)

        for i, depth in enumerate(depths):

            hdf = h5py.File("mock_data/flares_segm_{}_{}_{}_{}_{}.hdf5"
                            .format(reg, snap, Type, orientation, f),
                            "r")

            fdepth_group = hdf[str(depth)]

            try:
                all_smls = fdepth_group["Smoothing_Length"][:]
                subfind_spos = fdepth_group["Star_Pos"][:]
                begin = fdepth_group["Start_Index"][:]
                group_len = fdepth_group["Image_Length"][:]
                fluxes = fdepth_group["Fluxes"][:]
                img_ids = fdepth_group["Image_ID"][:]
            except KeyError:
                all_smls = None
                subfind_spos = None
                begin = None
                group_len = None
                fluxes = None
                img_ids = None

            ind = int(img_id)

            print("Creating image", img_id, ind)

            if len(all_smls) > 0:

                if depth == "SUBFIND":
                    this_pos = subfind_spos[begin[ind]:
                                            begin[ind] + group_len[ind], (0, 1)]
                    smooth = all_smls[begin[ind]: begin[ind] + group_len[ind]]
                    this_flux = fluxes[begin[ind]: begin[ind] + group_len[ind]]

                    img = util.make_spline_img(this_pos, res, 0, 1, tree,
                                               this_flux, smooth)
                    sig = np.zeros((res, res))
                    segm = np.zeros((res, res))
                    db_segm = np.zeros((res, res))
                    resi = np.zeros((res, res))
                else:
                    imgs = fdepth_group["Images"]
                    noise = fdepth_group["Noise_value"]
                    img = imgs[ind, :, :]
                    sig = img / noise[ind]
                    try:
                        segm = phut.detect_sources(sig, thresh, npixels=5)
                        db_segm = phut.deblend_sources(img, segm, npixels=5,
                                                    nlevels=32, contrast=0.001)
                        segm = segm.data
                        db_segm = db_segm.data
                        resi = np.abs(segm - db_segm)
                        resi[resi > 0] = 1
                    except TypeError:
                        segm = np.zeros((res, res))
                        db_segm = np.zeros((res, res))
                        resi = np.zeros((res, res))

            else:
                img = np.zeros((res, res))
                sig = np.zeros((res, res))
                segm = np.zeros((res, res))
                db_segm = np.zeros((res, res))
                resi = np.zeros((res, res))

            plt_img = np.zeros_like(img)
            plt_img[img > 0] = np.log10(img[img > 0])
            segm[segm == 0] = np.nan
            db_segm[db_segm == 0] = np.nan
            axes[i, 0].imshow(plt_img, extent=imgextent, cmap="Greys_r",
                              norm=img_norm)
            axes[i, 1].imshow(sig, extent=imgextent, cmap="coolwarm",
                              norm=sig_norm)
            img_segm = axes[i, 2].imshow(segm, extent=imgextent,
                                         cmap="gist_rainbow")
            img_dbsegm = axes[i, 3].imshow(db_segm, extent=imgextent,
                                           cmap="gist_rainbow")
            axes[i, 4].imshow(resi, extent=imgextent, cmap=bi_cmap)

            if not os.path.exists("plots/Gal_imgs"):
                os.makedirs("plots/Gal_imgs")

            fig.colorbar(mpl.cm.ScalarMappable(norm=img_norm, cmap="Greys_r"),
                         cax=axes[0, -1], label='Flux (nJy)',fraction=0.046, pad=0.04)
            fig.colorbar(mpl.cm.ScalarMappable(norm=sig_norm, cmap="coolwarm"),
                         cax=axes[1, -1], label='Flux (nJy)',fraction=0.046, pad=0.04)
            fig.colorbar(img_segm,
                         cax=axes[2, -1], label='Label',fraction=0.046, pad=0.04)
            fig.colorbar(img_dbsegm,
                         cax=axes[3, -1], label='Label',fraction=0.046, pad=0.04)
            fig.colorbar(mpl.cm.ScalarMappable(norm=bi_norm, cmap=bi_cmap),
                         cax=axes[4, -1], ticks=[0, 1], tick_labels=["UC",
                                                                     "DB"],
                         fraction=0.046, pad=0.04)

            fig.savefig("plots/Gal_imgs/gal_img_Filter-" + f
                        + "_Region-" + reg + "_Snap-" + snap + "_Group-"
                        + str(img_id) + ".png", dpi=600)
            plt.close(fig)

            hdf.close()





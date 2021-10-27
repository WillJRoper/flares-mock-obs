#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import os
import re
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

os.environ['FLARE'] = '/cosma7/data/dp004/dc-wilk2/flare'

matplotlib.use('Agg')
warnings.filterwarnings('ignore')
import seaborn as sns
import photutils as phut
from matplotlib.colors import LogNorm, Normalize
import matplotlib.gridspec as gridspec
from photutils.segmentation import SourceCatalog
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma
import h5py
import sys
from flare.photom import m_to_flux, flux_to_m

sns.set_context("paper")
sns.set_style('white')

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
filters = [f'Hubble.ACS.{f}'
           for f in ['f435w', 'f606w', 'f775w', 'f814w', 'f850lp']] \
          + [f'Hubble.WFC3.{f}' for f in ['f105w', 'f125w', 'f140w', 'f160w']]

# Set up depths relative to the Xtreme deep field
XDF_depth_m = 31.2
XDF_depth_flux = m_to_flux(XDF_depth_m)
depths = [XDF_depth_flux * 0.01, XDF_depth_flux * 0.1,
          XDF_depth_flux, 10 * XDF_depth_flux, 100 * XDF_depth_flux]
depths_m = [flux_to_m(d) for d in depths]
print(depths)
print([d / 5 for d in depths])

reg_ind = int(sys.argv[1])
snap_ind = int(sys.argv[2])

reg, snap = regions[reg_ind], snaps[snap_ind]

z_str = snap.split('z')[1].split('p')
z = float(z_str[0] + '.' + z_str[1])

n_img = int(sys.argv[5])

ind = 0

hdf = h5py.File("mock_data/flares_segm_{}_{}_{}_{}_{}.hdf5"
                .format(reg, snap, Type, orientation, filters[-1]), "r")

imgs = hdf[str(depths[0])]["Images"][:]
sinds = np.argsort(np.nansum(imgs, axis=(1, 2)))[::-1]
# sinds = np.arange(0, imgs.shape[0])
hdf.close()

# Set up segmentation image variables and kernel
thresh = 2.5
sigma = 3.0 * gaussian_fwhm_to_sigma  # FWHM = 3.
kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
kernel.normalize()
quantities = ('label', 'kron_flux')

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
print("Filters rdder then the Lyman break:", detect_filters)

while ind < n_img:

    img_ind = sinds[ind]

    print("Creating image", ind, img_ind)

    img_dict = {}
    noise_dict = {}

    # Set up dictionary to store the flux in each filter
    fluxes = {}

    for depth, mdepth in zip(depths, depths_m):
        for f in filters:
            hdf = h5py.File("mock_data/flares_segm_{}_{}_{}_{}_{}.hdf5"
                            .format(reg, snap, Type, orientation, f), "r")

            fdepth_group = hdf[str(depth)]

            img = fdepth_group["Images"][img_ind]
            noise = fdepth_group["Noise_value"][img_ind]
            mimg = fdepth_group["Mass_Images"][img_ind]

            fluxes.setdefault(mdepth, []).append(np.nansum(img))

            img_dict.setdefault(mdepth, {})[f] = img
            noise_dict.setdefault(mdepth, {})[f] = noise
            img_dict.setdefault(mdepth, {})["Mass"] = mimg

            hdf.close()

    lams = []
    for f in filters:
        if f.split(".")[1] == "WFC3":
            lams.append(int(re.findall(r'\d+', f.split(".")[-1])[0]) * 100)
        else:
            lams.append(int(re.findall(r'\d+', f.split(".")[-1])[0]) * 10)
    lams = np.array(lams)
    lams_sinds = np.argsort(lams)

    segms = {}
    db_segms = {}

    for d in depths_m:

        print("----------------", d, "----------------")

        detection_img = np.zeros(img_dict[d][filters[0]].shape)
        weight_img = np.zeros(img_dict[d][filters[0]].shape)
        noise_img = np.zeros(img_dict[d][filters[0]].shape)

        for f in detect_filters:
            detection_img += (img_dict[d][f] / noise_dict[d][f] ** 2)
            weight_img += (1 / noise_dict[d][f] ** 2)
            noise_img += (1 / noise_dict[d][f])

        detection_img /= weight_img
        noise_img /= weight_img
        n = np.max(noise_img)  # All values are the same

        sig = detection_img / noise_img

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.tick_params(axis='both', top=False, bottom=False,
                       labeltop=False, labelbottom=False,
                       left=False, right=False,
                       labelleft=False, labelright=False)
        im = ax.imshow(detection_img, cmap="plasma")
        fig.colorbar(im)
        fig.savefig("plots/detect_Depth-%.2f" % d
                    + "Orientation-"
                    + orientation + "_Type-" + Type
                    + "_Region-" + reg + "_Snap-" + snap + "_Group-"
                    + str(img_ind) + ".png", bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.tick_params(axis='both', top=False, bottom=False,
                       labeltop=False, labelbottom=False,
                       left=False, right=False,
                       labelleft=False, labelright=False)
        im = ax.imshow(sig, cmap="plasma")
        fig.colorbar(im)
        fig.savefig("plots/SNRmap_Depth-%.2f" % d
                    + "Orientation-"
                    + orientation + "_Type-" + Type
                    + "_Region-" + reg + "_Snap-" + snap + "_Group-"
                    + str(img_ind) + ".png", bbox_inches="tight")
        plt.close(fig)

        try:
            segm = phut.detect_sources(sig, thresh, npixels=5, kernel=kernel)
            segms[d] = segm
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.tick_params(axis='both', top=False, bottom=False,
                           labeltop=False, labelbottom=False,
                           left=False, right=False,
                           labelleft=False, labelright=False)
            im = ax.imshow(segm, cmap="turbo")
            fig.colorbar(im)
            fig.savefig("plots/segmap_Depth-%.2f" % d
                        + "Orientation-"
                        + orientation + "_Type-" + Type
                        + "_Region-" + reg + "_Snap-" + snap + "_Group-"
                        + str(img_ind) + ".png", bbox_inches="tight")
            plt.close(fig)
            segm = phut.deblend_sources(detection_img, segm,
                                        npixels=5, nlevels=16,
                                        contrast=0.01, kernel=kernel)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.tick_params(axis='both', top=False, bottom=False,
                           labeltop=False, labelbottom=False,
                           left=False, right=False,
                           labelleft=False, labelright=False)
            im = ax.imshow(segm, cmap="turbo")
            fig.colorbar(im)
            fig.savefig("plots/deblend_Depth-%.2f" % d
                        + "Orientation-"
                        + orientation + "_Type-" + Type
                        + "_Region-" + reg + "_Snap-" + snap + "_Group-"
                        + str(img_ind) + ".png", bbox_inches="tight")
            plt.close(fig)
            source_cat = SourceCatalog(detection_img, segm,
                                       error=None, mask=None,
                                       kernel=kernel, background=None,
                                       wcs=None, localbkg_width=0,
                                       apermask_method='correct',
                                       kron_params=(2.5, 0.0),
                                       detection_cat=None)

            tab = source_cat.to_table(columns=quantities)

            for i in tab["label"]:
                if flux_to_m(tab['kron_flux'][i - 1]) > d:
                    segm.remove_label(i)

            db_segms[d] = segm
            print("Sources:", np.unique(segm.data).size - 1)

        except TypeError:
            segms[d] = np.zeros(img_dict[d][filters[0]].shape)
            db_segms[d] = np.zeros(img_dict[d][filters[0]].shape)

    all_imgs = np.array([img_dict[d][f] for f in filters for d in depths_m])
    all_mimgs = np.array([img_dict[d]["Mass"] for d in depths_m])
    vmins = {d: np.percentile([img_dict[d][f] for f in filters], 16)
             for d in depths_m}
    # vmin = np.min(all_imgs)
    vmaxs = {d: np.percentile([img_dict[d][f] for f in filters], 99)
             for d in depths_m}
    mass_vmin = np.percentile(all_mimgs[all_mimgs > 0], 16)
    # mass_vmin = np.min(all_mimgs)
    mass_vmax = np.percentile(all_mimgs[all_mimgs > 0], 99)

    mimg_norm = LogNorm(vmin=mass_vmin, vmax=mass_vmax, clip=True)

    fig = plt.figure(figsize=(len(filters) + 4, len(depths)),
                     dpi=all_imgs.shape[-1])
    gs = gridspec.GridSpec(ncols=len(filters) + 4, nrows=len(depths),
                           width_ratios=(len(filters) + 4) * [1, ],
                           height_ratios=len(depths) * [1., ])
    gs.update(wspace=0.0, hspace=0.0)

    # flux_ax = fig.add_subplot(gs[-1, :])
    # flux_ax.grid(True)
    # flux_ax.semilogy()

    axes = np.zeros((len(depths), len(filters) + 4), dtype=object)
    for i in range(len(depths)):
        for j in range(len(filters) + 4):
            axes[i, j] = fig.add_subplot(gs[i, j])

    for i, d in enumerate(depths_m):

        img_norm = Normalize(vmin=vmins[d], vmax=vmaxs[d], clip=True)

        # if d == flux_to_m(XDF_depth_flux):
        #     line = "-"
        #     fs = np.array(fluxes[d])
        #
        #     flux_ax.plot(lams[lams_sinds], fs[lams_sinds],
        #                  linestyle=line, marker="+",
        #                  label=r"$m=%.1f \times m_{\mathrm{XDF}}$"
        #                        % (depths[i] / XDF_depth_flux))

        for j, f in enumerate(filters):
            ax = axes[i, j + 1]
            ax.tick_params(axis='both', top=False, bottom=False,
                           labeltop=False, labelbottom=False,
                           left=False, right=False,
                           labelleft=False, labelright=False)

            plt_img = img_dict[d][f]
            ax.imshow(plt_img, cmap="magma", norm=img_norm)

            if i == 0:
                ax.set_title(f.split(".")[-1])
            # if j == 0:
            #     ax.set_ylabel(r"$%.2f \times m_{\mathrm{XDF}}$"
            #                   % (depths[i] / XDF_depth_flux), fontsize=6)

        ax = axes[i, 0]
        ax.tick_params(axis='both', top=False, bottom=False,
                       labeltop=False, labelbottom=False,
                       left=False, right=False,
                       labelleft=False, labelright=False)

        plt_img = img_dict[d]["Mass"]
        ax.imshow(plt_img, cmap="plasma", norm=mimg_norm)

        ax.set_ylabel(r"$%.2f \times m_{\mathrm{XDF}}$"
                      % (depths[i] / XDF_depth_flux), fontsize=6)

        if i == 0:
            ax.set_title("Mass")

        ax = axes[i, -3]
        ax.tick_params(axis='both', top=False, bottom=False,
                       labeltop=False, labelbottom=False,
                       left=False, right=False,
                       labelleft=False, labelright=False)

        plt_img = segms[d]
        ax.imshow(plt_img, cmap="turbo")

        if i == 0:
            ax.set_title("Segm")

        ax = axes[i, -2]
        ax.tick_params(axis='both', top=False, bottom=False,
                       labeltop=False, labelbottom=False,
                       left=False, right=False,
                       labelleft=False, labelright=False)

        plt_img = db_segms[d]
        ax.imshow(plt_img, cmap="turbo")

        if i == 0:
            ax.set_title("Deblend")

        ax = axes[i, -1]
        ax.tick_params(axis='both', top=False, bottom=False,
                       labeltop=False, labelbottom=False,
                       left=False, right=False,
                       labelleft=False, labelright=False)

        try:
            okinds = db_segms[d].data > 0
            plt_img = np.full_like(img_dict[d]["Mass"], np.nan)
            plt_img[okinds] = img_dict[d]["Mass"][okinds]
            ax.imshow(plt_img, cmap="plasma", norm=mimg_norm)

            plt_img = np.full_like(img_dict[d]["Mass"], np.nan)
            plt_img[~okinds] = img_dict[d]["Mass"][~okinds]
            ax.imshow(plt_img, cmap="plasma", norm=mimg_norm, alpha=0.1)
        except (AttributeError, TypeError):
            ax.imshow(img_dict[d]["Mass"], cmap="plasma",
                      norm=mimg_norm, alpha=0.1)

        if i == 0:
            ax.set_title("Segm Mass")

    # cmap = mpl.cm.magma
    # cmap2 = mpl.cm.plasma

    # cax = flux_ax.inset_axes([0.9, 0.1, 0.02, 0.5])
    # cax2 = flux_ax.inset_axes([0.8, 0.1, 0.02, 0.5])
    #
    # for ax in (cax, cax2):
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['bottom'].set_visible(False)
    #     ax.spines['left'].set_visible(False)
    #
    # cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
    #                                  norm=img_norm)
    # cbar2 = mpl.colorbar.ColorbarBase(cax2, cmap=cmap2,
    #                                   norm=mimg_norm)
    # cbar.set_label("$F/[\mathrm{nJy}]$")
    # cbar2.set_label("$M/M_\odot$")

    # flux_ax.set_ylabel("$F / [\mathrm{nJy}]$")
    # flux_ax.set_xlabel(r"$\lambda / [\AA]$")
    # flux_ax.legend()

    fig.savefig("plots/gal_img_grid_Orientation-"
                + orientation + "_Type-" + Type
                + "_Region-" + reg + "_Snap-" + snap + "_Group-"
                + str(img_ind) + ".png", bbox_inches="tight")
    plt.close(fig)

    ind += 1

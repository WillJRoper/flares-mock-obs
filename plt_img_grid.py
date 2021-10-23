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
from matplotlib.colors import LogNorm, Normalize
import matplotlib.gridspec as gridspec
import matplotlib as mpl
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
depths = [XDF_depth_flux * 0.1, XDF_depth_flux,
          2 * XDF_depth_flux, 10 * XDF_depth_flux]
depths_m = [flux_to_m(XDF_depth_flux * 0.01), flux_to_m(XDF_depth_flux * 0.1),
            flux_to_m(XDF_depth_flux), flux_to_m(10 * XDF_depth_flux)]
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
hdf.close()

while ind < n_img:

    img_ind = sinds[ind]

    print("Creating image", ind, img_ind)

    img_dict = {}

    mass_vmin, mass_vmax = np.inf, 0

    for depth, mdepth in zip(depths, depths_m):
        for f in filters:

            hdf = h5py.File("mock_data/flares_segm_{}_{}_{}_{}_{}.hdf5"
                            .format(reg, snap, Type, orientation, f), "r")

            fdepth_group = hdf[str(depth)]

            img = fdepth_group["Images"][img_ind]
            mimg = fdepth_group["Mass_Images"][img_ind]
            print(depth, mdepth, np.max(fdepth_group["Noise_value"][img_ind]))

            # if np.max(mimg) > mass_vmax:
            #     mass_vmax = np.max(mimg)
            # if np.min(mimg) < mass_vmin:
            #     mass_vmin = np.min(mimg)

            img_dict.setdefault(mdepth, {})[f] = img
            img_dict.setdefault(mdepth, {})["Mass"] = mimg

            hdf.close()

    all_imgs = np.array([img_dict[d][f] for f in filters for d in depths_m])
    all_mimgs = np.array([img_dict[d]["Mass"] for d in depths_m])
    vmin = np.percentile(all_imgs, 16)
    vmax = np.percentile(all_imgs, 99)
    mass_vmin = np.percentile(all_mimgs[all_mimgs > 0], 16)
    mass_vmax = np.percentile(all_mimgs[all_mimgs > 0], 99)
    img_norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    mimg_norm = LogNorm(vmin=mass_vmin, vmax=mass_vmax, clip=True)
    print(vmin, vmax, mass_vmax)
    fig = plt.figure(figsize=(len(filters) + 1, len(depths)),
                     dpi=all_imgs.shape[-1])
    gs = gridspec.GridSpec(ncols=len(filters) + 2, nrows=len(depths),
                           width_ratios=(len(filters) + 1) * [10, ] + [1, ])
    gs1 = gridspec.GridSpec(ncols=len(filters) + 2, nrows=len(depths),
                            width_ratios=(len(filters) + 1) * [10, ] + [1, ])
    gs.update(wspace=0.0, hspace=0.0)
    gs1.update(wspace=0.2, hspace=0.0)
    cax = fig.add_subplot(gs1[:, -1])
    # cax2 = cax.twinx()
    axes = np.zeros((len(depths), len(filters) + 1), dtype=object)
    for i in range(len(depths)):
        for j in range(len(filters) + 1):
            axes[i, j] = fig.add_subplot(gs[i, j])

    for i, d in enumerate(depths_m):
        for j, f in enumerate(filters):
            ax = axes[i, j]
            ax.tick_params(axis='both', top=False, bottom=False,
                           labeltop=False, labelbottom=False,
                           left=False, right=False,
                           labelleft=False, labelright=False)

            plt_img = img_dict[d][f]
            ax.imshow(plt_img, cmap="magma", norm=img_norm)

            if i == 0:
                ax.set_title(f.split(".")[-1])
            if j == 0:
                ax.set_ylabel("$m=$%.1f" % d)

        ax = axes[i, -1]
        ax.tick_params(axis='both', top=False, bottom=False,
                       labeltop=False, labelbottom=False,
                       left=False, right=False,
                       labelleft=False, labelright=False)

        plt_img = img_dict[d]["Mass"]
        ax.imshow(plt_img, cmap="magma", norm=mimg_norm)

        if i == 0:
            ax.set_title("Mass")

    cmap = mpl.cm.magma
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                     norm=img_norm)
    cbar.ax.set_aspect('auto')
    pos = cbar.ax.get_position()
    cax2 = cbar.ax.twinx()
    cax2.set_ylim([mass_vmin, mass_vmax])
    pos.x0 += 0.05
    cbar.ax.set_position(pos)
    cax2.set_position(pos)
    cax2.set_yscale("log")
    # cbar2 = mpl.colorbar.ColorbarBase(cax2, cmap=cmap,
    #                                   norm=mimg_norm, alpha=0)

    cbar.set_label("$F/[\mathrm{nJy}]$")
    cax2.set_ylabel("$M/M_\odot$")

    fig.savefig("plots/gal_img_grid_Orientation-"
                + orientation + "_Type-" + Type
                + "_Region-" + reg + "_Snap-" + snap + "_Group-"
                + str(img_ind) + ".png", bbox_inches="tight")
    plt.close(fig)

    ind += 1

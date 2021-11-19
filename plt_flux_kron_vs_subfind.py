#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

os.environ['FLARE'] = '/cosma7/data/dp004/dc-wilk2/flare'

matplotlib.use('Agg')
warnings.filterwarnings('ignore')
import matplotlib.gridspec as gridspec
import seaborn as sns
from flare.photom import m_to_flux, flux_to_m
import h5py
import sys

sns.set_context("paper")
sns.set_style('whitegrid')

regions = []
for reg in range(0, 40):
    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))

# snaps = ['000_z015p000', '001_z014p000', '002_z013p000',
#          '003_z012p000', '004_z011p000', '005_z010p000',
#          '006_z009p000', '007_z008p000', '008_z007p000',
#          '009_z006p000', '010_z005p000', '011_z004p770']
snaps = ['009_z006p000', '010_z005p000', '011_z004p770']

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
filters = [f'Euclid.NISP.{f}' for f in ['Y', 'J', 'H']]

# Set up depths relative to the Xtreme deep field
depths_m = [23.24, 25.24, 24, 26]
depths_aperture = [2, 2, 2, 2]
depths_significance = [10, 10, 5, 5]
depths = [m_to_flux(d) for d in depths_m]

thresh = 2.5

for n_z in range(len(snaps)):

    if len(sys.argv) > 3:
        if n_z != int(sys.argv[3]):
            continue

    flux_segm_dict = {}
    flux_segmerr_dict = {}
    flux_subf_dict = {}
    lumin_segm_dict = {}

    for f in filters:

        snap = snaps[n_z]

        z_str = snap.split('z')[1].split('p')
        z = float(z_str[0] + '.' + z_str[1])

        for reg in regions:

            try:
                hdf = h5py.File("mock_data/flares_mock_cat_{}_{}_{}_{}.hdf5"
                                .format(reg, snap, Type, orientation), "r")
            except (OSError, KeyError) as e:
                print(reg, snap, e)
                continue

            try:
                f_group = hdf[f]
                fdepth_group = f_group["SUBFIND"]

                fluxes = fdepth_group["Fluxes"][:]

                hdf.close()
            except KeyError as e:
                print(reg, snap, e)
                hdf.close()
                continue

            flux_subf_dict.setdefault(f + "." + "SUBFIND", []).extend(fluxes)

            for depth in depths:

                hdf = h5py.File("mock_data/flares_mock_cat_{}_{}_{}_{}.hdf5"
                                .format(reg, snap, Type, orientation),
                                "r")

                try:
                    f_group = hdf[f]
                    fdepth_group = f_group[str(depth)]

                    flux_segm = fdepth_group['kron_flux'][...]
                    flux_segm_err = fdepth_group['kron_fluxerr'][...]

                except KeyError as e:
                    print(reg, snap, e)
                    hdf.close()
                    continue

                hdf.close()

                flux_segm_dict.setdefault(f + "." + str(depth), []).extend(
                    flux_segm)
                flux_segmerr_dict.setdefault(f + "." + str(depth), []).extend(
                    flux_segm_err)

        if f + "." + "SUBFIND" in flux_subf_dict.keys():
            flux_subfind = np.array(flux_subf_dict[f + "." + "SUBFIND"])
        else:
            flux_subfind = np.array([])

        if len(flux_subfind) == 0:
            continue

        fig = plt.figure()
        gs = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=(6, 2))
        gs.update(wspace=0.0, hspace=0.0)
        ax = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[1, 0])

        ax1.axhline(1, linestyle="--", color="k")

        bin_edges = np.logspace(np.log10(min(depths)),
                                4.5, 75)

        bin_wid = bin_edges[1] - bin_edges[0]
        bin_cents = bin_edges[:-1] + (bin_wid / 2)

        H, bins = np.histogram(flux_subfind, bins=bin_edges)
        n = np.sum(H)
        print("SUBFIND:", n)
        sub_H = H

        ax.bar(bin_edges[:-1], H, width=np.diff(bin_edges), color="grey",
               edgecolor="grey",
               label="SUBFIND ({})".format(n),
               alpha=0.6, align="edge")

        for depth in depths:

            fdepth = f + "." + str(depth)

            if not fdepth in flux_segm_dict.keys():
                continue

            H, bins = np.histogram(flux_segm_dict[fdepth], bins=bin_edges)

            n = np.sum(H)

            print(f"Kron ({depth}):", n)

            ax.plot(bin_edges[:-1], H,
                    label=r"$%.2f \times m_{\mathrm{XDF}}$ (%d)"
                          % ((depth / XDF_depth_flux), n))
            ax1.plot(bin_edges[:-1], H / sub_H,
                     label=r"$%.2f \times m_{\mathrm{XDF}}$ (%d)"
                          % ((depth / XDF_depth_flux), n))
        ax.tick_params(axis='x', top=False, bottom=False,
                       labeltop=False, labelbottom=False)

        ax1.set_xlabel("$F/[\mathrm{nJy}]$")
        ax.set_ylabel("$N$")
        ax1.set_ylabel("$N_\mathrm{Obs} / N_\mathrm{SUBFIND}$")

        ax.set_yscale("log")
        ax.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xscale("log")

        ax.set_xlim(min(depths), 10 ** 4.5)
        ax1.set_xlim(min(depths), 10**4.5)

        ax.legend()

        if not os.path.exists("plots/Flux_Kron"):
            os.makedirs("plots/Flux_Kron")

        fig.savefig(
            "plots/Flux_Kron/flux_kron_hist_Filter-" + f + "_Orientation-"
            + orientation + "_Type-" + Type + "_Snap-" + snap + ".png",
            bbox_inches="tight")

        plt.close(fig)

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

# # Define filter
# filters = [f'Hubble.ACS.{f}'
#            for f in ['f435w', 'f606w', 'f775w', 'f814w', 'f850lp']] \
#           + [f'Hubble.WFC3.{f}' for f in ['f105w', 'f125w', 'f140w', 'f160w']]
# Define filter
filters = [f'Hubble.WFC3.{f}' for f in ['f105w', 'f125w', 'f140w', 'f160w']]

# Set up depths relative to the Xtreme deep field
XDF_depth_m = 31.2
XDF_depth_flux = m_to_flux(XDF_depth_m)
depths = [XDF_depth_flux * 0.01, XDF_depth_flux * 0.1,
          XDF_depth_flux, 10 * XDF_depth_flux, 100 * XDF_depth_flux]
depths_m = [flux_to_m(d) for d in depths]

thresh = 2.5

for n_z in range(len(snaps)):

    if len(sys.argv) > 3:
        if n_z != int(sys.argv[3]):
            continue

    mass_kron_dict = {}
    flux_kronerr_dict = {}
    mass_subf_dict = {}
    lumin_kron_dict = {}

    for f in filters:

        snap = snaps[n_z]

        z_str = snap.split('z')[1].split('p')
        z = float(z_str[0] + '.' + z_str[1])

        for reg in regions:

            try:
                hdf = h5py.File("mock_data/flares_kron_{}_{}_{}_{}_{}.hdf5"
                                .format(reg, snap, Type, orientation, f), "r")
            except (OSError, KeyError) as e:
                print(e)
                continue

            try:

                masses = hdf["Galaxy_Mass"]

                hdf.close()
            except KeyError as e:
                print(e)
                hdf.close()
                continue

            mass_subf_dict.setdefault(f + "." + "SUBFIND", []).extend(masses)

            for depth in depths:

                hdf = h5py.File("mock_data/flares_mock_cat_{}_{}_{}_{}.hdf5"
                                .format(reg, snap, Type, orientation),
                                "r")

                try:
                    f_group = hdf[f]
                    fdepth_group = f_group[str(depth)]

                    flux_kron = fdepth_group['Mass_kron_flux'][...]
                    flux_kron_err = fdepth_group['Mass_kron_fluxerr'][...]

                except KeyError as e:
                    print(e)
                    hdf.close()
                    continue

                hdf.close()

                mass_kron_dict.setdefault(f + "." + str(depth), []).extend(
                    flux_kron)
                flux_kronerr_dict.setdefault(f + "." + str(depth), []).extend(
                    flux_kron_err)

        if f + "." + "SUBFIND" in mass_subf_dict.keys():
            flux_subfind = np.array(mass_subf_dict[f + "." + "SUBFIND"])
        else:
            flux_subfind = np.array([])

        if len(flux_subfind) == 0:
            continue

        fig = plt.figure()
        ax = fig.add_subplot(111)

        bin_edges = np.logspace(np.log10(7.5),
                                11.5, 75)

        bin_wid = bin_edges[1] - bin_edges[0]
        bin_cents = bin_edges[:-1] + (bin_wid / 2)

        H, bins = np.histogram(flux_subfind, bins=bin_edges)
        n = np.sum(H)
        print("SUBFIND:", n)

        ax.bar(bin_edges[:-1], H, width=np.diff(bin_edges), color="grey",
               edgecolor="grey",
               label="SUBFIND ({})".format(n),
               alpha=0.8, align="edge")

        for depth in depths:

            fdepth = f + "." + str(depth)

            if not fdepth in mass_kron_dict.keys():
                continue

            H, bins = np.histogram(mass_kron_dict[fdepth], bins=bin_edges)

            n = np.sum(H)

            print(f"Kron ({depth}):", n)

            ax.plot(bin_edges[:-1], H,
                    label="Kron: %.2f nJy (%d)"
                          % (depth, n))

        ax.set_xlabel("$F/[\mathrm{nJy}]$")
        ax.set_ylabel("$N$")

        ax.set_yscale("log")
        ax.set_xscale("log")

        ax.legend()

        if not os.path.exists("plots/Mass_Kron"):
            os.makedirs("plots/Mass_Kron")

        fig.savefig(
            "plots/Mass_Kron/Mass_kron_hist_Filter-" + f + "_Orientation-"
            + orientation + "_Type-" + Type + "_Snap-" + snap + ".png",
            bbox_inches="tight")

        plt.close(fig)

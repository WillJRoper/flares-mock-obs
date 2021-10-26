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
snaps = ['005_z010p000','006_z009p000', '007_z008p000', '008_z007p000',
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
filters = [f'Hubble.ACS.{f}'
           for f in ['f435w', 'f606w', 'f775w', 'f814w', 'f850lp']] \
          + [f'Hubble.WFC3.{f}' for f in ['f105w', 'f125w', 'f140w', 'f160w']]

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

    flux_segm_dict = {}
    flux_segmerr_dict = {}
    flux_subf_dict = {}
    lumin_segm_dict = {}

    for f in filters:

        snap = snaps[n_z]

        z_str = snap.split('z')[1].split('p')
        z = float(z_str[0] + '.' + z_str[1])

        for reg in regions:

            print("Getting SUBFIND occupancy with orientation {o}, type {t}, "
                  "and extinction {e} for region {x} and "
                  "snapshot {u} and filter  {i}".format(o=orientation, t=Type,
                                                        e=extinction,
                                                        x=reg, u=snap, i=f))
            try:
                hdf = h5py.File("mock_data/flares_mock_cat_{}_{}_{}_{}.hdf5"
                                .format(reg, snap, Type, orientation), "r")
                print(hdf.keys())
            except OSError as e:
                print(e)
                continue

            try:
                f_group = hdf[f]
                fdepth_group = f_group["SUBFIND"]

                fluxes = fdepth_group["Fluxes"][:]

                hdf.close()
            except KeyError as e:
                print(e)
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
                    print(e)
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
        print("SUBFIND:", flux_subfind.size)

        if len(flux_subfind) == 0:
            continue

        fig = plt.figure()
        ax = fig.add_subplot(111)

        bin_edges = np.logspace(np.log10(flux_subfind.min()) - 0.5,
                                np.log10(flux_subfind.max()) + 0.5, 75)

        bin_wid = bin_edges[1] - bin_edges[0]
        bin_cents = bin_edges[:-1] + (bin_wid / 2)

        H, bins = np.histogram(flux_subfind, bins=bin_edges)

        ax.bar(bin_edges[:-1], H, width=np.diff(bin_edges), color="grey",
               edgecolor="grey",
               label="SUBFIND ({})".format(len(flux_subfind)),
               alpha=0.8, align="edge")

        for depth in depths:

            fdepth = f + "." + str(depth)

            if not fdepth in flux_segm_dict.keys():
                continue

            print(f"Kron ({depth}):", len(flux_segm_dict[fdepth]))

            H, bins = np.histogram(flux_segm_dict[fdepth], bins=bin_edges)

            ax.plot(bin_edges[:-1], H,
                    label="Kron: {} nJy ({})"
                    .format(depth, len(flux_segm_dict[fdepth])))

        ax.set_xlabel("$F/[\mathrm{nJy}]$")
        ax.set_ylabel("$N$")

        ax.set_yscale("log")
        ax.set_xscale("log")

        ax.legend()

        if not os.path.exists("plots/Flux_Kron"):
            os.makedirs("plots/Flux_Kron")

        fig.savefig(
            "plots/Flux_Kron/flux_kron_hist_Filter-" + f + "_Orientation-"
            + orientation + "_Type-" + Type + "_Snap-" + snap + ".png",
            bbox_inches="tight")

        plt.close(fig)

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
import photutils as phut
from photutils.segmentation import SourceCatalog
import h5py
import sys
from astropy.cosmology import Planck13 as cosmo
import astropy.units as u

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
filters = [f'Hubble.ACS.{f}'
           for f in ['f435w', 'f606w', 'f775w', 'f814w', 'f850lp']] \
          + [f'Hubble.WFC3.{f}' for f in ['f105w', 'f125w', 'f140w', 'f160w']]

depths = [0.1, 1, 5, 10, 20]

thresh = 2.5

for n_z in range(len(snaps)):

    if len(sys.argv) > 3:
        if n_z != int(sys.argv[3]):
            continue

    flux_segm_dict = {}
    flux_kron_dict = {}
    flux_segmerr_dict = {}
    flux_kronerr_dict = {}

    for f in filters:

        snap = snaps[n_z]

        z_str = snap.split('z')[1].split('p')
        z = float(z_str[0] + '.' + z_str[1])

        for reg in regions:

            print("Getting SUBFIND occupancy with orientation {o}, type {t}, "
                  "and extinction {e} for region {x} and "
                  "snapshot {u}".format(o=orientation, t=Type, e=extinction,
                                        x=reg, u=snap))
            for depth in depths:

                try:
                    hdf = h5py.File("mock_data/flares_mock_cat_{}_{}_{}_{}.hdf5"
                                    .format(reg, snap, Type, orientation),
                                    "r")
                except OSError as e:
                    print(e)
                    continue

                try:
                    f_group = hdf[f]
                    fdepth_group = f_group[str(depth)]

                    flux_segm = fdepth_group['segment_flux'][...]
                    flux_segm_err = fdepth_group['segment_fluxerr'][...]
                    flux_kron = fdepth_group['kron_flux'][...]
                    flux_kron_err = fdepth_group['kron_fluxerr'][...]

                except KeyError as e:
                    hdf.close()
                    print(e)
                    continue

                hdf.close()

                flux_segm_dict.setdefault(f + "." + str(depth), []).extend(
                    flux_segm)
                flux_kron_dict.setdefault(f + "." + str(depth), []).extend(
                    flux_kron)
                flux_segmerr_dict.setdefault(f + "." + str(depth), []).extend(
                    flux_segm_err)
                flux_kronerr_dict.setdefault(f + "." + str(depth), []).extend(
                    flux_kron_err)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        for depth in depths:

            fdepth = f + "." + str(depth)

            if not fdepth in flux_segm_dict.keys():
                continue

            print(f"Segmentation ({depth}):", len(flux_segm_dict[fdepth]))
            print(f"Kron ({depth}):", len(flux_kron_dict[fdepth]))

            ax.scatter(flux_segm_dict[fdepth], flux_kron_dict[fdepth],
                        marker="^",
                        label="{} nJy ({})".format(depth,
                                                   len(flux_segm_dict[fdepth])))

        ax.set_xlabel("$F_{\mathrm{Segm}}/[\mathrm{nJy}]$")
        ax.set_ylabel("$F_{\mathrm{Kron}}/[\mathrm{nJy}]$")

        ax.set_yscale("log")
        ax.set_xscale("log")

        ax.legend()

        if not os.path.exists("plots/Flux_SegmKron"):
            os.makedirs("plots/Flux_SegmKron")

        fig.savefig("plots/Flux_SegmKron/flux_kronvssegm_Filter-" + f + "_Orientation-"
                + orientation + "_Type-" + Type + "_Snap-" + snap + ".png",
                    bbox_inches="tight")

        plt.close(fig)


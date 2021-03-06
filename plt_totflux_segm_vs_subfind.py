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
filters = ('Hubble.WFC3.f160w', )

depths = [0.1, 1, 5, 10, 20]

thresh = 2.5

for n_z in range(len(snaps)):

    if len(sys.argv) > 3:
        if n_z != int(sys.argv[3]):
            continue

    flux_segm_dict = {}
    lumin_segm_dict = {}

    for f in filters:

        snap = snaps[n_z]

        z_str = snap.split('z')[1].split('p')
        z = float(z_str[0] + '.' + z_str[1])

        flux_subfind = []

        for reg in regions:

            print("Getting SUBFIND occupancy with orientation {o}, type {t}, "
                  "and extinction {e} for region {x} and "
                  "snapshot {u}".format(o=orientation, t=Type, e=extinction,
                                        x=reg, u=snap))
            try:
                hdf = h5py.File("mock_data/flares_segm_{}_{}_{}_{}.hdf5"
                                .format(reg, snap, Type, orientation), "r")
            except OSError as e:
                print(e)
                continue

            try:
                f_group = hdf[f]

                fluxes = f_group["Fluxes"][:]
                subgrpids = f_group["Part_subgrpids"][:]
                begin = f_group["Start_Index"][:]
                group_len = f_group["Group_Length"][:]
                gal_ids = set(f_group["Subgroup_IDs"][:])

            except KeyError as e:
                hdf.close()
                continue

            for (ind, beg), img_len in zip(enumerate(begin), group_len):

                flux_subfind.append(np.sum(fluxes[beg: beg + img_len]))

            for depth in depths:

                fdepth_group = f_group[str(depth)]

                imgs = fdepth_group["Images"]
                sigs = fdepth_group["Significance_Images"]

                for img, sig in zip(imgs, sigs):

                    try:
                        segm = phut.detect_sources(sig, thresh, npixels=5)
                        segm = phut.deblend_sources(img, segm, npixels=5,
                                                    nlevels=32, contrast=0.001)
                    except TypeError as e:
                        flux_segm_dict.setdefault(f + "." + str(depth),
                                                  []).append(0)
                        continue

                    source_cat = SourceCatalog(img, segm, error=None,
                                               mask=None, kernel=None,
                                               background=None, wcs=None,
                                               localbkg_width=0,
                                               apermask_method='correct',
                                               kron_params=(2.5, 0.0),
                                               detection_cat=None)

                    flux_segm_dict.setdefault(f + "." + str(depth),
                                              []).append(np.sum(source_cat.segment_flux))

            hdf.close()

        flux_subfind = np.array(flux_subfind)
        print("SUBFIND:", flux_subfind.size)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        for depth in depths:

            fdepth = f + "." + str(depth)

            if not fdepth in flux_segm_dict.keys():
                continue

            flux_segm_dict[fdepth] = np.array(flux_segm_dict[fdepth])

            print(f"Segmentation ({depth}):", flux_segm_dict[fdepth].size)

            ax.scatter(flux_subfind, flux_segm_dict[fdepth], marker="^",
                    label="Segm: " + str(depth) + " (nJy)")

        ax.plot((np.min((ax.get_xlim()[0], ax.get_ylim()[0])),
                 np.max((ax.get_xlim()[1], ax.get_ylim()[1]))),
                (np.min((ax.get_xlim()[0], ax.get_ylim()[0])),
                 np.max((ax.get_xlim()[1], ax.get_ylim()[1]))),
                linestyle="--", color="k", alpha=0.8, zorder=0)

        ax.set_xlabel("$F_{\mathrm{SUBFIND}}/[\mathrm{nJy}]$")
        ax.set_ylabel("$F_{\mathrm{Segm}}/[\mathrm{nJy}]$")

        ax.set_yscale("log")
        ax.set_xscale("log")

        ax.legend()

        fig.savefig("plots/totflux_Filter-" + f + "_Orientation-"
                + orientation + "_Type-" + Type + "_Snap-" + snap + ".png",
                    bbox_inches="tight")

        plt.close(fig)


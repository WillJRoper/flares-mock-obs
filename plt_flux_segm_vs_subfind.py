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
f = 'Hubble.WFC3.f160w'

for n_z in range(len(snaps)):

    flux_segm = []
    flux_subfind = []

    for reg in regions:

        print("Getting SUBFIND occupancy with orientation {o}, type {t}, "
              "and extinction {e} for region {x} and "
              "snapshot {u}".format(o=orientation, t=Type, e=extinction,
                                    x=reg, u=snap))
        try:
            hdf = h5py.File("mock_data/"
                            "flares_segm_{}_{}_Webb.hdf5".format(reg, snap), "r")
        except OSError as e:
            print(e)
            continue

        try:
            type_group = hdf[Type]
            orientation_group = type_group[orientation]
            f_group = orientation_group[f]

            imgs = f_group["Images"]
            segms = f_group["Segmentation_Maps"]
            fluxes = f_group["Fluxes"]
            subgrpids = f_group["Part_subgrpids"]
            begin = f_group["Start_Index"]
            group_len = f_group["Group_Length"]

            print(segms.shape[0])

            for ind in range(segms.shape[0]):

                segm = segms[ind, :, :]
                img = imgs[ind, :, :]
                source_ids = np.unique(segm)
                source_ids = set(list(source_ids))

                while len(source_ids) > 0:

                    sid = source_ids.pop()

                    if sid == 0:
                        continue

                    flux_segm.append(np.sum(img[segm == sid]))

            for beg, img_len in zip(begin, group_len):

                subgrps = set(list(subgrpids[beg: beg + img_len]))

                while len(subgrps) > 0:

                    sid = subgrps.pop()

                    flux_subfind.append(np.sum(fluxes[beg: beg + img_len][subgrpids[beg: beg + img_len] == sid]))

            hdf.close()
        except KeyError as e:
            print(e)
            hdf.close()
            continue

    flux_segm = np.array(flux_segm)
    flux_subfind= np.array(flux_subfind)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    bin_edges = np.linspace(-2.5, 3.5, 50)

    H, bins = np.histogram(np.log10(flux_subfind), bins=bin_edges)
    bin_wid = bins[1] - bins[0]
    bin_cents = bins[1:] - (bin_wid / 2)

    ax.bar(bin_cents, H, width=bin_wid, color="b", edgecolor="b", label="SUBFIND")

    H, bins = np.histogram(np.log10(flux_segm), bins=bin_edges)
    bin_wid = bins[1] - bins[0]
    bin_cents = bins[1:] - (bin_wid / 2)

    ax.plot(bin_cents, H, color="r", linestyle="--", label="Segmentation map")

    ax.set_xlabel("$\log_{10}(F/[\mathrm{nJy}])$")
    ax.set_ylabel("$N$")

    ax.set_yscale("log")

    ax.legend()

    fig.savefig("plots/flux_hist_Snap-" + snaps[n_z] + "_Filter-" + f + ".png",
                bbox_inches="tight")

    plt.close(fig)












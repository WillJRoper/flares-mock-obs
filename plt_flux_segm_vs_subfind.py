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
f = 'Hubble.WFC3.f160w'

for n_z in range(len(snaps)):

    flux_segm = []
    flux_subfind = []

    snap = snaps[n_z]

    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

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

            imgs = f_group["Images"][:]
            segms = f_group["Segmentation_Maps"][:]
            fluxes = f_group["Fluxes"][:]
            subgrpids = f_group["Part_subgrpids"][:]
            begin = f_group["Start_Index"][:]
            group_len = f_group["Group_Length"][:]

            hdf.close()
        except KeyError as e:
            print(e)
            hdf.close()
            continue

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

            subgrps, inverse_inds = np.unique(subgrpids[beg: beg + img_len], return_inverse=True)

            this_flux = np.zeros(subgrps.size)

            for flux, i in zip(fluxes[beg: beg + img_len], inverse_inds):

                this_flux[i] += flux

            flux_subfind.extend(this_flux)

    flux_segm = np.array(flux_segm)
    flux_subfind = np.array(flux_subfind)

    lumin_segm = 4 * np.pi * cosmo.luminosity_distance(z)**2 * flux_segm * u.nJy
    lumin_subfind = 4 * np.pi * cosmo.luminosity_distance(z)**2 * flux_subfind * u.nJy

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if lumin_segm.size > 0 and lumin_subfind.size > 0:
        bin_edges = np.logspace(np.log10(np.min((lumin_segm.min(), lumin_subfind.min()))),
                                np.log10(np.max((lumin_segm.max(), lumin_subfind.max()))),
                                75)
    elif lumin_subfind.size == 0 and lumin_segm.size > 0:
        bin_edges = np.logspace(lumin_segm.min(),
                                lumin_segm.max(),
                                75)
    elif lumin_segm.size == 0 and lumin_subfind.size > 0:
        bin_edges = np.logspace(lumin_subfind.min(),
                                lumin_subfind.max(),
                                75)
    else: 
        continue

    interval = bin_edges[1:] - bin_edges[:-1]

    # Histogram the LF
    H_segm, bins = np.histogram(lumin_segm.value, bins=bin_edges)
    H_sf, _ = np.histogram(lumin_subfind.value, bins=bin_edges)

    # Compute bin centres
    bin_cents = bins[1:] - ((bins[1] - bins[0]) / 2)

    # Plot each histogram
    ax.loglog(bin_cents, np.log10(H_sf / interval), linestyle='--',
              label="SUBFIND")
    ax.loglog(bin_cents, np.log10(H_segm / interval), label="Segmentation map")

    ax.set_xlabel("$\log_{10}(L[\mathrm{erg} \mathrm{s}^{-1} \mathrm{Hz}^{-1}])$")
    ax.set_ylabel(r'$\mathrm{log}_{10}(\Phi/(\mathrm{cMpc}^{-3}\mathrm{dex}^{-1}))$')

    ax.legend()

    fig.savefig("plots/LF_Snap-" + snaps[n_z] + "_Filter-" + f + ".png",
                bbox_inches="tight")

    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if flux_segm.size > 0 and flux_subfind.size > 0:
        bin_edges = np.linspace(np.log10(np.min((flux_segm.min(), flux_subfind.min()))),
                                np.log10(np.max((flux_segm.max(), flux_subfind.max()))),
                                75)
    elif flux_subfind.size == 0 and flux_segm.size > 0:
        bin_edges = np.linspace(flux_segm.min(),
                                flux_segm.max(),
                                75)
    elif flux_segm.size == 0 and flux_subfind.size > 0:
        bin_edges = np.linspace(flux_subfind.min(),
                                flux_subfind.max(),
                                75)
    else: 
        continue

    H, bins = np.histogram(np.log10(flux_subfind), bins=bin_edges)
    bin_wid = bins[1] - bins[0]
    bin_cents = bins[1:] - (bin_wid / 2)

    ax.bar(bin_cents, H, width=bin_wid, color="b", edgecolor="b", label="SUBFIND")

    H, bins = np.histogram(np.log10(lumin_subfind.cgs.value), bins=bin_edges)
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












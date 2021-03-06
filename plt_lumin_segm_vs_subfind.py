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

flux_segm_dict = {}
lumin_segm_dict = {}

thresh = 2.0

for n_z in range(len(snaps)):

    if len(sys.argv) > 3:
        if n_z != int(sys.argv[3]):
            continue

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

                hdf.close()
            except KeyError as e:
                print(e)
                hdf.close()
                continue

            for beg, img_len in zip(begin, group_len):

                this_subgrpids = subgrpids[beg: beg + img_len]

                subgrps, inverse_inds = np.unique(this_subgrpids,
                                                  return_inverse=True)

                this_flux = np.zeros(subgrps.size)

                for flux, i, subgrpid in zip(fluxes[beg: beg + img_len],
                                             inverse_inds, this_subgrpids):

                    if subgrpid in gal_ids:

                        this_flux[i] += flux

                this_flux = this_flux[this_flux > 0]
                flux_subfind.extend(this_flux)

            for depth in depths:

                hdf = h5py.File("mock_data/flares_segm_{}_{}_{}_{}.hdf5"
                                .format(reg, snap, Type, orientation),
                                "r")

                try:
                    f_group = hdf[f]
                    fdepth_group = f_group[str(depth)]

                    imgs = fdepth_group["Images"]
                    sigs = fdepth_group["Significance_Images"]

                except KeyError as e:
                    print(e)
                    hdf.close()
                    continue

                flux_segm = []

                if sigs.max() < thresh:
                    continue

                for ind in range(imgs.shape[0]):

                    sig = sigs[ind, :, :]
                    img = imgs[ind, :, :]

                    if sig.max() < thresh:
                        continue

                    try:
                        segm = phut.detect_sources(sig, thresh, npixels=5)
                        segm = phut.deblend_sources(img, segm, npixels=5,
                                                    nlevels=32, contrast=0.001)
                    except TypeError:
                        continue

                    source_ids = np.unique(segm)
                    source_ids = set(list(source_ids))

                    while len(source_ids) > 0:

                        sid = source_ids.pop()

                        if sid == 0:
                            continue

                        flux_segm.append(np.sum(img[segm == sid]))

                hdf.close()

                flux_segm_dict[f + "." + str(depth)] = np.array(flux_segm)
                lumin_segm_dict[f + "." + str(depth)] = (4 * np.pi
                                                        * cosmo.luminosity_distance(z) ** 2
                                                        * flux_segm_dict[f + "." + str(depth)] * u.nJy).to(u.erg)

        flux_subfind = np.array(flux_subfind)
        print("SUBFIND:", flux_subfind.size)
        lumin_subfind = (4 * np.pi * cosmo.luminosity_distance(z)**2
                         * flux_subfind * u.nJy).to(u.erg).value

        fig = plt.figure()
        ax = fig.add_subplot(111)

        try:
            all_lumin_segm = np.concatenate(list(lumin_segm_dict.values())).value
        except ValueError:
            all_lumin_segm = np.array([])

        if all_lumin_segm.size > 0 and lumin_subfind.size > 0:
            bin_edges = np.linspace(
                np.log10(np.min((all_lumin_segm.min(), lumin_subfind.min()))),
                np.log10(np.max((all_lumin_segm.max(), lumin_subfind.max()))),
                75)
        elif lumin_subfind.size == 0 and all_lumin_segm.size > 0:
            bin_edges = np.linspace(np.log10(all_lumin_segm.min()),
                                    np.log10(all_lumin_segm.max()),
                                    75)
        elif all_lumin_segm.size == 0 and lumin_subfind.size > 0:
            bin_edges = np.linspace(np.log10(lumin_subfind.min()),
                                    np.log10(lumin_subfind.max()),
                                    75)
        else:
            continue

        interval = bin_edges[1:] - bin_edges[:-1]

        # Compute bin centres
        bin_cents = bin_edges[1:] - ((bin_edges[1] - bin_edges[0]) / 2)

        for depth in depths:

            fdepth = f + "." + str(depth)

            lumin_segm = np.log10(lumin_segm_dict[fdepth].value)

            # Histogram the LF
            H_segm, bins = np.histogram(lumin_segm, bins=bin_edges)

            # Plot each histogram
            ax.plot(bin_cents, H_segm / interval,
                    label="Segmentation map: " + str(depth) + " nJy", zorder=3)

        H_sf, _ = np.histogram(np.log10(lumin_subfind), bins=bin_edges)
        ax.loglog(bin_cents, H_sf / interval, linestyle='--',
                  label="SUBFIND", zorder=0)

        ax.set_xlabel("$\log_{10}(L[\mathrm{erg} \mathrm{s}^{-1} \mathrm{Hz}^{-1}])$")
        ax.set_ylabel(r'$\mathrm{log}_{10}(\Phi/(\mathrm{cMpc}^{-3}\mathrm{dex}^{-1}))$')

        ax.legend()

        fig.savefig("plots/LF_Type-" + Type + "_Snap-"
                    + snaps[n_z] + "_Filter-" + f + ".png",
                    bbox_inches="tight")

        plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        try:
            all_flux_segm = np.concatenate(list(flux_segm_dict.values()))
        except ValueError:
            all_flux_segm = np.array([])

        if all_flux_segm.size > 0 and flux_subfind.size > 0:
            bin_edges = np.linspace(
                np.log10(np.min((all_flux_segm.min(), flux_subfind.min()))),
                np.log10(np.max((all_flux_segm.max(), flux_subfind.max()))),
                50)
        elif flux_subfind.size == 0 and all_flux_segm.size > 0:
            bin_edges = np.linspace(np.log10(all_flux_segm.min()),
                                    np.log10(all_flux_segm.max()),
                                    50)
        elif all_flux_segm.size == 0 and flux_subfind.size > 0:
            bin_edges = np.linspace(np.log10(flux_subfind.min()),
                                    np.log10(flux_subfind.max()),
                                    50)
        else:
            continue

        H, bins = np.histogram(np.log10(flux_subfind), bins=bin_edges)
        bin_wid = bins[1] - bins[0]
        bin_cents = bins[1:] - (bin_wid / 2)

        # ax.bar(bin_cents, H, width=bin_wid, color="b", edgecolor="b", label="SUBFIND")

        for depth in depths:

            fdepth = f + "." + str(depth)

            print(f"Segmentation ({depth}):", flux_segm_dict[fdepth].size)

            H, bins = np.histogram(np.log10(flux_segm_dict[fdepth]), bins=bin_edges)
            bin_wid = bins[1] - bins[0]
            bin_cents = bins[1:] - (bin_wid / 2)

            ax.plot(bin_cents, H, label="Segmentation map: " + str(depth) + " nJy")

        ax.set_xlabel("$\log_{10}(F/[\mathrm{nJy}])$")
        ax.set_ylabel("$N$")

        ax.set_yscale("log")
        ax.set_xscale("log")

        ax.legend()

        fig.savefig("plots/flux_hist_Type-" + Type + "_Snap-"
                    + snaps[n_z] + "_Filter-" + f + ".png",
                    bbox_inches="tight")

        plt.close(fig)


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
fromm photutils.segmentation import SourceCatalog
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

                hdf.close()
            except KeyError as e:
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

                if sigs.shape[0] == 0:
                    continue

                if sigs[:].max() < thresh:
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
                    except TypeError as e:
                        continue

                    source_cat = SourceCatalog(img, segm, error=None, mask=None,  kernel=None, background=None, wcs=None, localbkg_width=0, apermask_method='correct', kron_params=(2.5, 0.0), detection_cat=None)

                    source_ids = np.unique(segm)
                    source_ids = set(list(source_ids))

                    print(source_cat.labels)
                    print(source_cat.kron_flux)

                    for sid in source_cat.labels:

                        flux_segm.append(source_cat.kron_flux[source_cat.labels == sid])

                hdf.close()

                flux_segm_dict[f + "." + str(depth)] = np.array(flux_segm)

        flux_subfind = np.array(flux_subfind)
        print("SUBFIND:", flux_subfind.size)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        try:
            all_flux_segm = np.concatenate(list(flux_segm_dict.values()))
        except ValueError:
            all_flux_segm = np.array([])

        if all_flux_segm.size > 0 and flux_subfind.size > 0:
            bin_edges = np.logspace(
                np.log10(np.min((all_flux_segm.min(), flux_subfind.min()))) - 0.5,
                np.log10(np.max((all_flux_segm.max(), flux_subfind.max()))) + 0.5,
                50)
        elif flux_subfind.size == 0 and all_flux_segm.size > 0:
            bin_edges = np.logspace(np.log10(all_flux_segm.min()) - 0.5,
                                    np.log10(all_flux_segm.max()) + 0.5,
                                    50)
        elif all_flux_segm.size == 0 and flux_subfind.size > 0:
            bin_edges = np.logspace(np.log10(flux_subfind.min()) - 0.5,
                                    np.log10(flux_subfind.max()) + 0.5,
                                    50)
        else:
            continue

        bin_wid = bin_edges[1] - bin_edges[0]
        bin_cents = bin_edges[:-1] + (bin_wid / 2)

        H, bins = np.histogram(flux_subfind, bins=bin_edges)

        ax.bar(bin_edges[:-1], H, width=np.diff(bin_edges), color="grey",
               edgecolor="grey", label="SUBFIND", alpha=0.8, align="edge")

        for depth in depths:

            fdepth = f + "." + str(depth)

            if not fdepth in flux_segm_dict.keys():
                continue

            print(f"Segmentation ({depth}):", flux_segm_dict[fdepth].size)

            H, bins = np.histogram(flux_segm_dict[fdepth], bins=bin_edges)

            ax.plot(bin_cents, H,
                    label="Segmentation map: " + str(depth) + " nJy")

        ax.set_xlabel("$F/[\mathrm{nJy}]$")
        ax.set_ylabel("$N$")

        ax.set_yscale("log")
        ax.set_xscale("log")

        ax.legend()

        fig.savefig("plots/flux_hist_Filter-" + f + "_Orientation-"
                + orientation + "_Type-" + Type + "_Snap-" + snap + ".png",
                    bbox_inches="tight")

        plt.close(fig)


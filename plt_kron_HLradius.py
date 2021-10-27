#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt

os.environ['FLARE'] = '/cosma7/data/dp004/dc-wilk2/flare'

matplotlib.use('Agg')
warnings.filterwarnings('ignore')
import seaborn as sns
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
import h5py
import sys
from astropy.cosmology import Planck13 as cosmo
import eritlux.simulations.imagesim as imagesim

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

    kron_radii_dict = {}
    kron_flux_dict = {}

    for f in filters:

        snap = snaps[n_z]

        z_str = snap.split('z')[1].split('p')
        z = float(z_str[0] + '.' + z_str[1])

        for reg in regions:

            for depth in depths:

                try:
                    hdf = h5py.File(
                        "mock_data/flares_mock_cat_{}_{}_{}_{}.hdf5"
                            .format(reg, snap, Type, orientation), "r")
                except OSError as e:
                    print(e)
                    continue

                try:
                    f_group = hdf[f]
                    fdepth_group = f_group[str(depth)]

                    kron_radii = fdepth_group["Kron_HLR"][...]
                    fluxes = fdepth_group['kron_flux'][...]

                except KeyError as e:
                    print(e)
                    hdf.close()
                    continue

                hdf.close()

                kron_radii_dict.setdefault(f + "." + str(depth), []).extend(
                    kron_radii)
                kron_flux_dict.setdefault(f + "." + str(depth), []).extend(
                    fluxes)

        fig = plt.figure(figsize=(4, 10))
        gs = gridspec.GridSpec(len(depths), 1)
        gs.update(wspace=0.0, hspace=0.0)
        axes = []
        for i in range(len(depths)):
            axes.append(fig.add_subplot(gs[i, 0]))

        for ax, depth in zip(axes, depths):

            fdepth = f + "." + str(depth)

            if not fdepth in kron_radii_dict.keys():
                continue

            try:
                cbar = ax.hexbin(kron_flux_dict[fdepth],
                                 kron_radii_dict[fdepth],
                                 gridsize=50, mincnt=1,
                                 xscale='log', yscale='log',
                                 norm=LogNorm(), linewidths=0.2,
                                 cmap='viridis')
            except ValueError as e:
                print(e)
                continue

            ax.text(0.95, 0.05, '$5\sigma=$%.2f nJy' % depth,
                    bbox=dict(boxstyle="round,pad=0.3", fc='w',
                              ec="k", lw=1, alpha=0.8),
                    transform=ax.transAxes, horizontalalignment='right',
                    fontsize=8)

        # Label axes
        axes[-1].set_xlabel(r'$F/$ [nJy]')
        for ax in axes:
            ax.set_ylabel('$R_{1/2}/ [pkpc]$')
            ax.set_ylim(10 ** -1.5, 10 ** 2)
            ax.set_xlim(10 ** -2, 10 ** 4)

        for ax in axes[:-1]:
            ax.tick_params(axis='x', top=False, bottom=False,
                           labeltop=False, labelbottom=False)

        axes[-1].tick_params(axis='x', which='minor', bottom=True)

        if not os.path.exists("plots/HLRs"):
            os.makedirs("plots/HLRs")

        fig.savefig(
            "plots/HLRs/HalfLightRadius_Filter-" + f + "_Orientation-"
            + orientation + "_Type-" + Type + "_Snap-" + snap + ".png",
            bbox_inches="tight")

        plt.close(fig)

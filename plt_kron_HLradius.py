#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import os
import numpy as np
import warnings

import matplotlib
import matplotlib.pyplot as plt

os.environ['FLARE'] = '/cosma7/data/dp004/dc-wilk2/flare'

matplotlib.use('Agg')
warnings.filterwarnings('ignore')
import seaborn as sns
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from flare.photom import m_to_flux, flux_to_m, flux_to_lum
from flare.photom import M_to_lum
import flare.photom as photconv
from astropy.cosmology import Planck13 as cosmo
import h5py
import sys

sns.set_context("paper")
sns.set_style('whitegrid')


def m_to_M(m, cosmo, z):
    flux = photconv.m_to_flux(m)
    lum = photconv.flux_to_L(flux, cosmo, z)
    M = photconv.lum_to_M(lum)
    return M


def M_to_m(M, cosmo, z):
    lum = photconv.M_to_lum(M)
    flux = photconv.lum_to_flux(lum, cosmo, z)
    m = photconv.flux_to_m(flux)
    return m

kawa_fit = lambda l, r0, b: r0 * (l / M_to_lum(-21)) ** b

kawa_params = {'beta': {6: 0.46, 7: 0.46, 8: 0.38, 9: 0.56},
               'r_0': {6: 0.94, 7: 0.94, 8: 0.81, 9: 1.2}}

regions = []
for reg in range(0, 40):
    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))

snaps = ['005_z010p000', '007_z008p000', '008_z007p000', '010_z005p000']

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
          XDF_depth_flux, 10 * XDF_depth_flux]
depths_m = [flux_to_m(d) for d in depths]

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
                cbar = ax.hexbin(flux_to_lum(np.array(kron_flux_dict[fdepth]),
                                             cosmo, z),
                                 kron_radii_dict[fdepth],
                                 gridsize=50, mincnt=1,
                                 xscale='log', yscale='log',
                                 norm=LogNorm(), linewidths=0.2,
                                 cmap='viridis', extent=(27, 31.5,
                                                         np.log10(0.09), 1.5))
            except ValueError as e:
                print(e)
                continue

            ax.text(0.95, 0.05, r"$%.2f \times m_{\mathrm{XDF}}$"
                    % (depth / XDF_depth_flux),
                    bbox=dict(boxstyle="round,pad=0.3", fc='w',
                              ec="k", lw=1, alpha=0.8),
                    transform=ax.transAxes, horizontalalignment='right',
                    fontsize=8)

        # Label axes
        axes[-1].set_xlabel(r"$L_{" + f.split(".")[-1]
                               + "}/$ [erg $/$ s $/$ Hz]")
        for ax in axes:
            ax.set_ylabel('$R_{1/2}/ [pkpc]$')
            ax.set_ylim(0.09, 10 ** 1.5)
            ax.set_xlim(10**26.8, 10**31.5)

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

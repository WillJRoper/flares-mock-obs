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
import matplotlib as mpl
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from scipy.stats import binned_statistic
from matplotlib.lines import Line2D
from astropy.cosmology import Planck13 as cosmo
import astropy.units as u
from FLARE.photom import lum_to_M, M_to_lum, lum_to_flux, m_to_flux
import FLARE.photom as photconv
from scipy.spatial import cKDTree
import h5py
import sys
import pandas as pd
import utilities as util
import phot_modules as phot
import utilities as util

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
f = ('JWST.NIRCAM.F150W', )

snr = 20

ngals_segm = []
ngals_subfind = []
grp_ms = []

for ind in range(len(reg_snaps)):

    reg, snap = reg_snaps[ind]

    print("Getting SUBFIND occupancy with orientation {o}, type {t}, "
          "and extinction {e} for region {x} and "
          "snapshot {u}".format(o=orientation, t=Type, e=extinction,
                                x=reg, u=snap))

    hdf = h5py.File("mock_data/"
                    "flares_segm_{}_{}_Webb.hdf5".format(reg, snap), "r")

    S_mass_ini, S_Z, S_age, G_Z, G_sml, S_sml, G_mass, S_coords, \
    G_coords, S_mass, grp_cops, r_200, all_grp_ms, S_subgrpid, \
    gal_cops, gal_ms, gal_grpid = phot.get_data(reg, snap, masslim=0)

    tree = cKDTree(gal_cops)

    type_group = hdf[Type]
    orientation_group = type_group[orientation]
    f_group = orientation_group[f]

    grp_ids = f_group[snr]["Group_ID"][:]
    segm_ngals = f_group[snr]["NGalaxy"][:]

    for grpind, ngal in zip(grp_ids, segm_ngals):

        okinds = tree.query_ball_point(grp_cops[grpind], r=0.5)

        ngals_segm.append(ngal)
        ngals_subfind.append(len(okinds))
        grp_ms.append(all_grp_ms[grpind])

ngals_segm = np.array(ngals_segm)
ngals_subfind= np.array(ngals_subfind)
grp_ms = np.array(grp_ms)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(ngals_segm, ngals_subfind, marker="+")

ax.set_ylabel("$N_{\mathrm{gal, SUBFIND}}$")
ax.set_ylabel("$N_{\mathrm{gal, Segmentation}}$")

fig.savefig("plots/ngal_subfindvssegm.png", bbox_inches="tight")

plt.close(fig)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(grp_ms, ngals_subfind / ngals_segm, marker="+")

ax.set_ylabel("$N_{\mathrm{gal, SUBFIND}} / N_{\mathrm{gal, Segmentation}}$")
ax.set_ylabel("$M_{FOF}/M_{\odot}$")

fig.savefig("plots/ngal_subfindsegm_ratio_vs_mass.png", bbox_inches="tight")

plt.close(fig)













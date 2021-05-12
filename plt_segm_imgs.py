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
f = 'JWST.NIRCAM.F150W'

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
        snr_group = f_group[str(snr)]

        imgs = snr_group["Images"][:]
        segms = snr_group["Segmentation_Maps"][:]
        subfind_spos = f_group["SUBFIND_NGalaxy"][:]

        hdf.close()
    except KeyError as e:
        print(e)
        hdf.close()
        continue
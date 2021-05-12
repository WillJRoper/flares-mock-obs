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
orientation = sys.argv[2]

# Define luminosity and dust model types
Type = sys.argv[3]
extinction = 'default'

# Define filter
f = 'JWST.NIRCAM.F150W'

snr = 20

ngals_segm = []
ngals_subfind = []
grp_ms = []

reg_ind = int(sys.argv[1])

reg, snap = reg_snaps[reg_ind]

z_str = snap.split('z')[1].split('p')
z = float(z_str[0] + '.' + z_str[1])

n_img = int(sys.argv[4])

print("Making images for with orientation {o}, type {t}, "
      "and extinction {e} for region {x} and "
      "snapshot {u}".format(o=orientation, t=Type, e=extinction,
                            x=reg, u=snap))

arcsec_per_kpc_proper = cosmo.arcsec_per_kpc_proper(z).value

# Define filter
filters = ('JWST.NIRCAM.F150W', )
f = filters[0]

# Define width
ini_width = 500 * arcsec_per_kpc_proper

# Define arc_second resolution
if int(filters[0].split(".")[-1][1:4]) < 230:
    arc_res = 0.031
else:
    arc_res = 0.063

# Compute the resolution
ini_res = ini_width / arc_res
res = int(np.ceil(ini_res))

# Compute the new width
width = arc_res * res

print("Image width and resolution (in arcseconds):", width, arc_res)
print("Image width and resolution (in pkpc):",
      width / arcsec_per_kpc_proper,
      arc_res / arcsec_per_kpc_proper)
print("Image width (in pixels):", res)

# Define pixel area in pkpc
single_pixel_area = arc_res * arc_res \
                    / (arcsec_per_kpc_proper * arcsec_per_kpc_proper)

# Define range and extent for the images in arc seconds
imgrange = ((-width / 2, width / 2), (-width / 2, width / 2))
imgextent = [-width / 2, width / 2, -width / 2, width / 2]

hdf = h5py.File("mock_data/"
                "flares_segm_{}_{}_Webb.hdf5".format(reg, snap), "r")

type_group = hdf[Type]
orientation_group = type_group[orientation]
f_group = orientation_group[f]
snr_group = f_group[str(snr)]

imgs = snr_group["Images"][:]
segms = snr_group["Segmentation_Maps"][:]
subfind_spos = f_group["Star_Pos"][:]
all_smls = f_group["Smoothing_Length"][:]
subgrpids = f_group["Part_subgrpids"][:]
begin = f_group["Start_Index"][:]
group_len = f_group["Group_Length"][:]

hdf.close()

# Define x and y positions of pixels
X, Y = np.meshgrid(np.linspace(imgrange[0][0], imgrange[0][1], res),
                   np.linspace(imgrange[1][0], imgrange[1][1], res))

# Define pixel position array for the KDTree
pix_pos = np.zeros((X.size, 2))
pix_pos[:, 0] = X.ravel()
pix_pos[:, 1] = Y.ravel()

# Build KDTree
tree = cKDTree(pix_pos)

print("Pixel tree built")

ind = 0
while ind < n_img:

    print("Creating image", ind)

    img = imgs[ind, :, :]
    segm = segms[ind, :, :]
    poss = subfind_spos[begin[ind]: begin[ind] + group_len[ind], (0, 1)]
    subgrp = subgrpids[begin[ind]: begin[ind] + group_len[ind]]
    smooth = all_smls[begin[ind]: begin[ind] + group_len[ind]]

    subfind_img = util.make_subfind_spline_img(poss, res, 0, 1, tree, subgrp,
                                               smooth, spline_cut_off=5/2)

    fig = plt.figure()
    ax1 = fig.add_subplot(321)
    ax2 = fig.add_subplot(322)
    ax3 = fig.add_subplot(323)
    ax4 = fig.add_subplot(324)
    ax5 = fig.add_subplot(325)
    ax6 = fig.add_subplot(326)
    axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    for ax in axes:
        ax.grid(False)
    plt_img = np.zeros_like(img)
    plt_img[img > 0] = np.log10(img[img > 0])
    axes[0].imshow(plt_img, extent=imgextent, cmap="Greys_r")
    axes[1].imshow(segm, extent=imgextent, cmap="plasma")
    axes[2].imshow(subfind_img, extent=imgextent, cmap="plasma")

    print(subfind_img.shape)

    max_ind = np.unravel_index(np.argmax(plt_img), plt_img.shape)
    axes[3].imshow(plt_img[max_ind[0] - 100: max_ind[0] + 100,
               max_ind[1] - 100: max_ind[1] + 100],
               extent=imgextent, cmap="Greys_r")
    axes[4].imshow(segm[max_ind[0] - 100: max_ind[0] + 100, max_ind[1] - 100: max_ind[1] + 100], extent=imgextent,
               cmap="plasma")
    axes[5].imshow(subfind_img[max_ind[0] - 100: max_ind[0] + 100,
               max_ind[1] - 100: max_ind[1] + 100], extent=imgextent,
               cmap="plasma")
    fig.savefig("plots/gal_img_log_" + f + "_" + str(ind) + ".png", dpi=300)
    plt.close(fig)

    ind += 1





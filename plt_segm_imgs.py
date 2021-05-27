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
from flare.photom import lum_to_M, M_to_lum, lum_to_flux, m_to_flux
import flare.photom as photconv
from scipy.spatial import cKDTree
import h5py
import sys
import pandas as pd
import utilities as util
import phot_modules as phot
import utilities as util
import eritlux.simulations.imagesim as imagesim
import flare.surveys
import flare.plots.image

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
f = 'Hubble.WFC3.f160w'

depth = int(sys.argv[4])

reg_ind = int(sys.argv[1])

reg, snap = reg_snaps[reg_ind]

z_str = snap.split('z')[1].split('p')
z = float(z_str[0] + '.' + z_str[1])

n_img = int(sys.argv[5])

print("Making images for with orientation {o}, type {t}, "
      "and extinction {e} for region {x} and "
      "snapshot {u}".format(o=orientation, t=Type, e=extinction,
                            x=reg, u=snap))

arcsec_per_kpc_proper = cosmo.arcsec_per_kpc_proper(z).value

# Define width
ini_width_pkpc = 100
ini_width = ini_width_pkpc * arcsec_per_kpc_proper

survey_id = 'XDF'  # the XDF (updated HUDF)
field_id = 'dXDF'  # deepest sub-region of XDF (defined by a mask)

# --- get field info object. This contains the filters, depths,
# image location etc. (if real image)
field = flare.surveys.surveys[survey_id].fields[field_id]

# --- initialise ImageCreator object
image_creator = imagesim.Idealised(f, field)

arc_res = image_creator.pixel_scale

# Compute the resolution
ini_res = ini_width / arc_res
res = int(np.ceil(ini_res))
cutout_halfsize = int(res * 0.1)

# Compute the new width
width = arc_res * res

print("Filter:", f)
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
fdepth_group = f_group[str(depth)]

imgs = fdepth_group["Images"][:]
segms = fdepth_group["Segmentation_Maps"][:]
sigs = fdepth_group["Significance_Images"][:]
subfind_spos = f_group["Star_Pos"][:]
all_smls = f_group["Smoothing_Length"][:]
subgrpids = f_group["Part_subgrpids"][:]
begin = f_group["Start_Index"][:]
group_len = f_group["Group_Length"][:]
gal_ids = set(f_group["Subgroup_IDs"][:])

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

print(subfind_spos)
print(begin)
print(group_len)

ind = 0
while ind < n_img and ind < imgs.shape[0]:

    print("Creating image", ind)

    img = imgs[ind, :, :]
    segm = segms[ind, :, :]
    poss = subfind_spos[begin[ind]: begin[ind] + group_len[ind], (0, 1)]
    subgrp = subgrpids[begin[ind]: begin[ind] + group_len[ind]]
    smooth = all_smls[begin[ind]: begin[ind] + group_len[ind]]

    subfind_img = util.make_subfind_spline_img(poss, res, 0, 1, tree, subgrp,
                                               smooth, gal_ids, spline_cut_off=5/2)
    subfind_img[segm == 0] = np.nan

    fig = plt.figure(figsize=(4, 6.4))
    gs = gridspec.GridSpec(3, 2)
    gs.update(wspace=0.0, hspace=0.0)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[0, 1])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[2, 1])

    axes = [ax1, ax2, ax3, ax4, ax5, ax6]

    for i, ax in enumerate(axes):
        ax.grid(False)

        if i < 2:
            ax.tick_params(axis='x', top=False, bottom=False,
                        labeltop=False, labelbottom=False)
        elif i > 2 and i < 5:
            ax.tick_params(axis='both', top=False, bottom=False,
                           labeltop=False, labelbottom=False,
                           left=False, right=False,
                           labelleft=False, labelright=False)
        elif i == 5:
            ax.tick_params(axis='y', left=False, right=False,
                           labelleft=False, labelright=False)

    plt_img = np.zeros_like(img)
    plt_img[img > 0] = np.log10(img[img > 0])
    axes[0].imshow(plt_img, extent=imgextent, cmap="Greys_r")
    axes[1].imshow(segm, extent=imgextent, cmap="plasma")
    axes[2].imshow(subfind_img, extent=imgextent, cmap="gist_rainbow")

    max_ind = np.unravel_index(np.argmax(plt_img), plt_img.shape)
    ind_slice = [np.max((0, max_ind[0] - cutout_halfsize)),
                 np.min((plt_img.size, max_ind[0] + cutout_halfsize)),
                 np.max((0, max_ind[1] - cutout_halfsize)),
                 np.min((plt_img.size, max_ind[1] + cutout_halfsize))]
    axes[3].imshow(plt_img[ind_slice[0]: ind_slice[1],
                   ind_slice[2]: ind_slice[3]],
                   extent=imgextent, cmap="Greys_r")
    axes[4].imshow(segm[ind_slice[0]: ind_slice[1],
                   ind_slice[2]: ind_slice[3]], extent=imgextent,
                   cmap="plasma")
    axes[5].imshow(subfind_img[ind_slice[0]: ind_slice[1],
                   ind_slice[2]: ind_slice[3]], extent=imgextent,
                   cmap="gist_rainbow")

    ax1.set_title(str(ini_width_pkpc) + " pkpc")
    ax4.set_title("Brightest Source")

    ax1.set_ylabel('y (")')
    ax2.set_ylabel('y (")')
    ax3.set_ylabel('y (")')
    ax3.set_xlabel('x (")')
    ax6.set_xlabel('x (")')

    fig.savefig("plots/gal_img_log_Filter-" + f + "_Depth-" + str(depth)
                + "_Region-" + reg + "_Snap-" + snap + "_Group-"
                + str(ind) + ".png", dpi=600)
    plt.close(fig)

    ind += 1





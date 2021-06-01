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

depths = [0.1, 1, 5, 10, 20]

reg_ind = int(sys.argv[1])

reg, snap = reg_snaps[reg_ind]

z_str = snap.split('z')[1].split('p')
z = float(z_str[0] + '.' + z_str[1])

n_img = int(sys.argv[4])

arcsec_per_kpc_proper = cosmo.arcsec_per_kpc_proper(z).value

# Define width
ini_width_pkpc = 150
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

ind = 0

while ind < n_img:

    print("Creating image", ind)

    img_dict = {}

    for depth in depths:

        hdf = h5py.File("mock_data/flares_segm_{}_{}_{}_{}.hdf5"
                        .format(reg, snap, Type, orientation),
                        "r")

        f_group = hdf[f]
        fdepth_group = f_group[str(depth)]

        imgs = fdepth_group["Images"]

        if ind > imgs.shape[0]:
            ind += 1
            continue

        img_dict[depth] = imgs[ind, :, :]
        print(img_dict[depth].min(), img_dict[depth].max(),
              np.std(img_dict[depth]))

        hdf.close()

    all_imgs = np.array(list(img_dict.values()))
    img_norm = LogNorm(vmin=0, vmax=np.percentile(all_imgs, 99))
    resi_norm = LogNorm(vmin=-5, vmax=5)
    
    print(np.percentile(all_imgs, 99), np.std(all_imgs))

    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(len(depths), len(depths))
    gs.update(wspace=0.0, hspace=0.0)
    axes = []
    combos = []
    ijs = []
    for i in range(len(depths)):
        for j in range(len(depths)):
            if j > i:
                continue
            axes.append(fig.add_subplot(gs[i, j]))
            combos.append((depths[i], depths[j]))
            ijs.append((i, j))

    for combo, ax, (i, j) in zip(combos, axes, ijs):
        ax.tick_params(axis='both', top=False, bottom=False,
                       labeltop=False, labelbottom=False,
                       left=False, right=False,
                       labelleft=False, labelright=False)

        if combo[0] == combo[1]:

            plt_img = img_dict[combo[0]]
            ax.imshow(plt_img, extent=imgextent, cmap="Greys_r", norm=img_norm)

        else:
            img1 = img_dict[combo[0]]
            img2 = img_dict[combo[1]]
            var1 = np.var(img1)
            var2 = np.var(img2)
            plt_img = (img1 - img2) / np.sqrt(var1 + var2)
            ax.imshow(plt_img, extent=imgextent, cmap="plasma", norm=resi_norm)

        if i == 0:
            ax.set_ylabel("Depth {} nJY".format(combo[i]))
        if j == 0:
            ax.set_xlabel("Depth {} nJY".format(combo[j]))

    fig.savefig("plots/gal_img_comp_Filter-" + f
                + "_Region-" + reg + "_Snap-" + snap + "_Group-"
                + str(ind) + ".png", dpi=600)
    plt.close(fig)

    ind += 1





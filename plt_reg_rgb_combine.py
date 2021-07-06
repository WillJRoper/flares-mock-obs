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
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from astropy.cosmology import Planck13 as cosmo
import h5py
import sys
import eritlux.simulations.imagesim as imagesim
import flare.surveys
import flare.plots.image
import utilities as util

sns.set_context("paper")
sns.set_style('white')


regions = []
for reg in range(0, 40):
    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))

snaps = ['004_z011p000', '005_z010p000',
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
filters = [f'Hubble.WFC3.{f}'
           for f in ['f105w', 'f125w', 'f140w', 'f160w']] + \
          [f'Hubble.ACS.{f}'
           for f in ['f435w', 'f606w', 'f775w', 'f814w', 'f850lp']]

rgb_filters= {"R": (filters[0], filters[1], filters[2], filters[3]),
              "G": (filters[6], filters[8]),
              "B": (filters[4], filters[5])}

depths = [0.1, 1, 5, 10, 20]

depth = depths[int(sys.argv[3])]

for snap in snaps:

    f = filters[0]

    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    arcsec_per_kpc_proper = cosmo.arcsec_per_kpc_proper(z).value

    # Define width
    ini_width = 160
    ini_width_pkpc = ini_width / arcsec_per_kpc_proper

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

    # Define pixel area in pkpc
    single_pixel_area = arc_res * arc_res \
                        / (arcsec_per_kpc_proper * arcsec_per_kpc_proper)

    # Define range and extent for the images in arc seconds
    imgrange = ((-width / 2, width / 2), (-width / 2, width / 2))
    imgextent = [-width / 2, width / 2, -width / 2, width / 2]

    hdf = h5py.File("mock_data/flares_segm_{}_{}_{}_{}_{}.hdf5"
                    .format("00", snap, Type, orientation, f), "r")

    ijk = hdf["Cell_Image_Number"][:]

    hdf.close()

    rgb_img = np.zeros((ijk.shape[0] * res * 8,
                        ijk.shape[1] * res * 8,
                        3), dtype=np.float32)

    for reg in regions:

        try:
            img = np.load("mock_data/rgb_region_wrapped_"
                          + "Orientation-" + orientation
                          + "_Type-" + Type
                          + "_Depth-" + str(depth)
                          + "_Region-" + reg
                          + "_Snap-" + snap + ".npy")
        except OSError:
            continue

        left = np.random.choice(np.arange(0, rgb_img.shape[0] - img.shape[0]))
        top = np.random.choice(np.arange(0, rgb_img.shape[1] - img.shape[1]))
        print(reg, left, top)
        rgb_img[left: left + img.shape[0], top: top + img.shape[1], :] += img

    plt_img = np.zeros(rgb_img.shape)

    dpi = rgb_img.shape[0]
    fig = plt.figure(figsize=(1, 1), dpi=dpi)
    ax = fig.add_subplot(111)

    ax.tick_params(axis='x', top=False, bottom=False,
                   labeltop=False, labelbottom=False)
    ax.tick_params(axis='y', left=False, right=False,
                   labelleft=False, labelright=False)

    for i in range(3):
        norm = mpl.colors.Normalize(vmin=0,
                                    vmax=np.percentile(rgb_img[:, :, i],
                                                       99),
                                    clip=True)

        plt_img[:, :, i] = norm(rgb_img[:, :, i])

    ax.imshow(plt_img)

    fig.savefig("plots/Region_slices/rgb_region_img_wrapped_"
                + "Orientation-" + orientation
                + "_Type-" + Type
                + "_Depth-" + str(depth)
                + "_Snap-" + snap + ".png",
                bbox_inches="tight")
    plt.close(fig)

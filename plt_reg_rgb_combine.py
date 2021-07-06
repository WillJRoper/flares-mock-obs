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
    for reg in regions:

        img = np.load("mock_data/rgb_region_wrapped_"
                      + "Orientation-" + orientation
                      + "_Type-" + Type
                      + "_Depth-" + str(depth)
                      + "_Region-" + reg
                      + "_Snap-" + snap + ".npy")
        if reg == regions[0]:
            rgb_img = np.zeros_like(img)
        rgb_img += img

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
                + "_Region-" + reg
                + "_Snap-" + snap + ".png",
                bbox_inches="tight")
    plt.close(fig)





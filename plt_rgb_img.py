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

snaps = ['009_z006p000', '010_z005p000', '011_z004p770']

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

rgb_filters= {"R": (filters[0], filters[1] , filters[2], filters[3]),
              "G": (filters[6], filters[8]),
              "B": (filters[4], filters[5])}

depths = [0.1, 1, 5, 10, 20]

depth = depths[int(sys.argv[3])]

for reg in regions:
    for snap in snaps:

        f = filters[0]

        z_str = snap.split('z')[1].split('p')
        z = float(z_str[0] + '.' + z_str[1])

        arcsec_per_kpc_proper = cosmo.arcsec_per_kpc_proper(z).value

        # Define width
        ini_width_pkpc = 500
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

        # Define pixel area in pkpc
        single_pixel_area = arc_res * arc_res \
                            / (arcsec_per_kpc_proper * arcsec_per_kpc_proper)

        # Define range and extent for the images in arc seconds
        imgrange = ((-width / 2, width / 2), (-width / 2, width / 2))
        imgextent = [-width / 2, width / 2, -width / 2, width / 2]

        hdf = h5py.File("mock_data/flares_segm_{}_{}_{}_{}_{}.hdf5"
                        .format(reg, snap, Type, orientation, f), "r")

        fdepth_group = hdf[str(depth)]

        img_ids = fdepth_group["Image_ID"][...]

        hdf.close()

        for img_ind in img_ids:

            rgb_img = np.zeros((res,
                                res,
                                3), dtype=np.float32)
            rgb_wht = np.zeros((res,
                                res,
                                3), dtype=np.float32)

            for ind, key in enumerate(rgb_filters.keys()):

                for f in rgb_filters[key]:

                    print(f, reg, snap, img_ind)

                    hdf = h5py.File("mock_data/flares_segm_{}_{}_{}_{}_{}.hdf5"
                                    .format(reg, snap, Type, orientation, f),
                                    "r")
                    try:
                        fdepth_group = hdf[str(depth)]

                        img = fdepth_group["Images"][img_ind, :, :]
                        noise = fdepth_group["Noise_value"][img_ind]

                    except KeyError as e:
                        print(e)
                        hdf.close()
                        continue

                    rgb_img[:, :, ind] += img * (1 / noise**2)
                    rgb_wht[:, :, ind] += (1 / noise**2)

                    hdf.close()

            rgb_img /= rgb_wht
            print(rgb_img.max(), rgb_img.min(), np.percentile(rgb_img, 99))

            plt_img = np.zeros(rgb_img.shape)
            # plt_img[rgb_img > 0] = np.log10(rgb_img[rgb_img > 0])
            # plt_img[rgb_img <= 0] = np.nan

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

            if not os.path.exists("plots/RGB_Images"):
                os.makedirs("plots/RGB_Images")
            if not os.path.exists("plots/RGB_Images/{}".format(reg)):
                os.makedirs("plots/RGB_Images/{}".format(reg))

            fig.savefig("plots/RGB_Images/" + reg
                        + "/rgb_img_Orientation-" + orientation
                        + "_Type-" + Type
                        + "_Region-" + reg
                        + "_Snap-" + snap
                        + "_Img-" + str(img_ind) + ".png",
                        bbox_inches="tight")
            plt.close(fig)





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
f = 'Hubble.WFC3.f160w'

depths = [0.1, 1, 5, 10, 20]

depth = depths[int(sys.argv[3])]

for reg in regions:
    for snap in snaps:

        print(reg, snap)

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

        ind = 0

        r = width / arcsec_per_kpc_proper / 1000

        path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data'

        cen, radius, _ = util.spherical_region(path, snap)

        mins = cen - radius
        maxs = cen + radius
        reg_width = 2 * radius
        print(int(reg_width / r), "images along each axis,",
              int(reg_width / r) ** 3, "total, with a width of", reg_width)
        xcents = np.linspace(mins[0] + r, maxs[0] - r, int(reg_width / r))
        ycents = np.linspace(mins[1] + r, maxs[1] - r, int(reg_width / r))
        zcents = np.linspace(mins[2] + r, maxs[2] - r, int(reg_width / r))

        cents = []
        ijk = np.zeros((int(reg_width / r), int(reg_width / r), int(reg_width / r)))
        num = 0
        for i, x in enumerate(xcents):
            for j, y in enumerate(ycents):
                for k, z in enumerate(zcents):
                    cents.append(np.array([x, y, z]))
                    ijk[i, j, k] = num
                    num += 1

        cents = np.array(cents)

        kth = int(reg_width / r) // 2

        print("Filter:", f)
        print("Image width and resolution (in arcseconds):", width * xcents.size, arc_res)
        print("Image width and resolution (in pkpc):",
              width / arcsec_per_kpc_proper * xcents.size,
              arc_res / arcsec_per_kpc_proper)
        print("Image width (in pixels):", res * xcents.size)

        reg_img = np.zeros((int(reg_width / r) * res,
                            int(reg_width / r) * res), dtype=np.float32)

        hdf = h5py.File("mock_data/flares_segm_{}_{}_{}_{}_{}.hdf5"
                        .format(reg, snap, Type, orientation, f), "r")

        try:
            fdepth_group = hdf[str(depth)]

            imgs = fdepth_group["Images"]
            img_ids = fdepth_group["Image_ID"][...]

        except KeyError as e:
            print(e)
            hdf.close()
            continue

        for i in range(xcents.size):
            for j in range(ycents.size):
                img_id = ijk[i, j, kth]
                if int(img_id) in img_ids:
                    print(i, j, kth, img_id, np.sum(imgs[img_ids == img_id, :, :]))
                    reg_img[i * res: (i + 1) * res, j * res: (j + 1) * res,] = imgs[img_ids == img_id, :, :]
                else:
                    noise = image_creator.pixel.noise * np.random.randn(res,
                                                                        res)
                    reg_img[i * res: (i + 1) * res, j * res: (j + 1) * res, ] = noise

        hdf.close()

        plt_img = np.zeros(reg_img.shape)
        plt_img[reg_img > 0] = np.log10(reg_img[reg_img > 0])
        plt_img[reg_img <= 0] = np.nan

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        ax.imshow(plt_img, cmap="plasma",
                  vmin=np.percentile(plt_img, 33.175),
                  vmax=np.percentile(plt_img, 99))

        fig.savefig("plots/region_img_Filter-" + f + "_Orientation-"
                    + orientation + "_Type-" + Type
                    + "_Region-" + reg + "_Snap-" + snap + ".png",
                    dpi=3000, bbox_inches="tight")
        plt.close(fig)





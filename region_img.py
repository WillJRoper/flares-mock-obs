#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from photutils import CircularAperture

os.environ['FLARE'] = '/cosma7/data/dp004/dc-wilk2/flare'

matplotlib.use('Agg')
warnings.filterwarnings('ignore')
import seaborn as sns
from matplotlib.colors import LogNorm
import phot_modules_whole_region as phot
import utilities as util
import flare
from flare.photom import lum_to_M, M_to_lum
from astropy.cosmology import Planck13 as cosmo
import h5py
import photutils as phut
from astropy.convolution import Gaussian2DKernel
from scipy import signal
import sys
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
import time
import eritlux.simulations.imagesim as imagesim
import flare.surveys
import flare.plots.image
import mpi4py
from mpi4py import MPI

mpi4py.rc.recv_mprobe = False

# Initializations and preliminaries
comm = MPI.COMM_WORLD  # get MPI communicator object
size = comm.size  # total number of processes
rank = comm.rank  # rank of this process

sns.set_context("paper")
sns.set_style('whitegrid')

regions = []
for reg in range(10, -1, -1):
    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))

# snaps = ['000_z015p000', '001_z014p000', '002_z013p000',
#          '003_z012p000', '004_z011p000', '005_z010p000',
#          '006_z009p000', '007_z008p000', '008_z007p000',
#          '009_z006p000', '010_z005p000', '011_z004p770']

snaps = ['010_z005p000', ]

# Set orientation
orientation = sys.argv[1]

# Define flux and dust model types
Type = sys.argv[2]
extinction = 'default'

# Define filter
filters = ('Hubble.WFC3.f160w',)

survey_id = 'XDF'  # the XDF (updated HUDF)
field_id = 'dXDF'  # deepest sub-region of XDF (defined by a mask)

# --- get field info object. This contains the filters, depths,
# image location etc. (if real image)
field = flare.surveys.surveys[survey_id].fields[field_id]

print("Rank", rank, "of", size)

for tag in snaps:

    print(tag)

    for reg in regions:

        if rank == 0:
            print(
                "Computing HLRs with orientation {o}, type {t}, and extinction {e}"
                " for region {x} and snapshot {u}".format(o=orientation, t=Type,
                                                          e=extinction, x=reg,
                                                          u=tag))

        z_str = tag.split('z')[1].split('p')
        z = float(z_str[0] + '.' + z_str[1])

        if z <= 2.8:
            csoft = 0.000474390 / 0.6777 * 1e3
        else:
            csoft = 0.001802390 / (0.6777 * (1 + z)) * 1e3

        arcsec_per_kpc_proper = cosmo.arcsec_per_kpc_proper(z).value

        # Define width
        ini_width_pkpc = 2 * 1000
        ini_width = ini_width_pkpc * arcsec_per_kpc_proper

        for f in filters:

            # --- initialise ImageCreator object
            image_creator = imagesim.Idealised(f, field)

            arc_res = image_creator.pixel_scale

            # Compute the resolution
            ini_res = ini_width / arc_res
            res = int(np.ceil(ini_res))

            # Compute the new width
            width = arc_res * res

            if rank == 0:
                print("Filter:", f)
                print("Image width and resolution (in arcseconds):",
                      width, arc_res)
                print("Image width and resolution (in pkpc):",
                      width / arcsec_per_kpc_proper,
                      arc_res / arcsec_per_kpc_proper)
                print("Image width (in pixels):", res)

            # Kappa with DTM 0.0795, BC_fac=1., without 0.0063 BC_fac=1.25
            try:
                reg_dict = phot.flux(reg, kappa=0.0795, tag=tag, BC_fac=1,
                                     IMF='Chabrier_300',
                                     filters=(f, ), Type=Type, log10t_BC=7.,
                                     extinction=extinction,
                                     orientation=orientation,
                                     r=width / arcsec_per_kpc_proper / 1000 / 2)
            except KeyError:
                continue
            except ValueError:
                continue
            except OSError:
                continue

            print("Got the dictionary for the region")

            # Define pixel area in pkpc
            single_pixel_area = arc_res * arc_res \
                                / (arcsec_per_kpc_proper * arcsec_per_kpc_proper)

            # Define range and extent for the images in arc seconds
            imgrange = ((-width / 2, width / 2), (-width / 2, width / 2))
            imgextent = [-width / 2, width / 2, -width / 2, width / 2]

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

            # Create rank bins for poss
            rank_inds = np.linspace(0, reg_dict["coords"].shape[0], size + 1)

            psf = imagesim.create_PSF(f, field, res)

            print("Creating image for Filter, Pixel noise, Depth:",
                  f, image_creator.pixel.noise, field.depths[f])

            beg, end = rank_inds[rank], rank_inds[rank + 1]

            this_pos = reg_dict["coords"][beg: end]
            this_pos *= 10 ** 3 * arcsec_per_kpc_proper
            this_smls = reg_dict["smls"][beg: end]
            this_smls *= 10 ** 3 * arcsec_per_kpc_proper

            this_flux = reg_dict[f][beg: end]

            if np.nansum(this_flux) != 0:

                this_radii = util.calc_rad(this_pos, i=0, j=1)

                img = util.make_spline_img(this_pos, res, 0, 1, tree,
                                           this_flux, this_smls)

                if Type != "Intrinsic":
                    img = signal.fftconvolve(img, psf, mode="same")
            else:
                img = np.zeros((res, res))

            if comm.rank == 0:
                full_image = np.zeros_like(img)
            else:
                full_image = None

            # use MPI to get the totals
            comm.Reduce(
                [img, MPI.DOUBLE],
                [full_image, MPI.DOUBLE],
                op=MPI.SUM,
                root=0
            )

            if rank == 0:

                img, img_obj = util.noisy_img(full_image, image_creator)

                significance_image = img / img_obj.noise
                significance_image[significance_image < 0] = 0

                try:
                    segm = phut.detect_sources(significance_image, 2.5,
                                               npixels=5)
                    segm = phut.deblend_sources(img, segm, npixels=5,
                                                nlevels=32, contrast=0.001)

                    fig = plt.figure()
                    ax = fig.add_subplot(111)

                    ax.imshow(img,
                              norm=LogNorm(vmin=-np.percentile(img, 31.75),
                                           vmax=np.percentile(img, 99)))

                    ax.grid(False)

                    fig.savefig(
                        "plots/region_img_log_Filter-" + f
                        + "_Region-" + reg + "_Snap-"
                        + tag + ".png", dpi=600)

                    util.plot_images(img, segm.data, significance_image, reg,
                                     f, "XDF", tag, reg, imgextent,
                                     ini_width_pkpc,
                                     cutout_halfsize=int(0.1 * res))

                except TypeError as e:
                    print(e)
                    print(img.min(), img.max(),
                          significance_image.min(), significance_image.max())
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
import photutils as phut
from photutils.segmentation import SourceCatalog
import h5py
import sys
from astropy.cosmology import Planck13 as cosmo
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
orientation = sys.argv[1]

# Define luminosity and dust model types
Type = sys.argv[2]
extinction = 'default'

# Define filter
filters = ('Hubble.WFC3.f160w', )

depths = [0.1, 1, 5, 10, 20]

thresh = 2.5

survey_id = 'XDF'  # the XDF (updated HUDF)
field_id = 'dXDF'  # deepest sub-region of XDF (defined by a mask)

# --- get field info object. This contains the filters, depths,
# image location etc. (if real image)
field = flare.surveys.surveys[survey_id].fields[field_id]

for n_z in range(len(snaps)):

    if len(sys.argv) > 3:
        if n_z != int(sys.argv[3]):
            continue

    kron_radii_dict = {}

    for f in filters:

        snap = snaps[n_z]

        z_str = snap.split('z')[1].split('p')
        z = float(z_str[0] + '.' + z_str[1])

        arcsec_per_kpc_proper = cosmo.arcsec_per_kpc_proper(z).value

        # --- initialise ImageCreator object
        image_creator = imagesim.Idealised(f, field)

        arc_res = image_creator.pixel_scale
        kpc_res = arc_res / arcsec_per_kpc_proper

        for reg in regions:

            print("Getting SUBFIND occupancy with orientation {o}, type {t}, "
                  "and extinction {e} for region {x} and "
                  "snapshot {u}".format(o=orientation, t=Type, e=extinction,
                                        x=reg, u=snap))

            for depth in depths:

                try:
                    hdf = h5py.File("mock_data/flares_segm_{}_{}_{}_{}.hdf5"
                                    .format(reg, snap, Type, orientation), "r")
                except OSError as e:
                    print(e)
                    continue

                try:
                    f_group = hdf[f]
                    fdepth_group = f_group[str(depth)]

                    imgs = fdepth_group["Images"]
                    sigs = fdepth_group["Significance_Images"]

                except KeyError as e:
                    print(e)
                    hdf.close()
                    continue

                kron_radii = []

                if sigs.shape[0] == 0:
                    continue

                if sigs[:].max() < thresh:
                    continue

                for ind in range(imgs.shape[0]):

                    sig = sigs[ind, :, :]
                    img = imgs[ind, :, :]

                    if sig.max() < thresh:
                        continue

                    try:
                        segm = phut.detect_sources(sig, thresh, npixels=5)
                        segm = phut.deblend_sources(img, segm, npixels=5,
                                                    nlevels=32, contrast=0.001)
                    except TypeError as e:
                        continue

                    source_cat = SourceCatalog(img, segm, error=None, mask=None,  kernel=None, background=None, wcs=None, localbkg_width=0, apermask_method='correct', kron_params=(2.5, 0.0), detection_cat=None)

                    kron_radii.extend(source_cat.kron_flux * kpc_res)

                hdf.close()

                kron_radii_dict.setdefault(f + "." + str(depth), []).extend(kron_radii)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        # bin_edges = np.logspace(-3, 4, 75)

        for depth in depths:

            fdepth = f + "." + str(depth)

            if not fdepth in kron_radii_dict.keys():
                continue

            print(f"Kron ({depth}):", len(kron_radii_dict[fdepth]))

            H, bin_edges = np.histogram(kron_radii_dict[fdepth], bins=100)

            bin_wid = bin_edges[1] - bin_edges[0]
            bin_cents = bin_edges[:-1] + (bin_wid / 2)

            ax.plot(bin_cents, H,
                    label="Kron: {} nJy ({})"
                    .format(depth, len(kron_radii_dict[fdepth])))

        ax.set_xlabel("$R_{\mathrm{Kron}}/[\mathrm{kpc}]$")
        ax.set_ylabel("$N$")

        ax.set_yscale("log")
        ax.set_xscale("log")

        ax.legend()

        fig.savefig("plots/kron_radius_hist_Filter-" + f + "_Orientation-"
                + orientation + "_Type-" + Type + "_Snap-" + snap + ".png",
                    bbox_inches="tight")

        plt.close(fig)


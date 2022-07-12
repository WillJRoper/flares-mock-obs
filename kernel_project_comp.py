import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import time


np.random.seed(100)

def spline_func(q):
    k = 21 / (2 * np.pi)

    w = np.zeros_like(q)
    okinds = q <= 1

    w[okinds] = (1 - q[okinds]) ** 4 * (1 + 4 * q[okinds])

    return k * w


def make_spline_img(pos, Ndim, i, j, tree, ls, smooth,
                    spline_func=spline_func, spline_cut_off=1):
    # Define 2D projected particle position array
    part_pos = pos[:, (i, j)]

    # Initialise the image array
    smooth_img = np.zeros((Ndim, Ndim))

    # Define x and y positions of pixels
    X, Y = np.meshgrid(np.arange(0, Ndim, 1),
                       np.arange(0, Ndim, 1))

    # Define pixel position array for the KDTree
    pix_pos = np.zeros((X.size, 2), dtype=int)
    pix_pos[:, 0] = X.ravel()
    pix_pos[:, 1] = Y.ravel()

    tot = 0

    for ipos, l, sml in zip(part_pos, ls, smooth):

        # Query the tree for this particle
        dist, inds = tree.query(ipos, k=pos.shape[0],
                                distance_upper_bound=spline_cut_off * sml)

        if type(dist) is float:
            continue

        okinds = dist < spline_cut_off * sml
        dist = dist[okinds]
        inds = inds[okinds]

        # Get the kernel
        w = spline_func(dist / sml)

        # Place the kernel for this particle within the img
        kernel = w / sml ** 3
        norm_kernel = kernel / np.sum(kernel)
        smooth_img[pix_pos[inds, 0], pix_pos[inds, 1]] += l * norm_kernel
        tot += np.sum(l * norm_kernel)

    print(tot)

    return smooth_img


def make_spline_img_3d(part_pos, Ndim, i, j, k, tree, ls, smooth,
                    spline_func=spline_func, spline_cut_off=1):
    # Define 2D projected particle position array
    pos = np.zeros_like(part_pos)
    pos[:, 0] = part_pos[:, i]
    pos[:, 1] = part_pos[:, j]
    pos[:, 2] = part_pos[:, k]

    # Initialise the image array
    smooth_img = np.zeros((Ndim, Ndim, Ndim))

    # Define x and y positions of pixels
    X, Y, Z = np.meshgrid(np.arange(0, Ndim, 1),
                          np.arange(0, Ndim, 1),
                          np.arange(0, Ndim, 1))

    # Define pixel position array for the KDTree
    pix_pos = np.zeros((X.size, 3), dtype=int)
    pix_pos[:, 0] = X.ravel()
    pix_pos[:, 1] = Y.ravel()
    pix_pos[:, 2] = Z.ravel()

    for ipos, l, sml in zip(pos, ls, smooth):

        # Query the tree for this particle
        dist, inds = tree.query(ipos, k=pos.shape[0],
                                distance_upper_bound=spline_cut_off * sml)

        if type(dist) is float:
            continue

        okinds = dist < spline_cut_off * sml
        dist = dist[okinds]
        inds = inds[okinds]

        # Get the kernel
        w = spline_func(dist / sml)

        # Place the kernel for this particle within the img
        kernel = w / sml ** 3
        norm_kernel = kernel / np.sum(kernel)
        smooth_img[pix_pos[inds, 0], pix_pos[inds, 1], pix_pos[inds, 2]] += l * norm_kernel

    return np.sum(smooth_img, axis=-1)

nstar = 1000
Ndim = 100
res = Ndim
sml = 2
smls = np.random.uniform(1, 5, nstar)
poss = np.random.normal(0, 2, (nstar, 3))
pos = np.array([0, 0, 0])
pix_width = 0.1
width = Ndim * pix_width

# Define range and extent for the images
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

start = time.time()
kd_2d = make_spline_img(poss, Ndim, 0, 1, tree, np.array([1] * poss.shape[0]),
                smls,
                spline_func=spline_func, spline_cut_off=1)
print("Took:", time.time() - start)

# Define x and y positions of pixels
X, Y, Z = np.meshgrid(np.linspace(imgrange[0][0], imgrange[0][1], res),
                      np.linspace(imgrange[1][0], imgrange[1][1], res),
                      np.linspace(imgrange[1][0], imgrange[1][1], res))

# Define pixel position array for the KDTree
pix_pos = np.zeros((X.size, 3))
pix_pos[:, 0] = X.ravel()
pix_pos[:, 1] = Y.ravel()
pix_pos[:, 2] = Z.ravel()

# Build KDTree
tree = cKDTree(pix_pos)

print("Pixel tree built")

start = time.time()
kd_3d = make_spline_img_3d(poss, Ndim, 0, 1, 2, tree,
                           np.array([1] * poss.shape[0]),
                           smls, spline_func=spline_func, spline_cut_off=1)
print("Took:", time.time() - start)

ipos = np.array((0, 0))

start = time.time()

# # Initialise the image array
# smooth_img = np.zeros((Ndim, Ndim))
#
# # for ipos, sml in zip(poss, np.array([sml] * poss.shape[0])):
#
# low = int(-sml / pix_width)
# high = int(sml / pix_width)
#
# pix_range = np.arange(low, high + 1, 1, dtype=int)
#
# ii, jj = np.meshgrid(pix_range, pix_range)
#
# dists = np.sqrt(ii ** 2 + jj ** 2) * pix_width
#
# # Get the kernel
# w = spline_func(dists / sml)
#
# # Place the kernel for this particle within the img
# kernel = w / sml ** 3
# norm_kernel = kernel / np.sum(kernel)
#
# i, j = int((ipos[1] / pix_width) + Ndim / 2), \
#        int((ipos[0] / pix_width) + Ndim / 2)
# i_low = int(i - (sml / pix_width))
# j_low = int(j - (sml / pix_width))
# i_high = int(i + (sml / pix_width))
# j_high = int(j + (sml / pix_width))
#
# # Place the kernel for this particle within the img
# smooth_img[i_low: i_high + 1, j_low: j_high + 1] += norm_kernel

print("Took:", time.time() - start)

# plt.imshow(smooth_img)
# plt.colorbar()
# plt.show()

start = time.time()

# # Initialise the image array
# smooth_img_project = np.zeros((Ndim, Ndim))
#
# for ipos, sml in zip(poss, smls):
#
#     low = int(-sml * 1.5 / pix_width)
#     high = int(sml * 1.5 / pix_width)
#
#     pix_range = np.arange(low, high + 1, 1, dtype=int)
#
#     ii, jj, kk = np.meshgrid(pix_range, pix_range, pix_range)
#
#     dists = np.sqrt(ii ** 2 + jj ** 2 + kk ** 2) * pix_width
#
#     # Get the kernel
#     w = spline_func(dists / sml)
#
#     # Place the kernel for this particle within the img
#     kernel = w / sml ** 3
#     norm_kernel = kernel / np.sum(kernel)
#
#     i, j = int((ipos[1] / pix_width) + Ndim / 2), \
#            int((ipos[0] / pix_width) + Ndim / 2)
#     i_low = int(i - (sml * 1.5 / pix_width))
#     j_low = int(j - (sml * 1.5 / pix_width))
#     i_high = int(i + (sml * 1.5 / pix_width))
#     j_high = int(j + (sml * 1.5 / pix_width))
#
#     # Place the kernel for this particle within the img
#     smooth_img_project[i_low: i_high + 1, j_low: j_high + 1] += norm_kernel

print("Took:", time.time() - start)

# plt.imshow(kd_3d - smooth_img_project)
# plt.show()
print(np.sum(kd_2d), np.sum(kd_3d))
plt.imshow(kd_2d)
plt.colorbar()
plt.show()
plt.imshow(kd_3d)
plt.colorbar()
plt.show()
# resi = smooth_img - smooth_img_project
# plt.imshow((smooth_img_project - smooth_img) / smooth_img * 100)
# plt.colorbar()
# plt.show()

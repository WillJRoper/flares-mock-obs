"""
    All the functions listed here requires the generation of the particle
    information file.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname("__file__"), '..')))
from functools import partial
import schwimmbad
from synthobs.sed import models
import flare
import flare.filters
from flare.photom import lum_to_M
import utilities as util
import eagle_IO.eagle_IO as E
from scipy.spatial import cKDTree


def DTM_fit(Z, Age):
    """
    Fit function from L-GALAXIES dust modeling
    Formula uses Age in Gyr while the supplied Age is in Myr
    """

    D0, D1, alpha, beta, gamma = 0.008, 0.329, 0.017, -1.337, 2.122
    tau = 5e-5 / (D0 * Z)
    DTM = D0 + (D1 - D0) * (1. - np.exp(-alpha * (Z ** beta)
                                        * ((Age / (1e3 * tau)) ** gamma)))
    if np.isnan(DTM) or np.isinf(DTM):
        DTM = 0.

    return DTM


def get_data(reg, snap, r):

    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data'

    S_coords = E.read_array('PARTDATA', path, snap,
                            'PartType4/Coordinates', noH=True,
                            physicalUnits=True, numThreads=8)
    G_coords = E.read_array('PARTDATA', path, snap,
                            'PartType0/Coordinates', noH=True,
                            physicalUnits=True, numThreads=8)

    S_sml = E.read_array('PARTDATA', path, snap,
                         'PartType4/SmoothingLength', noH=True,
                         physicalUnits=True, numThreads=8)
    G_sml = E.read_array('PARTDATA', path, snap,
                         'PartType0/SmoothingLength', noH=True,
                         physicalUnits=True, numThreads=8)
    a_born = E.read_array('PARTDATA', path, snap,
                          'PartType4/StellarFormationTime', noH=True,
                          physicalUnits=True, numThreads=8)
    S_Z = E.read_array('PARTDATA', path, snap,
                       'PartType4/SmoothedMetallicity', noH=True,
                       physicalUnits=True, numThreads=8)
    G_Z = E.read_array('PARTDATA', path, snap,
                       'PartType0/SmoothedMetallicity', noH=True,
                       physicalUnits=True, numThreads=8)
    S_mass_ini = E.read_array('PARTDATA', path, snap,
                              'PartType4/InitialMass',
                              noH=True, physicalUnits=True,
                              numThreads=8) * 10 ** 10
    S_mass = E.read_array('PARTDATA', path, snap, 'PartType4/Mass',
                          noH=True, physicalUnits=True,
                          numThreads=8) * 10 ** 10
    G_mass = E.read_array('PARTDATA', path, snap, 'PartType0/Mass',
                          noH=True, physicalUnits=True,
                          numThreads=8) * 10 ** 10

    # Calculate ages
    if len(a_born) > 0:
        S_age = util.calc_ages(z, a_born)
    else:
        S_age = a_born


    return S_mass_ini, S_Z, S_age, G_Z, G_sml, S_sml, G_mass, S_coords, \
           G_coords, S_mass


def lum(sim, kappa, tag, BC_fac, inp='FLARES', IMF='Chabrier_300', LF=True,
        filters=('FAKE.TH.FUV',), Type='Total', log10t_BC=7.,
        extinction='default', orientation="sim", masslim=None):
    kinp = np.load('/cosma7/data/dp004/dc-payy1/my_files/'
                   'los/kernel_sph-anarchy.npz',
                   allow_pickle=True)
    lkernel = kinp['kernel']
    header = kinp['header']
    kbins = header.item()['bins']

    if masslim == None:
        masslim = 100

    S_mass_ini, S_Z, S_age, G_Z, G_sml, S_sml, G_mass, S_coords, \
    G_coords, S_mass, cops = get_data(sim, tag, masslim)

    Lums = {f: {} for f in filters}

    model = models.define_model(
        F'BPASSv2.2.1.binary/{IMF}')  # DEFINE SED GRID -
    if extinction == 'default':
        model.dust_ISM = (
            'simple', {'slope': -1.})  # Define dust curve for ISM
        model.dust_BC = ('simple', {
            'slope': -1.})  # Define dust curve for birth cloud component
    elif extinction == 'Calzetti':
        model.dust_ISM = ('Starburst_Calzetti2000', {''})
        model.dust_BC = ('Starburst_Calzetti2000', {''})
    elif extinction == 'SMC':
        model.dust_ISM = ('SMC_Pei92', {''})
        model.dust_BC = ('SMC_Pei92', {''})
    elif extinction == 'MW':
        model.dust_ISM = ('MW_Pei92', {''})
        model.dust_BC = ('MW_Pei92', {''})
    elif extinction == 'N18':
        model.dust_ISM = ('MW_N18', {''})
        model.dust_BC = ('MW_N18', {''})
    else:
        ValueError("Extinction type not recognised")

    z = float(tag[5:].replace('p', '.'))

    # --- create rest-frame luminosities
    F = flare.filters.add_filters(filters, new_lam=model.lam)
    model.create_Lnu_grid(
        F)  # --- create new L grid for each filter. In units of erg/s/Hz

    if S_coords.shape[0] > 0:

        star_tree = cKDTree(S_coords)
        gas_tree = cKDTree(G_coords)

    for ind, cop in enumerate(cops):

        okinds = star_tree.query_ball_point(cop, r=1)
        g_okinds = gas_tree.query_ball_point(cop, r=1)

        # Extract values for this galaxy
        Masses = S_mass_ini[okinds]
        Ages = S_age[okinds]
        Metallicities = S_Z[okinds]
        Smls = S_sml[okinds]
        gasMetallicities = G_Z[g_okinds]
        gasSML = G_sml[g_okinds]
        gasMasses = G_mass[g_okinds]

        Lums[ind]["smls"] = Smls
        Lums[ind]["masses"] = Masses

        if orientation == "sim":

            starCoords = S_coords[okinds, :]
            gasCoords = G_coords[g_okinds, :]

            MetSurfaceDensities = util.get_Z_LOS(starCoords, gasCoords,
                                                 gasMasses, gasMetallicities,
                                                 gasSML, (0, 1, 2),
                                                 lkernel, kbins)

            Lums[ind]["coords"] = starCoords - cops

        # elif orientation == "face-on":
        #
        #     starCoords = S_coords[:, begin[jj]: end[jj]].T - cops[:, jj]
        #     gasCoords = G_coords[:, gbegin[jj]: gend[jj]].T - cops[:, jj]
        #     gasVels = G_vels[gbegin[jj]: gend[jj], :]
        #
        #     # Get angular momentum vector
        #     ang_vec = util.ang_mom_vector(gasMasses, gasCoords, gasVels)
        #
        #     # Rotate positions
        #     starCoords = util.get_rotated_coords(ang_vec, starCoords)
        #     gasCoords = util.get_rotated_coords(ang_vec, gasCoords)
        #     S_coords[:, begin[jj]: end[jj]] = starCoords.T
        #
        #     MetSurfaceDensities = util.get_Z_LOS(starCoords, gasCoords,
        #                                          gasMasses, gasMetallicities,
        #                                          gasSML, (0, 1, 2),
        #                                          lkernel, kbins)
        # elif orientation == "side-on":
        #
        #     starCoords = S_coords[:, begin[jj]: end[jj]].T - cops[:, jj]
        #     gasCoords = G_coords[:, gbegin[jj]: gend[jj]].T - cops[:, jj]
        #     gasVels = G_vels[:, gbegin[jj]: gend[jj]]
        #
        #     # Get angular momentum vector
        #     ang_vec = util.ang_mom_vector(gasMasses, gasCoords, gasVels)
        #
        #     # Rotate positions
        #     starCoords = util.get_rotated_coords(ang_vec, starCoords)
        #     gasCoords = util.get_rotated_coords(ang_vec, gasCoords)
        #     S_coords[:, begin[jj]: end[jj]] = starCoords.T
        #
        #     MetSurfaceDensities = util.get_Z_LOS(starCoords, gasCoords,
        #                                          gasMasses, gasMetallicities,
        #                                          gasSML, (2, 0, 1),
        #                                          lkernel, kbins)
        else:
            MetSurfaceDensities = None
            print(orientation,
                  "is not an recognised orientation. "
                  "Accepted types are 'sim', 'face-on', or 'side-on'")

        Mage = np.nansum(Masses * Ages) / np.nansum(Masses)
        Z = np.nanmean(gasMetallicities)

        MetSurfaceDensities = DTM_fit(Z, Mage) * MetSurfaceDensities

        if Type == 'Total':
            # --- calculate V-band (550nm) optical depth for each star particle
            tauVs_ISM = kappa * MetSurfaceDensities
            tauVs_BC = BC_fac * (Metallicities / 0.01)
            fesc = 0.0

        elif Type == 'Pure-stellar':
            tauVs_ISM = np.zeros(len(Masses))
            tauVs_BC = np.zeros(len(Masses))
            fesc = 1.0

        elif Type == 'Intrinsic':
            tauVs_ISM = np.zeros(len(Masses))
            tauVs_BC = np.zeros(len(Masses))
            fesc = 0.0

        elif Type == 'Only-BC':
            tauVs_ISM = np.zeros(len(Masses))
            tauVs_BC = BC_fac * (Metallicities / 0.01)
            fesc = 0.0

        else:
            tauVs_ISM = None
            tauVs_BC = None
            fesc = None
            ValueError(F"Undefined Type {Type}")

        # --- calculate rest-frame Luminosity. In units of erg/s/Hz
        for f in filters:
            Lnu = models.generate_Lnu_array(model, Masses, Ages, Metallicities,
                                            tauVs_ISM, tauVs_BC, F, f,
                                            fesc=fesc, log10t_BC=log10t_BC)
            
            Lums[f][ind] = Lnu

    return Lums


def flux(sim, kappa, tag, BC_fac, IMF='Chabrier_300',
        filters=('FAKE.TH.FUV',), Type='Total', log10t_BC=7.,
        extinction='default', orientation="sim", r=1):
    kinp = np.load('/cosma7/data/dp004/dc-payy1/my_files/'
                   'los/kernel_sph-anarchy.npz',
                   allow_pickle=True)
    lkernel = kinp['kernel']
    header = kinp['header']
    kbins = header.item()['bins']

    Masses, Metallicities, Ages, gasMetallicities, gasSML, Smls, \
    gasMasses, starCoords, gasCoords, S_mass = get_data(sim, tag, r)

    print("Got Data")

    model = models.define_model(
        F'BPASSv2.2.1.binary/{IMF}')  # DEFINE SED GRID -
    if extinction == 'default':
        model.dust_ISM = (
            'simple', {'slope': -1.})  # Define dust curve for ISM
        model.dust_BC = ('simple', {
            'slope': -1.})  # Define dust curve for birth cloud component
    elif extinction == 'Calzetti':
        model.dust_ISM = ('Starburst_Calzetti2000', {''})
        model.dust_BC = ('Starburst_Calzetti2000', {''})
    elif extinction == 'SMC':
        model.dust_ISM = ('SMC_Pei92', {''})
        model.dust_BC = ('SMC_Pei92', {''})
    elif extinction == 'MW':
        model.dust_ISM = ('MW_Pei92', {''})
        model.dust_BC = ('MW_Pei92', {''})
    elif extinction == 'N18':
        model.dust_ISM = ('MW_N18', {''})
        model.dust_BC = ('MW_N18', {''})
    else:
        ValueError("Extinction type not recognised")

    z = float(tag[5:].replace('p', '.'))
    F = flare.filters.add_filters(filters, new_lam=model.lam * (1. + z))

    cosmo = flare.default_cosmo()

    # --- create new Fnu grid for each filter. In units of nJy/M_sol
    model.create_Fnu_grid(F, z, cosmo)

    Fnus = {}

    if starCoords.shape[0] > 0:

        Fnus["smls"] = Smls
        Fnus["masses"] = Masses

        if orientation == "sim":

            MetSurfaceDensities = util.get_Z_LOS(starCoords, gasCoords,
                                                 gasMasses, gasMetallicities,
                                                 gasSML, (0, 1, 2),
                                                 lkernel, kbins)

            Fnus["coords"] = starCoords - np.mean(starCoords, axis=0)

            print("Calculated surface densities")

        # elif orientation == "face-on":
        #
        #     starCoords = S_coords[:, begin[jj]: end[jj]].T - cops[:, jj]
        #     gasCoords = G_coords[:, gbegin[jj]: gend[jj]].T - cops[:, jj]
        #     gasVels = G_vels[gbegin[jj]: gend[jj], :]
        #
        #     # Get angular momentum vector
        #     ang_vec = util.ang_mom_vector(gasMasses, gasCoords, gasVels)
        #
        #     # Rotate positions
        #     starCoords = util.get_rotated_coords(ang_vec, starCoords)
        #     gasCoords = util.get_rotated_coords(ang_vec, gasCoords)
        #     S_coords[:, begin[jj]: end[jj]] = starCoords.T
        #
        #     MetSurfaceDensities = util.get_Z_LOS(starCoords, gasCoords,
        #                                          gasMasses, gasMetallicities,
        #                                          gasSML, (0, 1, 2),
        #                                          lkernel, kbins)
        # elif orientation == "side-on":
        #
        #     starCoords = S_coords[:, begin[jj]: end[jj]].T - cops[:, jj]
        #     gasCoords = G_coords[:, gbegin[jj]: gend[jj]].T - cops[:, jj]
        #     gasVels = G_vels[:, gbegin[jj]: gend[jj]]
        #
        #     # Get angular momentum vector
        #     ang_vec = util.ang_mom_vector(gasMasses, gasCoords, gasVels)
        #
        #     # Rotate positions
        #     starCoords = util.get_rotated_coords(ang_vec, starCoords)
        #     gasCoords = util.get_rotated_coords(ang_vec, gasCoords)
        #     S_coords[:, begin[jj]: end[jj]] = starCoords.T
        #
        #     MetSurfaceDensities = util.get_Z_LOS(starCoords, gasCoords,
        #                                          gasMasses, gasMetallicities,
        #                                          gasSML, (2, 0, 1),
        #                                          lkernel, kbins)
        else:
            MetSurfaceDensities = None
            print(orientation,
                  "is not an recognised orientation. "
                  "Accepted types are 'sim', 'face-on', or 'side-on'")

        Mage = np.nansum(Masses * Ages) / np.nansum(Masses)
        Z = np.nanmean(gasMetallicities)

        MetSurfaceDensities = DTM_fit(Z, Mage) * MetSurfaceDensities

        if Type == 'Total':
            # --- calculate V-band (550nm) optical depth for each star particle
            tauVs_ISM = kappa * MetSurfaceDensities
            tauVs_BC = BC_fac * (Metallicities / 0.01)
            fesc = 0.0

        elif Type == 'Pure-stellar':
            tauVs_ISM = np.zeros(len(Masses))
            tauVs_BC = np.zeros(len(Masses))
            fesc = 1.0

        elif Type == 'Intrinsic':
            tauVs_ISM = np.zeros(len(Masses))
            tauVs_BC = np.zeros(len(Masses))
            fesc = 0.0

        elif Type == 'Only-BC':
            tauVs_ISM = np.zeros(len(Masses))
            tauVs_BC = BC_fac * (Metallicities / 0.01)
            fesc = 0.0

        else:
            tauVs_ISM = None
            tauVs_BC = None
            fesc = None
            ValueError(F"Undefined Type {Type}")

        # --- calculate rest-frame Luminosity. In units of erg/s/Hz
        for f in filters:

            # --- calculate rest-frame flux of each object in nJy
            Fnu = models.generate_Fnu_array(model, Masses, Ages, Metallicities,
                                            tauVs_ISM, tauVs_BC, F, f,
                                            fesc=fesc, log10t_BC=log10t_BC)

            Fnus[f] = Fnu

    return Fnus


def get_lum(sim, kappa, tag, BC_fac, IMF='Chabrier_300',
            bins=np.arange(-24, -16, 0.5), inp='FLARES', LF=True,
            filters=('FAKE.TH.FUV'), Type='Total', log10t_BC=7.,
            extinction='default', orientation="sim", masslim=None):
    try:
        Lums = lum(sim, kappa, tag, BC_fac=BC_fac, IMF=IMF, inp=inp, LF=LF,
                   filters=filters, Type=Type, log10t_BC=log10t_BC,
                   extinction=extinction, orientation=orientation,
                   masslim=masslim)

    except Exception as e:
        Lums = {f: np.array([], dtype=np.float64) for f in filters}
        Lums["coords"] = np.array([], dtype=np.float64)
        Lums["smls"] = np.array([], dtype=np.float64)
        Lums["masses"] = np.array([], dtype=np.float64)
        Lums["nstar"] = np.array([], dtype=np.float64)
        Lums["begin"] = np.array([], dtype=np.float64)
        Lums["end"] = np.array([], dtype=np.float64)
        print(e)

    if LF:
        tmp, edges = np.histogram(lum_to_M(Lums), bins=bins)
        return tmp

    else:
        return Lums


def get_lum_all(kappa, tag, BC_fac, IMF='Chabrier_300',
                bins=np.arange(-24, -16, 0.5), inp='FLARES', LF=True,
                filters=('FAKE.TH.FUV'), Type='Total', log10t_BC=7.,
                extinction='default', orientation="sim", numThreads=8,
                masslim=None):
    print(f"Getting luminosities for tag {tag} with kappa={kappa}")

    if inp == 'FLARES':
        df = pd.read_csv('../weight_files/weights_grid.txt')
        weights = np.array(df['weights'])

        sims = np.arange(0, len(weights))

        calc = partial(get_lum, kappa=kappa, tag=tag, BC_fac=BC_fac, IMF=IMF,
                       bins=bins, inp=inp, LF=LF, filters=filters, Type=Type,
                       log10t_BC=log10t_BC, extinction=extinction,
                       orientation=orientation, masslim=masslim)

        pool = schwimmbad.MultiPool(processes=numThreads)
        dat = np.array(list(pool.map(calc, sims)))
        pool.close()

        if LF:
            hist = np.sum(dat, axis=0)
            out = np.zeros(len(bins) - 1)
            err = np.zeros(len(bins) - 1)
            for ii, sim in enumerate(sims):
                err += np.square(np.sqrt(dat[ii]) * weights[ii])
                out += dat[ii] * weights[ii]

            return out, hist, np.sqrt(err)

        else:
            return dat

    else:
        out = get_lum(00, kappa=kappa, tag=tag, BC_fac=BC_fac, IMF=IMF,
                      bins=bins, inp=inp, LF=LF, filters=filters, Type=Type,
                      log10t_BC=log10t_BC, extinction=extinction,
                      orientation=orientation, masslim=masslim)

        return out


def get_flux(sim, kappa, tag, BC_fac, IMF='Chabrier_300', inp='FLARES',
             filters=flare.filters.NIRCam, Type='Total', log10t_BC=7.,
             extinction='default', orientation="sim"):
    try:
        Fnus = flux(sim, kappa, tag, BC_fac=BC_fac, IMF=IMF, inp=inp,
                    filters=filters, Type=Type, log10t_BC=log10t_BC,
                    extinction=extinction, orientation=orientation)

    except Exception as e:
        Fnus = np.ones(len(filters)) * np.nan
        print(e)

    return Fnus


def get_flux_all(kappa, tag, BC_fac, IMF='Chabrier_300', inp='FLARES',
                 filters=flare.filters.NIRCam, Type='Total', log10t_BC=7.,
                 extinction='default', orientation="sim", numThreads=8):
    print(f"Getting fluxes for tag {tag} with kappa={kappa}")

    if inp == 'FLARES':

        df = pd.read_csv('../weight_files/weights_grid.txt')
        weights = np.array(df['weights'])

        sims = np.arange(0, len(weights))

        calc = partial(get_flux, kappa=kappa, tag=tag, BC_fac=BC_fac, IMF=IMF,
                       inp=inp, filters=filters, Type=Type,
                       log10t_BC=log10t_BC, extinction=extinction,
                       orientation=orientation)

        pool = schwimmbad.MultiPool(processes=numThreads)
        out = np.array(list(pool.map(calc, sims)))
        pool.close()

    else:

        out = get_flux(00, kappa=kappa, tag=tag, BC_fac=BC_fac, IMF=IMF,
                       inp=inp, filters=filters, Type=Type,
                       log10t_BC=log10t_BC, extinction=extinction,
                       orientation=orientation)

    return out

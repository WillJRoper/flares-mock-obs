#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 10:17:37 2023

@author: u92876da
"""

# SFR calculation
import os
import astropy.units as u
import astropy.constants as const
import numpy as np
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit
from astropy.cosmology import FlatLambdaCDM
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.lines import Line2D


def weighted_quantile(
    values, quantiles, sample_weight=None, values_sorted=False, old_style=False
):
    """
    Taken from From https://stackoverflow.com/a/29677616/1718096
    Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """

    # do some housekeeping
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(
        quantiles <= 1
    ), "quantiles should be in [0, 1]"

    # if not sorted, sort values array
    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


def plot_meidan_stat(xs, ys, w, ax, lab, color, bins, ls="-"):
    # Compute binned statistics
    def func(y):
        return weighted_quantile(y, 0.5, sample_weight=w)

    y_stat, binedges, bin_ind = binned_statistic(
        xs, ys, statistic=func, bins=bins
    )

    # Compute bincentres
    bin_cents = (bins[1:] + bins[:-1]) / 2

    if color is not None:
        return ax.plot(bin_cents, y_stat, color=color, linestyle=ls, label=lab)
    else:
        return ax.plot(bin_cents, y_stat, color=color, linestyle=ls, label=lab)


# load FLARES public data
def get_phot(num, tag, jwstFilter, sim):
    with h5py.File(sim, "r") as hf:
        flux = hf[
            num
            + tag
            + "/Galaxy/BPASS_2.2.1/Chabrier300/Flux/DustModelI/JWST/NIRCAM/"
            + jwstFilter
        ][:]
    return flux


def get_mass(num, tag, sim):
    with h5py.File(sim, "r") as hf:
        Mstar = hf[num + tag + "/Galaxy/Mstar_aperture/30"][:] * 1e10
    return Mstar


def get_sfr(num, tag, sim, t_SFR=100):
    with h5py.File(sim, "r") as hf:
        sfr_100 = hf[num + tag + "/Galaxy" + f"/SFR_aperture/30/{t_SFR}Myr"][:]
    return sfr_100


def load_flares_public(
    z_arr,
    filters,
    sim="/cosma7/data/dp004/dc-payy1/my_files/"
    + "flares_pipeline/data/flares.hdf5",
):
    # Load weights
    df = pd.read_csv("../weight_files/weights_grid.txt")
    weights = np.array(df["weights"])

    f090w_list = []
    f115w_list = []
    f150w_list = []
    f200w_list = []
    f277w_list = []
    f356w_list = []
    f410m_list = []
    f444w_list = []

    massList = []
    sfr100List = []
    zList = []
    weights_out = []
    ids = []
    z_labels = {
        5: "010_z005p000",
        6: "009_z006p000",
        7: "008_z007p000",
        8: "007_z008p000",
        9: "006_z009p000",
        10: "005_z010p000",
        11: "004_z011p000",
        12: "003_z012p000",
        13: "002_z013p000",
        14: "001_z014p000",
        15: "000_z015p000",
    }

    for z, tag in zip(z_arr, [z_labels[redshift] for redshift in z_arr]):
        jwstFilters = filters  # ['F090W', 'F115W', 'F150W', 'F200W','F277W', 'F356W', 'F444W']

        for i, jwstFilter in enumerate(jwstFilters):
            print(z, jwstFilter)

            for j in range(40):
                num = str(j)
                if len(num) == 1:
                    num = "0" + num
                num = num + "/"

                flux = get_phot(num, tag, jwstFilter, sim)
                Mstar = get_mass(num, tag, sim)
                sfr100 = get_sfr(num, tag, sim, 100)

                with h5py.File(sim, "r") as hf:
                    grp_ids = hf[num + tag + "/Galaxy/GroupNumber"][:]
                    subgrp_ids = hf[num + tag + "/Galaxy/SubGroupNumber"][:]

                listDict = {
                    "f090W": f090w_list,
                    "f115W": f115w_list,
                    "f150W": f150w_list,
                    "f200W": f200w_list,
                    "f277W": f277w_list,
                    "f356W": f356w_list,
                    "f410M": f410m_list,
                    "f444W": f444w_list,
                }

                if jwstFilter == "F090W":
                    massList.extend(Mstar)
                    sfr100List.extend(sfr100)
                    zList.extend(np.full(len(Mstar), z))
                    weights_out.extend(np.full(len(Mstar), weights[j]))
                    for i in range(len(grp_ids)):
                        ids.append(
                            str(j)
                            + "_"
                            + str(grp_ids[i])
                            + "_"
                            + str(subgrp_ids[i])
                        )

                listDict[jwstFilter.replace("F", "f")].extend(flux)

    flares = {
        "ID": ids,
        "z": zList,
        "phot": listDict,
        "mass": massList,
        "sfr100": sfr100List,
        "weights": weights_out,
    }

    return flares


# %%

# NIRCam
NIRCam_band_wavelengths = {
    "f070W": 7_056.0,
    "f090W": 9_044.0,
    "f115W": 11_571.0,
    "f140M": 14_060.0,
    "f150W": 15_040.0,
    "f162M": 16_281.0,
    "f182M": 18_466.0,
    "f200W": 19_934.0,
    "f210M": 20_964.0,
    "f250M": 25_038.0,
    "f277W": 27_695.0,
    "f300M": 29_908.0,
    "f335M": 33_639.0,
    "f356W": 35_768.0,
    "f360M": 36_261.0,
    "f410M": 40_844.0,
    "f430M": 42_818.0,
    "f444W": 44_159.0,
    "f460M": 46_305.0,
    "f480M": 48_192.0,
}
NIRCam_band_wavelengths = {
    key: value * u.Angstrom for (key, value) in NIRCam_band_wavelengths.items()
}  # convert each individual value to Angstrom
NIRCam_band_FWHMs = {
    "f070W": 1_600.0,
    "f090W": 2_101.0,
    "f115W": 2_683.0,
    "f140M": 1_478.0,
    "f150W": 3_371.0,
    "f162M": 1_713.0,
    "f182M": 2_459.0,
    "f200W": 4_717.0,
    "f210M": 2_089.0,
    "f250M": 1_825.0,
    "f277W": 7_110.0,
    "f300M": 3_264.0,
    "f335M": 3_609.0,
    "f356W": 8_408.0,
    "f360M": 3_873.0,
    "f410M": 4_375.0,
    "f430M": 2_312.0,
    "f444W": 11_055.0,
    "f460M": 2_322.0,
    "f480M": 3_145.0,
}
NIRCam_band_FWHMs = {
    key: value * u.Angstrom for (key, value) in NIRCam_band_FWHMs.items()
}  # convert each individual value to Angstrom

# ACS_WFC
ACS_WFC_band_wavelengths = {
    "f435W": 4_340.0,
    "fr459M": 4_590.0,
    "f475W": 4_766.0,
    "f550M": 5_584.0,
    "f555W": 5_373.0,
    "f606W": 5_960.0,
    "f625W": 6_325.0,
    "fr647M": 6_472.0,
    "f775W": 7_706.0,
    "f814W": 8_073.0,
    "f850LP": 9_047.0,
    "fr914M": 9_072.0,
}
ACS_WFC_band_wavelengths = {
    key: value * u.Angstrom
    for (key, value) in ACS_WFC_band_wavelengths.items()
}  # convert each individual value to Angstrom
# FWHMs corrrespond to FWHM of the filters from SVO Filter Profile Service
ACS_WFC_band_FWHMs = {
    "f435W": 937.0,
    "fr459M": 350.0,
    "f475W": 1_437.0,
    "f550M": 546.0,
    "f555W": 1_240.0,
    "f606W": 2_322.0,
    "f625W": 1_416.0,
    "fr647M": 501.0,
    "f775W": 1_511.0,
    "f814W": 1_858.0,
    "f850LP": 1_208.0,
    "fr914M": 774.0,
}
ACS_WFC_band_FWHMs = {
    key: value * u.Angstrom for (key, value) in ACS_WFC_band_FWHMs.items()
}  # convert each individual value to Angstrom

# combine bands from different instruments
band_wavelengths = {**NIRCam_band_wavelengths, **ACS_WFC_band_wavelengths}
band_FWHMs = {**NIRCam_band_FWHMs, **ACS_WFC_band_FWHMs}


def obs_flux_lambda_to_rest(obs_flux_lambda, z):
    rest_flux_lambda = obs_flux_lambda * ((1 + z) ** 2)
    return rest_flux_lambda


def wav_obs_to_rest(wav_obs, z):
    wav_rest = wav_obs / (1 + z)
    return wav_rest


def rest_flux_lambda_from_nu(wav, obs_flux_Jy, z):
    obs_flux_lambda = (obs_flux_Jy * const.c / (wav**2)).to(
        u.erg / (u.s * (u.cm**2) * u.Angstrom)
    )
    rest_flux_lambda = obs_flux_lambda_to_rest(obs_flux_lambda, z)
    return rest_flux_lambda


def crop_phot_to_rest_UV(
    wavelength,
    wavelength_err,
    rest_flux_lambda,
    rest_UV_wav_lims=[1250.0, 3000.0] * u.Angstrom,
):
    crop_indices = []
    for i, (wav, wav_err) in enumerate(
        zip(wavelength.value, wavelength_err.value)
    ):
        wav *= u.Angstrom
        wav_err *= u.Angstrom
        # print(wav, wav_err, self.rest_UV_wav_lims)
        if (
            wav - wav_err < rest_UV_wav_lims[0]
            or wav + wav_err > rest_UV_wav_lims[1]
        ):
            crop_indices = np.append(crop_indices, i)
    # print(crop_indices)
    crop_indices = np.array(crop_indices).astype(int)
    wavelength = np.delete(wavelength, crop_indices)
    wavelength_err = np.delete(wavelength_err, crop_indices)
    rest_flux_lambda = np.delete(rest_flux_lambda, crop_indices)
    return wavelength, wavelength_err, rest_flux_lambda


def beta_slope_power_law_func(wav_rest, A, beta):
    return (10**A) * (wav_rest**beta)


def fit_rest_UV_power_law(
    wav, wav_err, obs_flux_nu, obs_flux_nu_unit, z, rest_UV_wav_lims, cosmo
):  # works for a single galaxy
    # convert flux to rest frame
    rest_flux_lambda = rest_flux_lambda_from_nu(
        wav * u.Angstrom, obs_flux_nu * obs_flux_nu_unit, z
    )
    # crop the fluxes to rest frame UV only
    rest_wav = wav_obs_to_rest(wav * u.Angstrom, z)
    rest_wav_err = wav_obs_to_rest(wav_err * u.Angstrom, z)
    wav_UV, wav_err_UV, rest_flux_lambda_UV = crop_phot_to_rest_UV(
        rest_wav, rest_wav_err, rest_flux_lambda, rest_UV_wav_lims
    )
    # perform fitting
    if not len(wav_UV) > 1:
        return -99.0, -99.0, -99.0
    else:
        popt, pcov = curve_fit(
            beta_slope_power_law_func, wav_UV, rest_flux_lambda_UV
        )  # , maxfev = 1_000)
        beta = popt[1]
        flux_lambda_1500 = (
            (beta_slope_power_law_func(1500.0, popt[0], popt[1]))
            * u.erg
            / (u.s * (u.cm**2) * u.Angstrom)
        )
        flux_Jy_1500 = (
            flux_lambda_1500 * ((1500.0 * u.Angstrom) ** 2) / const.c
        ).to(u.Jy)
        m_1500 = flux_Jy_1500.to(u.ABmag)
        lum_distance = cosmo.luminosity_distance(z).to(u.pc)
        M_UV = (
            m_1500.value
            - 5 * np.log10(lum_distance.value / 10)
            + 2.5 * np.log10(1 + z)
        )
        A_UV = 4.43 + (1.99 * beta)
        L_obs = ((4 * np.pi * flux_Jy_1500 * lum_distance**2) / (1 + z)).to(
            u.erg / (u.s * u.Hz)
        )
        L_int = L_obs * 10 ** (A_UV / 2.5) if A_UV > 0 else L_obs
        SFR = 1.15e-28 * L_int.value
        # print(beta, M_UV, SFR)
        return beta, M_UV, SFR


def sSFR_from_phot(
    bands,
    obs_flux,
    z,
    mass,
    rest_UV_wav_lims=[1250.0, 3000.0] * u.Angstrom,
    cosmo=FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05, Tcmb0=2.725),
):
    """

    Parameters
    ----------
    bands : np.array(str)
        bands with lower case 'f'
    obs_flux : np.array(), shape = (len(galaxies), len(bands))
        observed flux_nu
    z : np.array(float)
        galaxy redshifts
    mass : np.array(float)
        units of solar masses
    rest_UV_wav_lims : list
        upper and lower limits of the rest frame UV in units of Angstrom
    cosmo : astropy.cosmology.FlatLambdaCDM, optional
        cosmology The default is FlatLambdaCDM(H0 = 70, Om0 = 0.3, Ob0 = 0.05, Tcmb0 = 2.725).

    Returns
    -------
    TYPE
    beta, M_UV, SFR, sSFR
        output is dimensionless, but units of SFR is sol_mass/yr and sSFR is 1/yr

    """
    # store input flux units
    flux_in_unit = obs_flux.unit

    # calculate wavelengths and FWHMs from input bands
    wav = np.array(
        [band_wavelengths[band.replace("F", "f")].value for band in bands]
    )
    wav_err = np.array(
        [band_FWHMs[band.replace("F", "f")].value / 2 for band in bands]
    )

    # calculate SFR and other UV properties from photometry
    rest_UV_fit_output = np.array(
        [
            fit_rest_UV_power_law(
                wav,
                wav_err,
                flux,
                flux_in_unit,
                redshift,
                rest_UV_wav_lims,
                cosmo,
            )
            for (flux, redshift) in zip(obs_flux.value, z)
        ]
    )
    beta = rest_UV_fit_output[:, 0]
    M_UV = rest_UV_fit_output[:, 1]
    SFR = rest_UV_fit_output[:, 2]
    # print(beta, M_UV, SFR)
    # calculate sSFR
    return (
        beta,
        M_UV,
        SFR,
        np.array(
            [
                SFR[i] / mass[i] if SFR[i] != -99.0 else -99.0
                for i in range(len(mass))
            ]
        ),
    )


def fit(z, a, m):
    return a * (1 + z) ** (m)


if __name__ == "__main__":
    zs = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    # Does the file exist? Avoid remaking it
    if os.path.exists("mass_ssfr_FLARES.csv"):
        df = pd.read_csv("mass_ssfr_FLARES.csv")

        # Has the morphology data been linked in?
        if "Disc_Fractions" not in df.columns:
            # Load the morphology dictionary.
            morp_data = np.load(
                "/cosma7/data/dp004/dc-irod1/FLARES/morph_data_JWST.npy",
                allow_pickle=True,
            )
            morp_data = morp_data.item()

            # Convert dictionary to a set of arrays
            zs = df["Redshift"]
            ms = df["Stellar_Mass (Msun)"]
            ssfrs = df["sSFR (M_sun / Gyr)"]
            ws = df["Weights"]
            regions = df["Regions"]
            grps = df["GroupNumber"]
            subgrps = df["SubGroupNumber"]
            dts = np.full(len(df["SubGroupNumber"]), np.nan)
            disc_fracs = morp_data["disc_fractions"]
            morph_zs = morp_data["redshifts"]
            morph_regs = morp_data["regions"]
            morph_grps = morp_data["group_numbers"]
            morph_subgrps = morp_data["subgroup_numbers"]
            for i in range(len(morph_zs)):
                dt = disc_fracs[i]
                z = morph_zs[i]
                reg = int(morph_regs[i])
                grp = morph_grps[i]
                subgrp = morph_subgrps[i]
                dt_okinds = np.logical_and(regions == reg, grps == grp)
                dt_okinds = np.logical_and(dt_okinds, subgrps == subgrp)
                dt_okinds = np.logical_and(dt_okinds, zs == z)
                if len(zs[dt_okinds]) == 0:
                    print(z, reg, grp, subgrp, "No match")
                else:
                    print("Found DT:", dt)
                    dts[dt_okinds] = dt

            # And make ANOTHER dictionary to make a dataframe from
            csv_dict = {
                "Regions": regions,
                "GroupNumber": grps,
                "SubGroupNumber": subgrps,
                "Redshift": zs,
                "Stellar_Mass (Msun)": ms,
                "sSFR (M_sun / Gyr)": ssfrs,
                "Weights": ws,
                "Disc_Fractions": dts,
            }

            # Make the dataframe
            df = pd.DataFrame.from_dict(csv_dict)
            df.to_csv("mass_ssfr_FLARES.csv")

    else:
        bands = [
            band.replace("f", "F")
            for band in [
                "f090W",
                "f115W",
                "f150W",
                "f200W",
                "f277W",
                "f356W",
                "f444W",
            ]
        ]

        results = {}  # {z: (beta, M_UV, SFR, stellar_mass, sSFR, weights)}

        for z in zs:
            flares_data = load_flares_public(
                [
                    z,
                ],
                filters=bands,
            )
            flares_phot = (
                np.array(
                    [
                        flares_data["phot"][band.replace("F", "f")]
                        for band in bands
                    ]
                ).T
                * u.nJy
            )
            beta, M_UV, SFR, sSFR = sSFR_from_phot(
                bands, flares_phot, flares_data["z"], flares_data["mass"]
            )
            results[z] = (
                flares_data["ID"],
                beta,
                M_UV,
                SFR,
                flares_data["mass"],
                sSFR,
                flares_data["weights"],
            )

        # Convert dictionary to a set of arrays
        zs = []
        ms = []
        ssfrs = []
        ws = []
        regions = []
        grps = []
        subgrps = []
        for key in results:
            zs.extend(np.full(len(results[key][1]), key))
            ms.extend(results[key][4])
            ssfrs.extend(results[key][5])
            ws.extend(results[key][6])
            for i in range(len(results[key][0])):
                split_id = [int(j) for j in results[key][0][i].split("_")]
                regions.append(split_id[0])
                grps.append(split_id[1])
                subgrps.append(split_id[2])

        # And make ANOTHER dictionary to make a dataframe from
        csv_dict = {
            "Regions": regions,
            "GroupNumber": grps,
            "SubGroupNumber": subgrps,
            "Redshift": zs,
            "Stellar_Mass (Msun)": ms,
            "sSFR (M_sun / Gyr)": ssfrs,
            "Weights": ws,
        }

        # Make the dataframe
        df = pd.DataFrame.from_dict(csv_dict)
        df.to_csv("mass_ssfr_FLARES.csv")

    # Define the binning
    mass_bins = [8.5, 9.0, 9.5, 10.0, 10.5, np.inf]
    z_bins = np.array([4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5])
    colors = ["r", "g", "b", "m", "y"]
    linestyle = ["-", "--", "-.", ":", "-"]

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.semilogy()
    ax.grid(True)

    # Plot each mass bin
    for i in range(len(mass_bins) - 1):
        c = colors[i]

        # Get mask for this bin
        okinds = np.logical_and(
            np.log10(df["Stellar_Mass (Msun)"]) >= mass_bins[i],
            np.log10(df["Stellar_Mass (Msun)"]) < mass_bins[i + 1],
        )

        # Fit this mass bin
        popt, pcov = curve_fit(
            fit,
            df["Redshift"][okinds],
            df["sSFR (M_sun / Gyr)"][okinds] * 10**9,
            p0=(1, 0.5),
            sigma=df["Weights"][okinds],
        )
        print(
            mass_bins[i], popt, pcov, "count = ", len(df["Redshift"][okinds])
        )
        xs = np.linspace(5, 10, 1000)
        if mass_bins[i + 1] != np.inf:
            label = "$%.1f \leq \log_{10}(M_\star / M_\odot) < %.1f$" % (
                mass_bins[i],
                mass_bins[i + 1],
            )
        else:
            label = "$%.1f \leq \log_{10}(M_\star / M_\odot)$" % mass_bins[i]

        ax.plot(
            xs,
            fit(xs, popt[0], popt[1]),
            color=c,
            linestyle=linestyle[i],
            label=label,
        )

    ax.set_ylabel("sSFR [Gyr$^{-1}$]")
    ax.set_xlabel("$z$")

    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=2
    )

    fig.savefig("sSFR_evo_massbinned.png", dpi=300, bbox_inches="tight")

    # Define the binning
    dt_thresh = [
        0.25,
    ]
    z_bins = np.array([4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5])
    colors = [
        "k",
    ]

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(True)

    # Define line weights
    line_weights = [1, 2, 3, 4]

    # xs for plotting
    xs = np.linspace(5, 8, 1000)

    # Loop over thresholds
    for i in range(len(dt_thresh)):
        c = colors[i]

        # Fit and plot disks
        okinds = np.logical_and(
            df["Disc_Fractions"] >= dt_thresh[i],
            ~np.isnan(df["Disc_Fractions"]),
        )
        popt, pcov = curve_fit(
            fit,
            df["Redshift"][okinds],
            df["sSFR (M_sun / Gyr)"][okinds] * 10**9,
            p0=(1, 0.5),
            sigma=df["Weights"][okinds],
        )
        print(
            "D/T >=",
            dt_thresh[i],
            popt,
            pcov,
            "count = ",
            len(df["Redshift"][okinds]),
        )
        ax.plot(
            xs, fit(xs, popt[0], popt[1]), color=c, linestyle="-", label="Disk"
        )

        # Fit and plot not disks
        okinds = np.logical_and(
            df["Disc_Fractions"] < dt_thresh[i],
            ~np.isnan(df["Disc_Fractions"]),
        )
        popt, pcov = curve_fit(
            fit,
            df["Redshift"][okinds],
            df["sSFR (M_sun / Gyr)"][okinds] * 10**9,
            p0=(1, 0.5),
            sigma=df["Weights"][okinds],
        )
        print(
            "D/T <",
            dt_thresh[i],
            popt,
            pcov,
            "count = ",
            len(df["Redshift"][okinds]),
        )
        ax.plot(
            xs,
            fit(xs, popt[0], popt[1]),
            color=c,
            linestyle="--",
            label="Spheriod",
        )

    ax.set_ylabel("sSFR [Gyr$^{-1}$]")
    ax.set_xlabel("$z$")

    # Set up legend
    ax.legend()

    fig.savefig("sSFR_evo_DTthresh.png", dpi=300, bbox_inches="tight")

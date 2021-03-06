import numpy as np
from coastsat import SDS_slope, SDS_tools
from ih import plots
import pytz
from datetime import datetime
import pandas as pd
import os
import scipy.io as sio
import pickle
import csv


def beach_slope(filepath_data, sitename):

    with open(
        os.path.join(filepath_data, sitename, sitename + "_output" + ".pkl"), "rb"
    ) as f:
        output = pickle.load(f)

    with open(
        os.path.join(filepath_data, sitename, sitename + "_output_kvos" + ".pkl"), "rb"
    ) as f:
        output_kvos = pickle.load(f)

    geojson_file = os.path.join(filepath_data, sitename, sitename + '_transects.geojson')
    transects = SDS_slope.transects_from_geojson(geojson_file)

    # remove S2 shorelines (the slope estimation algorithm needs only Landsat)
    if "S2" in output["satname"]:
        idx_S2 = np.array([_ == "S2" for _ in output["satname"]])
        for key in output.keys():
            output[key] = [output[key][_] for _ in np.where(~idx_S2)[0]]

    # remove duplicates
    output = SDS_slope.remove_duplicates(output)

    # remove shorelines from images with poor georeferencing (RMSE > 10 m)
    output = SDS_slope.remove_inaccurate_georef(output, 10)

    # plot shorelines and transects
    plots.plot_shorelines_transects(filepath_data, sitename, output, transects)

    # a more robust method to compute intersection is needed here to avoid outliers
    # as these can affect the slope detection algorithm
    settings_transects = {  # parameters for shoreline intersections
        "along_dist": 25,  # along-shore distance to use for intersection
        "max_std": 15,  # max std for points around transect
        "max_range": 30,  # max range for points around transect
        "min_val": -100,  # largest negative value along transect (landwards of transect origin)
        # parameters for outlier removal
        "nan/max": "auto",  # mode for removing outliers ('auto', 'nan', 'max')
        "prc_std": 0.1,  # percentage to use in 'auto' mode to switch from 'nan' to 'max'
        "max_cross_change": 40,  # two values of max_cross_change distance to use
    }
    # compute intersections [advanced version]
    cross_distance = SDS_slope.compute_intersection(
        output, transects, settings_transects
    )

    # remove outliers [advanced version]
    cross_distance = SDS_slope.reject_outliers(
        cross_distance, output, settings_transects
    )
    # plot time-series
    SDS_slope.plot_cross_distance(
        filepath_data, sitename, output["dates"], cross_distance
    )

    # slope estimation settings
    days_in_year = 365.2425
    seconds_in_day = 24 * 3600
    settings_slope = {
        "slope_min": 0.01,
        "slope_max": 0.2,
        "delta_slope": 0.005,
        "date_range": [1999, 2019],  # range of dates over which to perform the analysis
        "n_days": 8,  # sampling period [days]
        "n0": 50,  # for Nyquist criterium
        "freqs_cutoff": 1.0 / (seconds_in_day * 30),  # 1 month frequency
        "delta_f": 100
        * 1e-10,  # deltaf for buffer around max peak                                           # True to save some plots of the spectrums
    }

    settings_slope['date_range'] = [pytz.utc.localize(datetime(settings_slope['date_range'][0],5,1)),
                                pytz.utc.localize(datetime(settings_slope['date_range'][1],1,1))]

    beach_slopes = SDS_slope.range_slopes(
        settings_slope["slope_min"],
        settings_slope["slope_max"],
        settings_slope["delta_slope"],
    )

    idx_dates = [np.logical_and(_>settings_slope['date_range'][0],_<settings_slope['date_range'][1]) for _ in output['dates']]
    dates_sat = [output['dates'][_] for _ in np.where(idx_dates)[0]]
    for key in cross_distance.keys():
        cross_distance[key] = cross_distance[key][idx_dates]

    with open(os.path.join(filepath_data, sitename, sitename + '_tide' + '.pkl'), 'rb') as f:
        tide_data = pickle.load(f) 
    tides_sat = tide_data['tide']

    plots.plot_water_levels(filepath_data, sitename, tide_data, dates_sat, tides_sat)

    plots.plot_tide_time_series(filepath_data, sitename, dates_sat, tides_sat)
    t = np.array([_.timestamp() for _ in dates_sat]).astype("float64")
    delta_t = np.diff(t)
    plots.plot_time_step_distribution(
        filepath_data, sitename, delta_t, seconds_in_day, settings_slope
    )

    # find tidal peak frequency
    settings_slope["freqs_max"] = SDS_slope.find_tide_peak(
        filepath_data, sitename, dates_sat, tides_sat, settings_slope
    )

    slope_est = dict([])
    with open(
        os.path.join(filepath_data, sitename, "transects_slope.csv"), mode="w"
    ) as csv_file:
        writer = csv.writer(
            csv_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        writer.writerow(["transect", "slope"])
        for key in cross_distance.keys():
            # remove NaNs
            idx_nan = np.isnan(cross_distance[key])
            dates = [dates_sat[_] for _ in np.where(~idx_nan)[0]]
            tide = tides_sat[~idx_nan]
            composite = cross_distance[key][~idx_nan]
            # apply tidal correction
            tsall = SDS_slope.tide_correct(composite, tide, beach_slopes)
            SDS_slope.plot_spectrum_all(
                filepath_data, sitename, key, dates, composite, tsall, settings_slope
            )
            slope_est[key] = SDS_slope.integrate_power_spectrum(
                filepath_data, sitename, key, dates, tsall, settings_slope
            )
            writer.writerow(["transect" + str(key), slope_est[key]])
            print("Beach slope at transect %s: %.3f" % (key, slope_est[key]))
    return slope_est, tides_sat, dates_sat, cross_distance, output
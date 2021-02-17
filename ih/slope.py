import numpy as np
from coastsat import SDS_slope, SDS_tools
from ih import plots, utils
import pytz
from datetime import datetime
import pandas as pd
import os
import scipy.io as sio
import pickle
import csv
import shapefile
from pyproj import Proj, transform
from os import walk
import numpy as np
from scipy.spatial import distance

def matDatenum2PYDatetime(datenumVec,unitTime = 'D'):
    datetimeVec = pd.to_datetime(datenumVec-719529, unit=unitTime,errors='coerce')
    datetimeNum = datenumVec-719529
    return datetimeVec,datetimeNum

def beach_slope(filepath_data, sitename, slope=None):

    with open(
        os.path.join(filepath_data, sitename, sitename + "_output" + ".pkl"), "rb"
    ) as f:
        output = pickle.load(f)

    sf = shapefile.Reader(os.path.join(filepath_data, sitename, "Perfiles.shp"))
    transects = dict([])

    i = 0
    records = sf.records()
    for shape in sf.shapes():
        transects[str(int(records[i]["ID_perfil"]))] = np.array(shape.points)
        i = i + 1
    del records, sf

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
        "slope_min": 0.005,
        "slope_max": 0.6,
        "delta_slope": 0.005,
        "date_range": [1950, 2050],  # range of dates over which to perform the analysis
        "n_days": 7,  # sampling period [days]
        "n0": 50,  # for Nyquist criterium
        "freqs_cutoff": 1.0 / (seconds_in_day * 30),  # 1 month frequency
        "delta_f": 100
        * 1e-10,  # deltaf for buffer around max peak                                           # True to save some plots of the spectrums
    }

    settings_slope["date_range"] = [
        pytz.utc.localize(datetime(settings_slope["date_range"][0], 5, 1)),
        pytz.utc.localize(datetime(settings_slope["date_range"][1], 1, 1)),
    ]

    beach_slopes = SDS_slope.range_slopes(
        settings_slope["slope_min"],
        settings_slope["slope_max"],
        settings_slope["delta_slope"],
    )

    idx_dates = [
        np.logical_and(
            _ > settings_slope["date_range"][0], _ < settings_slope["date_range"][1]
        )
        for _ in output["dates"]
    ]
    dates_sat = [output["dates"][_] for _ in np.where(idx_dates)[0]]
    tides = []
    slope_est = dict([])
    _, _, filenames = next(walk(os.path.join(filepath_data, "Marea_Astronomica")))
    for key in cross_distance.keys():
        cross_distance[key] = cross_distance[key][idx_dates]
        transect = transects[key]
        startPoint = transects[key][0]
        endPoint = transects[key][1]
        centroidX = startPoint[0]
        centroidY = startPoint[1]
        inProj = Proj(init='epsg:32628')
        outProj = Proj(init='epsg:4326')
        centroidXWgs84,centroidYWgs84 = transform(inProj,outProj,centroidX,centroidY)
        minDist = 1000.0
        tideFile = None
        for filename in filenames:
            y = float(filename[7:14])
            x = float(filename[16:24])
            dist = distance.cdist([(x,y)],[(centroidXWgs84,centroidYWgs84)])
            if dist < minDist:
                minDist = dist
                tideFile = filename
            del dist
        filepath = os.path.join(filepath_data, "Marea_Astronomica", tideFile)
        mat = sio.loadmat(filepath)
        dates_raw = mat['time'].flatten()
        tides_ts = mat["tide"].flatten()

        time_py = matDatenum2PYDatetime(dates_raw,unitTime = 'D')[0]

        dates_ts = []
        for time in time_py:
            dates_ts.append(datetime(time.year, time.month, time.day, time.hour, time.minute, tzinfo=pytz.utc))

        dates_sat = output["dates"]

        # get tide levels corresponding to the time of image acquisition

        tides_sat = SDS_tools.get_closest_datapoint(dates_sat, dates_ts, tides_ts)

        #plots.plot_water_levels(filepath_data, sitename, tide_data, dates_sat, tides_sat)

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

        # remove NaNs
        idx_nan = np.isnan(cross_distance[key])
        dates = [dates_sat[_] for _ in np.where(~idx_nan)[0]]
        tide = tides_sat[~idx_nan]
        composite = cross_distance[key][~idx_nan]
        # apply tidal correction
        tsall = SDS_slope.tide_correct(composite, tide, beach_slopes)
        try:
            SDS_slope.plot_spectrum_all(
                filepath_data, sitename, key, dates, composite, tsall, settings_slope
            )
            slope_est[key] = SDS_slope.integrate_power_spectrum(
                filepath_data, sitename, key, dates, tsall, settings_slope
            )
                
            print("Beach slope at transect %s: %.3f" % (key, slope_est[key]))
        except:
            pass
        del mat, dates_raw, tides_ts, dates_ts, tides_sat, composite, tsall, idx_nan
    with open(
            os.path.join(filepath_data, sitename, "transects_slope.csv"), mode="w"
    ) as csv_file:
        writer = csv.writer(
            csv_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        writer.writerow(["transect", "slope"])
        for key in cross_distance.keys():
            try:
                writer.writerow(["transect" + str(key), slope_est[key]])
            except:
                pass
    return slope_est
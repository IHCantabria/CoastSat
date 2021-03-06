from ih import plots, utils
import numpy as np
import argparse
import sys
import os
import pickle
import shapefile
from coastsat import SDS_transects, SDS_tools
from datetime import datetime
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument(
    "-S",
    "--site",
    help="Sitename (string)",
    default="default",
)

o = parser.parse_args(sys.argv[1:])

filepath_data = os.path.join(os.getcwd(), "data")
sitename = o.site

with open(os.path.join(filepath_data, sitename, sitename + "_output" + ".pkl"), "rb") as f:
    output = pickle.load(f)

sf = shapefile.Reader(os.path.join(filepath_data, sitename, "Perfiles.shp"))
shapes = sf.shapes()
transects = dict([])

for p in range(len(shapes)):
    transects[str(p)] = np.array(shapes[p].points)

inputs = {
        "sitename": sitename,
        "filepath": filepath_data,
    }

settings = utils.settings_for_shoreline_extraction(inputs)
settings["along_dist"] = 25
cross_distance = SDS_transects.compute_intersection(output, transects, settings)

cross_distance_f = (
    cross_distance  # cross_distance_f es la variable cross_distance filtrada
)

import calendar


def getUnixTimestamp(humanTime, dateFormat="%d-%m-%Y %H:%M:%S"):
    # unixstart = getUnixTimestamp(startdate,"%m/%d/%Y %H:%M")
    unixTimestamp = int(
        calendar.timegm(datetime.strptime(humanTime, dateFormat).timetuple())
    )
    return unixTimestamp / (3600 * 24) + 719529


fechas_costa = output["dates"]
fechas = np.zeros(len(fechas_costa))

for fe in range(len(fechas_costa)):
    label = fechas_costa[fe].strftime("%d-%m-%Y %H:%M:%S")
    fechas[fe] = getUnixTimestamp(label, dateFormat="%d-%m-%Y %H:%M:%S")

for key in cross_distance_f.keys():
    window = 5 * 365
    breaks = 0
    yl, yc = utils.disLongCrossRunMean(fechas, cross_distance_f[key], window, breaks)

    filtro = [
        np.nanmean(yc) - 1.5 * np.nanstd(yc),
        np.nanmean(yc) + 1.5 * np.nanstd(yc),
    ]
    pos = np.argwhere((yc <= filtro[0]) | (yc >= filtro[1]))
    cross_distance_f[key][pos] = np.nan

cross_distance = cross_distance_f


time = np.zeros((len(output["dates"]), 6))
a = output["dates"]
for j in range(len(output["dates"])):
    b = a[j]
    time[j, :] = [b.year, b.month, b.day, b.hour, b.minute, b.second]

plots.plot_time_series(filepath_data, sitename, output, cross_distance)

reference_elevation = (
    0  # elevation at which you would like the shoreline time-series to be
)

plots.plot_shorelines_transects(filepath_data, sitename, output, transects)
slope_est = dict([])
filepath = os.path.join(filepath_data, sitename, sitename + "_tides.csv")
tide_data = pd.read_csv(filepath, parse_dates=["dates"])
dates_ts = [_.to_pydatetime() for _ in tide_data["dates"]]
tides_ts = np.array(tide_data["tide"])
dates_sat = output["dates"]
tides_sat = SDS_tools.get_closest_datapoint(dates_sat, dates_ts, tides_ts)
for key in cross_distance.keys():
    slope_est[key] = 0.1

cross_distance_tidally_corrected = {}
for key in cross_distance.keys():
    correction = (tides_sat - reference_elevation) / slope_est[key]
    cross_distance_tidally_corrected[key] = cross_distance[key] + correction


out_dict = dict([])
out_dict["dates"] = dates_sat
out_dict["geoaccuracy"] = output["geoaccuracy"]
out_dict["satname"] = output["satname"]
for key in cross_distance_tidally_corrected.keys():
    out_dict["Transect " + str(key)] = cross_distance_tidally_corrected[key]
df = pd.DataFrame(out_dict)
fn = os.path.join(
    settings["inputs"]["filepath"],
    settings["inputs"]["sitename"],
    "transect_time_series_tidally_filtered.csv",
)
df.to_csv(fn, sep=",")
print(
    "Tidally-corrected time-series of the shoreline change along the transects saved as:\n%s"
    % fn
)

plots.plot_time_series_shoreline_change(
    filepath_data, sitename, cross_distance, output, cross_distance_tidally_corrected
)
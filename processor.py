# -*- coding: utf-8 -*-
"""
Created on Mon Nov 2 08:47:21 2020

@author: vegama
"""
# Procesador
import os
import numpy as np
import pickle
import warnings
import argparse
import sys
import shapefile

warnings.filterwarnings("ignore")
from coastsat import (
    SDS_download,
    SDS_preprocess,
    SDS_shoreline,
    SDS_tools,
    SDS_transects,
    SDS_slope,
)
import scipy.io as sio
import pandas as pd
import csv
from ih import plots, slope, utils

parser = argparse.ArgumentParser()

parser.add_argument(
    "-S",
    "--site",
    help="Sitename (string)",
    default="default",
)

parser.add_argument(
    "-s",
    "--start",
    help="Start date (yyyy-mm-dd) (string)",
    default="default",
)

parser.add_argument(
    "-e",
    "--end",
    help="End date (yyyy-mm-dd) (string)",
    default="default",
)

parser.add_argument(
    "-m",
    "--mode",
    help="1-Coast Detection, 2-Slope, 3-Full (integer)",
    default=1,
)

parser.add_argument(
    "--slope",
    help="Default slope value (double)",
    default="default",
)

o = parser.parse_args(sys.argv[1:])

slope_value = None

if o.slope != "default":
    slope_value = float(o.slope)

if o.site != "default" and o.start != "default" and o.end != "default":
    filepath_data = os.path.join(os.getcwd(), "data")
    sitename = o.site
    metadata = []
    kml_polygon = os.path.join(filepath_data, sitename, sitename + ".kml")
    polygon = SDS_tools.polygon_from_kml(kml_polygon)

    dates = [o.start, o.end]
    sat_list = ["L5", "L7", "L8", "S2"]

    pts_sl = np.expand_dims(np.array([np.nan, np.nan]), axis=0)
    with open(
        os.path.join(filepath_data, sitename, sitename + "_shoreline.csv")
    ) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            item = np.array([float(row[0]), float(row[1])])
            pts_sl = np.vstack((pts_sl, item))
    pts_sl = np.delete(pts_sl, 0, axis=0)

    pts_world_interp = utils.get_interpolate_points(pts_sl)

    with open(
        os.path.join(filepath_data, sitename, sitename + "_reference_shoreline.pkl"),
        "wb",
    ) as f:
        pickle.dump(pts_world_interp, f)

    inputs = {
        "polygon": polygon,
        "dates": dates,
        "sat_list": sat_list,
        "sitename": sitename,
        "filepath": filepath_data,
    }

    settings = utils.settings_for_shoreline_extraction(inputs)

    if int(o.mode) == 1 or int(o.mode) == 3:
        metadata = SDS_download.retrieve_images(inputs)

        # settings for the shoreline extraction

        SDS_preprocess.save_jpg(metadata, settings)
        # [OPTIONAL] create a reference shoreline (helps to identify outliers and false detections)
        settings["reference_shoreline"] = SDS_preprocess.get_reference_sl(
            metadata, settings
        )
        # set the max distance (in meters) allowed from the reference shoreline for a detected shoreline to be valid
        settings["max_dist_ref"] = 100

        # extract shorelines from all images (also saves output.pkl and shorelines.kml)
        output = SDS_shoreline.extract_shorelines(metadata, settings)
        output = SDS_tools.remove_duplicates(
            output
        )  # removes duplicates (images taken on the same date by the same satellite)
        output = SDS_tools.remove_inaccurate_georef(output, 10)

        # for GIS applications, save output into a GEOJSON layer
        geomtype = "points"  # choose 'points' or 'lines' for the layer geometry
        gdf = SDS_tools.output_to_gdf(output, geomtype)
        gdf.crs = {
            "init": "epsg:" + str(settings["output_epsg"])
        }  # set layer projection
        # save GEOJSON layer to file
        gdf.to_file(
            os.path.join(
                filepath_data, sitename, "%s_output_%s.geojson" % (sitename, geomtype)
            ),
            driver="GeoJSON",
            encoding="utf-8",
        )

        plots.plot_shorelines(filepath_data, sitename, output)

    if int(o.mode) == 2 or int(o.mode) == 3:

        # if you have already mapped the shorelines, load the output.pkl file
        filepath = os.path.join(inputs["filepath"], sitename)
        with open(os.path.join(filepath, sitename + "_output" + ".pkl"), "rb") as f:
            output = pickle.load(f)

        sf = shapefile.Reader(os.path.join(filepath_data, sitename, "Perfiles.shp"))
        transects = dict([])

        i = 0
        records = sf.records()
        for shape in sf.shapes():
            print(polygon)
            transects[str(int(records[i]["ID_perfil"]))] = np.array(shape.points)
            i = i + 1

        settings["along_dist"] = 25
        cross_distance = SDS_transects.compute_intersection(output, transects, settings)

        # pasamos las fechas a un vector con año, mes, dia, hora, minuto y segundo
        time = np.zeros((len(output["dates"]), 6))
        a = output["dates"]
        for j in range(len(output["dates"])):
            b = a[j]
            time[j, :] = [b.year, b.month, b.day, b.hour, b.minute, b.second]

        plots.plot_time_series(filepath_data, sitename, output, cross_distance)

        reference_elevation = (
            0  # elevation at which you would like the shoreline time-series to be
        )

        if slope_value == None:
            slope_est, tides_sat, dates_sat, cross_distance, output = slope.beach_slope(
                filepath_data, sitename
            )
        else:
            plots.plot_shorelines_transects(filepath_data, sitename, output, transects)
            slope_est = dict([])
            filepath = os.path.join(filepath_data, sitename, sitename + "_tides.csv")
            tide_data = pd.read_csv(filepath, parse_dates=["dates"])
            dates_ts = [_.to_pydatetime() for _ in tide_data["dates"]]
            tides_ts = np.array(tide_data["tide"])
            dates_sat = output["dates"]
            tides_sat = SDS_tools.get_closest_datapoint(dates_sat, dates_ts, tides_ts)
            for key in cross_distance.keys():
                slope_est[key] = slope_value

        cross_distance_tidally_corrected = {}
        for key in cross_distance.keys():
            correction = (tides_sat - reference_elevation) / slope_est[key]
            cross_distance_tidally_corrected[key] = cross_distance[key] + correction

        # store the tidally-corrected time-series in a .csv file
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
            "transect_time_series_tidally_corrected.csv",
        )
        df.to_csv(fn, sep=",")
        print(
            "Tidally-corrected time-series of the shoreline change along the transects saved as:\n%s"
            % fn
        )

        plots.plot_time_series_shoreline_change(
            filepath_data,
            sitename,
            cross_distance,
            output,
            cross_distance_tidally_corrected,
        )
else:
    print(
        "Arguments site, start date and end date are mandatory, use python processor.py -h to get help"
    )

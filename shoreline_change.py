import matplotlib.pyplot as plt
from matplotlib import gridspec
import csv
import os
from os import walk
import pickle
import pandas as pd
import numpy as np
import shapefile

def plot_time_series_shoreline_change(
        filepath_data, sitename, cross_distance, output, cross_distance_tidally_corrected, key
    ):
    # plot the time-series of shoreline change (both raw and tidally-corrected)
    fig = plt.figure(figsize=[15, 8], tight_layout=True)
    gs = gridspec.GridSpec(len(cross_distance), 1)
    gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.05)
    ax = fig.add_subplot(111)
    ax.grid(linestyle=":", color="0.5")
    ax.set_ylim([-50, 50])
    ax.plot(
        output["dates"],
        cross_distance - np.nanmedian(cross_distance),
        "-o",
        ms=6,
        mfc="w",
        linestyle = 'None',
        label="raw",
    )
    ax.plot(
        output["dates"],
        cross_distance_tidally_corrected
        - np.nanmedian(cross_distance),
        "-o",
        ms=6,
        mfc="w",
        linestyle = 'None',
        label="tidally-corrected",
    )
    ax.set_ylabel("distance [m]", fontsize=12)
    ax.text(
        0.5,
        0.95,
        key,
        bbox=dict(boxstyle="square", ec="k", fc="w"),
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=14,
    )
    ax.legend()
    fig.savefig(
        os.path.join(
            filepath_data,
            sitename,
            "shoreline_change",
            "time_series_shoreline_change_transect_{0}.png".format(str(key)),
        )
    )

filepath_data = os.path.join(os.getcwd(), "data")
_, sites, filenames = next(walk(os.path.join(filepath_data)))
for sitename in sites:
    print(sitename)
    if not os.path.exists(os.path.join(filepath_data, sitename, sitename + "_output.pkl")):
        print("No exists output.pkl")
        continue
    if not os.path.exists(os.path.join(filepath_data, sitename, "transect_time_series.csv")):
        print("No exists transect_time_series.csv")
        continue
    if not os.path.exists(os.path.join(filepath_data, sitename, "transect_time_series_tidally_corrected.csv")):
        print("No exists transect_time_series_tidally_corrected.csv")
        continue

    sf = shapefile.Reader(os.path.join(filepath_data, sitename, "Perfiles.shp"))
    transects = dict([])

    i = 0
    records = sf.records()
    for shape in sf.shapes():
        transects[str(int(records[i]["ID_perfil"]))] = np.array(shape.points)
        i = i + 1
        
    if not os.path.exists(os.path.join(filepath_data, sitename, "shoreline_change")):
        os.mkdir(os.path.join(filepath_data, sitename, "shoreline_change"))
    with open(os.path.join(filepath_data, sitename, sitename + "_output" + ".pkl"), "rb") as f:
        output = pickle.load(f)

    cross_distance = pd.read_csv(os.path.join(filepath_data, sitename, "transect_time_series.csv"))
    cross_distance_tidally_corrected = pd.read_csv(os.path.join(filepath_data, sitename, "transect_time_series_tidally_corrected.csv"))
    
    for key in transects.keys(): 
        try:
            value_cross_distance = cross_distance["Transect {0}".format(key)]
            value_cross_distance_tidally_corrected = cross_distance_tidally_corrected["Transect {0}".format(key)]
            plot_time_series_shoreline_change(filepath_data, sitename, value_cross_distance, output, value_cross_distance_tidally_corrected, key)
        except:
            pass


    
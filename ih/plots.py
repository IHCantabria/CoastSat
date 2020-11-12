import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import numpy as np
from coastsat import SDS_slope


def plot_shorelines(filepath_data, sitename, output):
    # plot the mapped shorelines
    fig = plt.figure(figsize=[15, 8], tight_layout=True)
    plt.axis("equal")
    plt.xlabel("Eastings")
    plt.ylabel("Northings")
    plt.grid(linestyle=":", color="0.5")
    for i in range(len(output["shorelines"])):
        sl = output["shorelines"][i]
        date = output["dates"][i]
        plt.plot(sl[:, 0], sl[:, 1], ".", label=date.strftime("%d-%m-%Y"))
    plt.legend()
    fig.savefig(os.path.join(filepath_data, sitename, "shorelines.png"))


def plot_time_series(filepath_data, sitename, output, cross_distance):
    fig = plt.figure(figsize=[15, 8], tight_layout=True)
    gs = gridspec.GridSpec(len(cross_distance), 1)
    gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.05)
    for i, key in enumerate(cross_distance.keys()):
        if np.all(np.isnan(cross_distance[key])):
            continue
        ax = fig.add_subplot(gs[i, 0])
        ax.grid(linestyle=":", color="0.5")
        ax.set_ylim([-50, 50])
        ax.plot(
            output["dates"],
            cross_distance[key] - np.nanmedian(cross_distance[key]),
            "-o",
            ms=6,
            mfc="w",
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
    fig.savefig(os.path.join(filepath_data, sitename, "times_series.png"))


def plot_time_series_shoreline_change(
    filepath_data, sitename, cross_distance, output, cross_distance_tidally_corrected
):
    # plot the time-series of shoreline change (both raw and tidally-corrected)
    fig = plt.figure(figsize=[15, 8], tight_layout=True)
    gs = gridspec.GridSpec(len(cross_distance), 1)
    gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.05)
    for i, key in enumerate(cross_distance.keys()):
        if np.all(np.isnan(cross_distance[key])):
            continue
        ax = fig.add_subplot(gs[i, 0])
        ax.grid(linestyle=":", color="0.5")
        ax.set_ylim([-50, 50])
        ax.plot(
            output["dates"],
            cross_distance[key] - np.nanmedian(cross_distance[key]),
            "-o",
            ms=6,
            mfc="w",
            label="raw",
        )
        ax.plot(
            output["dates"],
            cross_distance_tidally_corrected[key] - np.nanmedian(cross_distance[key]),
            "-o",
            ms=6,
            mfc="w",
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
        os.path.join(filepath_data, sitename, "time_series_shoreline_change.png")
    )


def plot_water_levels(filepath_data, sitename, tide_data, dates_sat, tides_sat):
    # plot the subsampled tide data
    fig, ax = plt.subplots(1, 1, figsize=(15, 4), tight_layout=True)
    ax.grid(which="major", linestyle=":", color="0.5")
    ax.plot(
        tide_data["dates"], tide_data["tide"], "-", color="0.6", label="all time-series"
    )
    ax.plot(
        dates_sat,
        tides_sat,
        "-o",
        color="k",
        ms=6,
        mfc="w",
        lw=1,
        label="image acquisition",
    )
    ax.set(
        ylabel="tide level [m]",
        xlim=[dates_sat[0], dates_sat[-1]],
        title="Water levels at the time of image acquisition",
    )
    ax.legend()
    fig.savefig(os.path.join(filepath_data, sitename, "water_levels.png"))


def plot_time_step_distribution(
    filepath_data, sitename, delta_t, seconds_in_day, settings_slope
):
    fig, ax = plt.subplots(1, 1, figsize=(12, 3), tight_layout=True)
    ax.grid(which="major", linestyle=":", color="0.5")
    bins = (
        np.arange(
            np.min(delta_t) / seconds_in_day, np.max(delta_t) / seconds_in_day + 1, 1
        )
        - 0.5
    )
    ax.hist(delta_t / seconds_in_day, bins=bins, ec="k", width=1)
    ax.set(
        xlabel="timestep [days]",
        ylabel="counts",
        xticks=settings_slope["n_days"] * np.arange(0, 20),
        xlim=[0, 50],
        title="Timestep distribution",
    )
    fig.savefig(os.path.join(filepath_data, sitename, "time_step_distribution.png"))


def plot_tide_time_series(filepath_data, sitename, dates_sat, tides_sat):
    fig, ax = plt.subplots(1, 1, figsize=(12, 3), tight_layout=True)
    ax.set_title("Sub-sampled tide levels")
    ax.grid(which="major", linestyle=":", color="0.5")
    ax.plot(dates_sat, tides_sat, "-o", color="k", ms=4, mfc="w", lw=1)
    ax.set_ylabel("tide level [m]")
    ax.set_ylim(SDS_slope.get_min_max(tides_sat))
    fig.savefig(os.path.join(filepath_data, sitename, "tide_time_series.png"))


def plot_shorelines_transects(filepath_data, sitename, output, transects):
    fig, ax = plt.subplots(1, 1, figsize=[12, 8])
    fig.set_tight_layout(True)
    ax.axis("equal")
    ax.set(xlabel="Eastings", ylabel="Northings", title=sitename)
    ax.grid(linestyle=":", color="0.5")
    for i in range(len(output["shorelines"])):
        coords = output["shorelines"][i]
        date = output["dates"][i]
        ax.plot(coords[:, 0], coords[:, 1], ".", label=date.strftime("%d-%m-%Y"))
    for key in transects.keys():
        ax.plot(transects[key][:, 0], transects[key][:, 1], "k--", lw=2)
        ax.text(transects[key][-1, 0], transects[key][-1, 1], key)

    fig.savefig(os.path.join(filepath_data, sitename, "shorelines_transects.png"))
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 08:47:21 2019

@author: alvcuestam
"""
#Procesador
import os
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from coastsat import SDS_download, SDS_preprocess, SDS_shoreline, SDS_tools, SDS_transects, SDS_slope
import scipy.io as sio
import pandas as pd
import pytz
from datetime import datetime

def plot_shorelines(filepath_data, sitename, output):
    # plot the mapped shorelines
    fig = plt.figure(figsize=[15,8], tight_layout=True)
    plt.axis('equal')
    plt.xlabel('Eastings')
    plt.ylabel('Northings')
    plt.grid(linestyle=':', color='0.5')
    for i in range(len(output['shorelines'])):
        sl = output['shorelines'][i]
        date = output['dates'][i]
        plt.plot(sl[:,0], sl[:,1], '.', label=date.strftime('%d-%m-%Y'))
    plt.legend() 
    fig.savefig(os.path.join(filepath_data, sitename, "shorelines.png"))

def plot_time_series(filepath_data, sitename, output, cross_distance):
    fig = plt.figure(figsize=[15,8], tight_layout=True)
    gs = gridspec.GridSpec(len(cross_distance),1)
    gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.05)
    for i,key in enumerate(cross_distance.keys()):
        if np.all(np.isnan(cross_distance[key])):
            continue
        ax = fig.add_subplot(gs[i,0])
        ax.grid(linestyle=':', color='0.5')
        ax.set_ylim([-50,50])
        ax.plot(output['dates'], cross_distance[key]- np.nanmedian(cross_distance[key]), '-o', ms=6, mfc='w')
        ax.set_ylabel('distance [m]', fontsize=12)
        ax.text(0.5,0.95, key, bbox=dict(boxstyle="square", ec='k',fc='w'), ha='center',
                va='top', transform=ax.transAxes, fontsize=14)
    fig.savefig(os.path.join(filepath_data, sitename, "times_series.png"))

def plot_time_series_shoreline_change(filepath_data, sitename, cross_distance, output):
    # plot the time-series of shoreline change (both raw and tidally-corrected)
    fig = plt.figure(figsize=[15,8], tight_layout=True)
    gs = gridspec.GridSpec(len(cross_distance),1)
    gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.05)
    for i,key in enumerate(cross_distance.keys()):
        if np.all(np.isnan(cross_distance[key])):
            continue
        ax = fig.add_subplot(gs[i,0])
        ax.grid(linestyle=':', color='0.5')
        ax.set_ylim([-50,50])
        ax.plot(output['dates'], cross_distance[key]- np.nanmedian(cross_distance[key]), '-o', ms=6, mfc='w', label='raw')
        ax.plot(output['dates'], cross_distance_tidally_corrected[key]- np.nanmedian(cross_distance[key]), '-o', ms=6, mfc='w', label='tidally-corrected')
        ax.set_ylabel('distance [m]', fontsize=12)
        ax.text(0.5,0.95, key, bbox=dict(boxstyle="square", ec='k',fc='w'), ha='center',
                va='top', transform=ax.transAxes, fontsize=14)
    ax.legend()
    fig.savefig(os.path.join(filepath_data, sitename, "time_series_shoreline_change.png"))

def plot_water_levels(filepath_data, sitename, dates_sat, tides_sat):
    # plot the subsampled tide data
    fig, ax = plt.subplots(1,1,figsize=(15,4), tight_layout=True)
    ax.grid(which='major', linestyle=':', color='0.5')
    ax.plot(tide_data['dates'], tide_data['tide'], '-', color='0.6', label='all time-series')
    ax.plot(dates_sat, tides_sat, '-o', color='k', ms=6, mfc='w',lw=1, label='image acquisition')
    ax.set(ylabel='tide level [m]',xlim=[dates_sat[0],dates_sat[-1]], title='Water levels at the time of image acquisition')
    ax.legend()
    fig.savefig(os.path.join(filepath_data, sitename, "water_levels.png"))

def plot_time_step_distribution(filepath_data, sitename, delta_t, seconds_in_day, settings_slope):
    fig, ax = plt.subplots(1,1,figsize=(12,3), tight_layout=True)
    ax.grid(which='major', linestyle=':', color='0.5')
    bins = np.arange(np.min(delta_t)/seconds_in_day, np.max(delta_t)/seconds_in_day+1,1)-0.5
    ax.hist(delta_t/seconds_in_day, bins=bins, ec='k', width=1)
    ax.set(xlabel='timestep [days]', ylabel='counts',
        xticks=settings_slope['n_days']*np.arange(0,20),
        xlim=[0,50], title='Timestep distribution')
    fig.savefig(os.path.join(filepath_data, sitename, "time_step_distribution.png"))

def plot_tide_time_series(filepath_data, sitename, dates_sat, tides_sat):
    fig, ax = plt.subplots(1,1,figsize=(12,3), tight_layout=True)
    ax.set_title('Sub-sampled tide levels')
    ax.grid(which='major', linestyle=':', color='0.5')
    ax.plot(dates_sat, tides_sat, '-o', color='k', ms=4, mfc='w',lw=1)
    ax.set_ylabel('tide level [m]')
    ax.set_ylim(SDS_slope.get_min_max(tides_sat))
    fig.savefig(os.path.join(filepath_data, sitename, "tide_time_series.png"))

def plot_shorelines_transects(filepath_data, sitename, output, transects):
    fig,ax = plt.subplots(1,1,figsize=[12,  8])
    fig.set_tight_layout(True)
    ax.axis('equal')
    ax.set(xlabel='Eastings', ylabel='Northings', title=sitename)
    ax.grid(linestyle=':', color='0.5')
    for i in range(len(output['shorelines'])):
        coords = output['shorelines'][i]
        date = output['dates'][i]
        ax.plot(coords[:,0], coords[:,1], '.', label=date.strftime('%d-%m-%Y'))
    for key in transects.keys():
        ax.plot(transects[key][:,0],transects[key][:,1],'k--',lw=2)
        ax.text(transects[key][-1,0], transects[key][-1,1], key)

    fig.savefig(os.path.join(filepath_data, sitename, "shorelines_transects.png"))

def beach_slope(filepath_data, sitename, output, transects):
    # remove S2 shorelines (the slope estimation algorithm needs only Landsat)
    if 'S2' in output['satname']:
        idx_S2 = np.array([_ == 'S2' for _ in output['satname']])
        for key in output.keys():
            output[key] = [output[key][_] for _ in np.where(~idx_S2)[0]]

    # remove duplicates 
    output = SDS_slope.remove_duplicates(output)
    # remove shorelines from images with poor georeferencing (RMSE > 10 m)
    output = SDS_slope.remove_inaccurate_georef(output, 10)

    # plot shorelines and transects
    plot_shorelines_transects(filepath_data, sitename, output, transects)

    # a more robust method to compute intersection is needed here to avoid outliers
    # as these can affect the slope detection algorithm
    settings_transects = { # parameters for shoreline intersections
                        'along_dist':         25,             # along-shore distance to use for intersection
                        'max_std':            15,             # max std for points around transect
                        'max_range':          30,             # max range for points around transect
                        'min_val':            -100,           # largest negative value along transect (landwards of transect origin)
                        # parameters for outlier removal
                        'nan/max':            'auto',         # mode for removing outliers ('auto', 'nan', 'max')
                        'prc_std':            0.1,            # percentage to use in 'auto' mode to switch from 'nan' to 'max'
                        'max_cross_change':   40,        # two values of max_cross_change distance to use
                        }
    # compute intersections [advanced version]
    cross_distance = SDS_slope.compute_intersection(output, transects, settings_transects) 
    # remove outliers [advanced version]
    cross_distance = SDS_slope.reject_outliers(cross_distance,output,settings_transects)        
    # plot time-series
    SDS_slope.plot_cross_distance(output['dates'],cross_distance)
        
    # slope estimation settings
    days_in_year = 365.2425
    seconds_in_day = 24*3600
    settings_slope = {'slope_min':        0.035,
                    'slope_max':        0.2, 
                    'delta_slope':      0.005,
                    'date_range':       [1999,2020],            # range of dates over which to perform the analysis
                    'n_days':           8,                      # sampling period [days]
                    'n0':               50,                     # for Nyquist criterium
                    'freqs_cutoff':     1./(seconds_in_day*30), # 1 month frequency
                    'delta_f':          100*1e-10,              # deltaf for buffer around max peak                                           # True to save some plots of the spectrums
                    }
    settings_slope['date_range'] = [pytz.utc.localize(datetime(settings_slope['date_range'][0],5,1)),
                                    pytz.utc.localize(datetime(settings_slope['date_range'][1],1,1))]
    beach_slopes = SDS_slope.range_slopes(settings_slope['slope_min'], settings_slope['slope_max'], settings_slope['delta_slope'])

    # clip the dates between 1999 and 2020 as we need at least 2 Landsat satellites 
    idx_dates = [np.logical_and(_>settings_slope['date_range'][0],_<settings_slope['date_range'][1]) for _ in output['dates']]
    dates_sat = [output['dates'][_] for _ in np.where(idx_dates)[0]]
    for key in cross_distance.keys():
        cross_distance[key] = cross_distance[key][idx_dates]

    plot_tide_time_series(filepath_data, sitename, dates_sat, tides_sat)
    t = np.array([_.timestamp() for _ in dates_sat]).astype('float64')
    delta_t = np.diff(t)
    plot_time_step_distribution(filepath_data, sitename, delta_t, seconds_in_day, settings_slope)

    # find tidal peak frequency
    settings_slope['freqs_max'] = SDS_slope.find_tide_peak(dates_sat,tides_sat,settings_slope)

    slope_est = dict([])
    for key in cross_distance.keys():
        # remove NaNs
        idx_nan = np.isnan(cross_distance[key])
        dates = [dates_sat[_] for _ in np.where(~idx_nan)[0]]
        tide = tides_sat[~idx_nan]
        composite = cross_distance[key][~idx_nan]
        # apply tidal correction
        tsall = SDS_slope.tide_correct(composite,tide,beach_slopes)
        SDS_slope.plot_spectrum_all(dates,composite,tsall,settings_slope)
        slope_est[key] = SDS_slope.integrate_power_spectrum(dates,tsall,settings_slope)
        print('Beach slope at transect %s: %.3f'%(key, slope_est[key]))
    return slope_est



filepath_data = os.path.join(os.getcwd(), 'data')
sitename='REYA'
metadata=[]
kml_polygon = os.path.join(filepath_data, sitename, sitename + ".kml")
polygon = SDS_tools.polygon_from_kml(kml_polygon)
dates = ['2018-06-01', '2019-03-01']
sat_list = ['L8']

inputs = {
        'polygon': polygon,
        'dates': dates,
        'sat_list': sat_list,
        'sitename': sitename,
        'filepath': filepath_data
}
    
metadata = SDS_download.retrieve_images(inputs)
    
# settings for the shoreline extraction
settings = { 
    # general parameters:
    'cloud_thresh': 0.0,        # threshold on maximum cloud cover
    'output_epsg': 32630,       # epsg code of spatial reference system desired for the output   
    # quality control:
    'check_detection': False,    # if True, shows each shoreline detection to the user for validation
    'save_figure': True,        # if True, saves a figure showing the mapped shoreline for each image
    'adjust_detection': False,
    # add the inputs defined previously
    'inputs': inputs,
    # [ONLY FOR ADVANCED USERS] shoreline detection parameters:
    'min_beach_area': 4500,     # minimum area (in metres^2) for an object to be labelled as a beach
    'buffer_size': 150,         # radius (in metres) of the buffer around sandy pixels considered in the shoreline detection
    'min_length_sl': 200,       # minimum length (in metres) of shoreline perimeter to be valid
    'cloud_mask_issue': False,  # switch this parameter to True if sand pixels are masked (in black) on many images  
    'sand_color': 'default',    # 'default', 'dark' (for grey/black sand beaches) or 'bright' (for white sand beaches)
}

SDS_preprocess.save_jpg(metadata, settings)
# [OPTIONAL] create a reference shoreline (helps to identify outliers and false detections)
settings['reference_shoreline'] = SDS_preprocess.get_reference_sl(metadata, settings)
# set the max distance (in meters) allowed from the reference shoreline for a detected shoreline to be valid
settings['max_dist_ref'] = 100     

# extract shorelines from all images (also saves output.pkl and shorelines.kml)
output = SDS_shoreline.extract_shorelines(metadata, settings)

plot_shorelines(filepath_data, sitename, output)

# if you have already mapped the shorelines, load the output.pkl file
filepath = os.path.join(inputs['filepath'], sitename)
with open(os.path.join(filepath, sitename + '_output' + '.pkl'), 'rb') as f:
    output = pickle.load(f) 
perf=sio.loadmat(os.path.join(filepath_data, sitename, 'coordinatesPerf.mat'))

# option 3: create the transects by manually providing the coordinates of two points 
transects = dict([])
rang=[0,14]
settings['along_dist'] = 25
for i in range(rang[0], rang[1]):
    transects[i-rang[0]] = np.array([[perf['coordinatesPerf'][i,0], perf['coordinatesPerf'][i,1]],[perf['coordinatesPerf'][i,2], perf['coordinatesPerf'][i,3]]]) 

cross_distance = SDS_transects.compute_intersection(output, transects, settings) 
# pasamos las fechas a un vector con año, mes, dia, hora, minuto y segundo
time=np.zeros((len(output['dates']),6))
a=output['dates']
for j in range(len(output['dates'])):
    b=a[j]
    time[j,:]=[b.year, b.month, b.day, b.hour, b.minute, b.second]

# guardamos    
for i in range(rang[0],rang[1]):
    buf = "Reya%d.mat"% (i+1)
    buf1='Transect%d'% (i+1)
    buf3='date%d'%(i+1)
    sio.savemat(os.path.join(filepath_data, sitename, buf), {buf1: cross_distance[i-rang[0]], 'data': time})

plot_time_series(filepath_data, sitename, output, cross_distance)

#%% 4. Tidal correction
    
# For this example, measured water levels for Sydney are stored in a csv file located in /examples.
# When using your own file make sure that the dates are in UTC time, as the CoastSat shorelines are also in UTC
# and the datum for the water levels is approx. Mean Sea Level. We assume a beach slope of 0.1 here.

# load the measured tide data
filepath = os.path.join(filepath_data,sitename, sitename + '_tides.csv')
tide_data = pd.read_csv(filepath, parse_dates=['dates'])
dates_ts = [_.to_pydatetime() for _ in tide_data['dates']]
tides_ts = np.array(tide_data['tide'])

# get tide levels corresponding to the time of image acquisition
dates_sat = output['dates']
tides_sat = SDS_tools.get_closest_datapoint(dates_sat, dates_ts, tides_ts)

plot_water_levels(filepath_data, sitename, dates_sat, tides_sat)

# tidal correction along each transect
reference_elevation = 0 # elevation at which you would like the shoreline time-series to be

slope_est = beach_slope(filepath_data, sitename, output, transects)
cross_distance_tidally_corrected = {}
for key in cross_distance.keys():
    correction = (tides_sat-reference_elevation)/slope_est[key]
    cross_distance_tidally_corrected[key] = cross_distance[key] + correction
    
# store the tidally-corrected time-series in a .csv file
out_dict = dict([])
out_dict['dates'] = dates_sat
for key in cross_distance_tidally_corrected.keys():
    out_dict['Transect '+ str(key)] = cross_distance_tidally_corrected[key]
df = pd.DataFrame(out_dict)
fn = os.path.join(settings['inputs']['filepath'],settings['inputs']['sitename'],
                  'transect_time_series_tidally_corrected.csv')
df.to_csv(fn, sep=',')
print('Tidally-corrected time-series of the shoreline change along the transects saved as:\n%s'%fn)

plot_time_series_shoreline_change(filepath_data, sitename, cross_distance, output)


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


filepath_data = os.path.join(os.getcwd(), 'data')
sitename='REYA'
metadata=[]
kml_polygon = os.path.join(filepath_data, sitename, sitename + ".kml")
polygon = SDS_tools.polygon_from_kml(kml_polygon)
dates = ['2019-01-01', '2019-03-01']
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


#%% 4. Shoreline analysis

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
beach_slope = 0.1
cross_distance_tidally_corrected = {}
for key in cross_distance.keys():
    correction = (tides_sat-reference_elevation)/beach_slope
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


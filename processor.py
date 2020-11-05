# -*- coding: utf-8 -*-
"""
Created on Mon Nov 2 08:47:21 2020

@author: vegama
"""
#Procesador
import os
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
from coastsat import SDS_download, SDS_preprocess, SDS_shoreline, SDS_tools, SDS_transects
import scipy.io as sio
import pandas as pd
import csv
from ih import plots, slope, utils

filepath_data = os.path.join(os.getcwd(), 'data')
sitename='Valencia'
metadata=[]
kml_polygon = os.path.join(filepath_data, sitename, sitename + ".kml")
polygon = SDS_tools.polygon_from_kml(kml_polygon)
dates = ['2019-02-01', '2019-03-01']
sat_list = ['L8']

pts_sl = np.expand_dims(np.array([np.nan, np.nan]),axis=0)
with open(os.path.join(filepath_data, sitename, sitename + "_shoreline.csv")) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        item = np.array([float(row[0]), float(row[1])])
        pts_sl = np.vstack((pts_sl, item))
pts_sl = np.delete(pts_sl,0,axis=0)

pts_world_interp = utils.get_interpolate_points(pts_sl)

with open(os.path.join(filepath_data, sitename, sitename + '_reference_shoreline.pkl'), 'wb') as f:
    pickle.dump(pts_world_interp, f)

inputs = {
        'polygon': polygon,
        'dates': dates,
        'sat_list': sat_list,
        'sitename': sitename,
        'filepath': filepath_data
}
    
metadata = SDS_download.retrieve_images(inputs)
    
# settings for the shoreline extraction
settings = utils.settings_for_shoreline_extraction(inputs)

SDS_preprocess.save_jpg(metadata, settings)
# [OPTIONAL] create a reference shoreline (helps to identify outliers and false detections)
settings['reference_shoreline'] = SDS_preprocess.get_reference_sl(metadata, settings)
# set the max distance (in meters) allowed from the reference shoreline for a detected shoreline to be valid
settings['max_dist_ref'] = 100     

# extract shorelines from all images (also saves output.pkl and shorelines.kml)
output = SDS_shoreline.extract_shorelines(metadata, settings)

# for GIS applications, save output into a GEOJSON layer
geomtype = 'lines' # choose 'points' or 'lines' for the layer geometry
gdf = SDS_tools.output_to_gdf(output, geomtype)
gdf.crs = {'init':'epsg:'+str(settings['output_epsg'])} # set layer projection
# save GEOJSON layer to file
gdf.to_file(os.path.join(filepath_data, sitename, '%s_output_%s.geojson'%(sitename,geomtype)),
                                driver='GeoJSON', encoding='utf-8')

plots.plot_shorelines(filepath_data, sitename, output)

# if you have already mapped the shorelines, load the output.pkl file
filepath = os.path.join(inputs['filepath'], sitename)
with open(os.path.join(filepath, sitename + '_output' + '.pkl'), 'rb') as f:
    output = pickle.load(f) 
perf=sio.loadmat(os.path.join(filepath_data, sitename, 'coordinatesPERF.mat'))

# option 3: create the transects by manually providing the coordinates of two points 
transects = dict([])
rang=[0,perf["coordinatesPerf"].shape[0]]
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
    buf = sitename + "%d.mat"% (i+1)
    buf1='Transect%d'% (i+1)
    buf3='date%d'%(i+1)
    sio.savemat(os.path.join(filepath_data, sitename, buf), {buf1: cross_distance[i-rang[0]], 'data': time})

plots.plot_time_series(filepath_data, sitename, output, cross_distance)

# Tidal correction
# load the measured tide data
filepath = os.path.join(filepath_data,sitename, sitename + '_tides.csv')
tide_data = pd.read_csv(filepath, parse_dates=['dates'])
dates_ts = [_.to_pydatetime() for _ in tide_data['dates']]
tides_ts = np.array(tide_data['tide'])

# get tide levels corresponding to the time of image acquisition
dates_sat = output['dates']
tides_sat = SDS_tools.get_closest_datapoint(dates_sat, dates_ts, tides_ts)

plots.plot_water_levels(filepath_data, sitename, tide_data, dates_sat, tides_sat)

# tidal correction along each transect
reference_elevation = 0 # elevation at which you would like the shoreline time-series to be

slope_est = slope.beach_slope(filepath_data, sitename, output, transects, dates_sat, tides_sat)
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

plots.plot_time_series_shoreline_change(filepath_data, sitename, cross_distance, output, cross_distance_tidally_corrected)

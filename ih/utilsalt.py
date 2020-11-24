import numpy as np
import skimage.transform as transform
import calendar
import datetime
import pandas as pd

# Convert to unix timestamp
def getUnixTimestamp(humanTime,dateFormat = "%m/%d/%Y %H:%M"):
    # unixstart = getUnixTimestamp(startdate,"%m/%d/%Y %H:%M")
    unixTimestamp = int(calendar.timegm(datetime.datetime.strptime(humanTime, dateFormat).timetuple()))
    return unixTimestamp


def disLongCrossRunMean(t,y,window,breaks):
  
    if not np.all(breaks == 0):    
        breaksn = getUnixTimestamp(breaks)
        
        # encontrar el t mas cercano a breaks
        breakn = np.zeros(len(breaksn))
        
        for i in range(len(breaksn)):
            dif = np.abs(t - breaksn[i])
            pos = dif[dif == np.min(dif)]
            breakn[i] = pos
            nintervals = len(breakn) + 1
    else:
        nintervals = 1
          
    yl = np.zeros(len(t))
    yc = np.zeros(len(t))

    for i in range(nintervals):
        if not np.all(breaks == 0) and i == 0:
            interval = [1, breakn[0] - 1]
        elif i == 0 and np.all(breaks == 0): 
            interval = [1, len(t)]
        elif i != 0 and i != len(breakn):
            interval = [breakn[i], breakn[i]]
        else:
            interval = [breakn[-1], len[t]]

        for z in range(interval[0]-1, interval[1]):
            ta = np.zeros(len(t))
            ta[:] = np.nan
            ya = np.zeros(len(t))
            ya[:] = np.nan
            ta[(interval[0]-1):(interval[1])] = t[(interval[0]-1):(interval[1])]
            ya[(interval[0]-1):(interval[1])] = y[(interval[0]-1):(interval[1])]
    
            twindow = [ta[z]-window/2, ta[z] + window/2]
            inpoints = np.argwhere((ta > twindow[0]) & (ta < twindow[1]))
            yl[z] = np.nanmean(ya[inpoints])
            yc[z] = y[z] - yl[z]

    return yl, yc


def get_interpolate_points(pts_sl):
    pts_world_interp = np.expand_dims(np.array([np.nan, np.nan]), axis=0)
    for k in range(len(pts_sl) - 1):
        pt_dist = np.linalg.norm(pts_sl[k, :] - pts_sl[k + 1, :])
        xvals = np.arange(0, pt_dist)
        yvals = np.zeros(len(xvals))
        pt_coords = np.zeros((len(xvals), 2))
        pt_coords[:, 0] = xvals
        pt_coords[:, 1] = yvals
        phi = 0
        deltax = pts_sl[k + 1, 0] - pts_sl[k, 0]
        deltay = pts_sl[k + 1, 1] - pts_sl[k, 1]
        phi = np.pi / 2 - np.math.atan2(deltax, deltay)
        tf = transform.EuclideanTransform(rotation=phi, translation=pts_sl[k, :])
        pts_world_interp = np.append(pts_world_interp, tf(pt_coords), axis=0)
    pts_world_interp = np.delete(pts_world_interp, 0, axis=0)
    return pts_world_interp


def settings_for_shoreline_extraction(inputs):
    # settings for the shoreline extraction
    return {
        # general parameters:
        "cloud_thresh": 0.0,  # threshold on maximum cloud cover
        "output_epsg": 3857,  # epsg code of spatial reference system desired for the output
        # quality control:
        "check_detection": False,  # if True, shows each shoreline detection to the user for validation
        "save_figure": True,  # if True, saves a figure showing the mapped shoreline for each image
        "adjust_detection": False,
        # add the inputs defined previously
        "inputs": inputs,
        # [ONLY FOR ADVANCED USERS] shoreline detection parameters:
        "min_beach_area": 4500,  # minimum area (in metres^2) for an object to be labelled as a beach
        "buffer_size": 150,  # radius (in metres) of the buffer around sandy pixels considered in the shoreline detection
        "min_length_sl": 200,  # minimum length (in metres) of shoreline perimeter to be valid
        "cloud_mask_issue": False,  # switch this parameter to True if sand pixels are masked (in black) on many images
        "sand_color": "default",  # 'default', 'dark' (for grey/black sand beaches) or 'bright' (for white sand beaches),
    }

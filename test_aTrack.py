# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 15:31:11 2024

@author: franc
"""

import os
file_PATH = os.path.realpath(__file__)
#print('file_PATH: ', str(__file__))
current_dir = file_PATH.rsplit(os.sep, 1)[0]
print('current_dir: ', current_dir)
os.chdir(current_dir)
print('current dir', os.getcwd()) # get current directory

import anomalous
import numpy as np

path = r'C:\Users\franc\OneDrive\Bureau\Anomalous\example_tracks.csv'
savepath = r'C:\Users\franc\OneDrive\Bureau\Anomalous\saved_results.csv'
length = np.array([99])
Fixed_LocErr = True
Initial_params = {'LocErr': 0.02, 'd': 0.1}
nb_epochs = 400

print(os.path.realpath(__file__))

tracks, frames, opt_metrics = anomalous.read_table(path, # path of the file to read or list of paths to read multiple files.
               lengths = length, # number of positions per track accepted (take the first position if longer than max
               dist_th = np.inf, # maximum distance allowed for consecutive positions 
               frames_boundaries = [-np.inf, np.inf], # min and max frame values allowed for peak detection
               fmt = 'csv', # format of the document to be red, 'csv' or 'pkl', one can also just specify a separator e.g. ' '. 
               colnames = ['POSITION_X', 'POSITION_Y', 'FRAME', 'TRACK_ID'],  # if multiple columns are required to identify a track, the string used to identify the track ID can be replaced by a list of strings represening the column names e.g. ['TRACK_ID', 'Movie_ID']
               opt_colnames = [], # list of additional metrics to collect e.g. ['QUALITY', 'ID']
               remove_no_disp = True)

tracks = tracks['99']

pd_params = anomalous.Brownian_fit(tracks, verbose = 1, Fixed_LocErr = Fixed_LocErr, Initial_params = Initial_params, nb_epochs = nb_epochs)

pd_params.to_csv(savepath)

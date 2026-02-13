# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 16:39:20 2026

@author: franc
"""


import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

# needs the libary extrack
from glob import glob
import pandas as pd
import tensorflow as tf

import sys
sys.path.insert(0, r'C:\Users\franc\OneDrive\Bureau\Anomalous\scripts') # add the path to atrack script (https://github.com/FrancoisSimon/aTrack)
import atrack

'''
Analysis of the SBS dataset (in the datasets folder)
'''
datapath = 'Path/to/the/folder/named/SPB'
lengths = np.arange(100, 101)

all_directed_pd_params = []
all_confined_pd_params = []
all_Brownian_pd_params = []

all_L0s = []
all_L2s = []

conds = np.array(['210303_KWY10328_+be_OD3_DMSO', '210303_KWY10328_+be_OD3_LatA', '210303_KWY10328_+be_OD3_nothing', 
 '221006_KWY10328_+be_DMSO', '221006_KWY10328_+be_LatA', '221006_KWY10328_+be_nothing', '221006_KWY10722_DMSO', '221006_KWY10722_LatA', '221006_KWY10722_nothing',
 '220909_KWY10328_+be_DMSO', '220909_KWY10328_+be_LatA', '220909_KWY10328_+be_nothing', '220909_KWY10722_+be_DMSO', '220909_KWY10722_+be_LatA', '220909_KWY10722_+be_nothing'])

xs = []
for cond in conds:
    x = 0
    if 'nothing' in cond:
        x+=0
    elif 'DMSO' in cond:
        x+=1
    else:
        x+=2
    if 'KWY10722' in cond:
        x+=3
    xs.append(x)
xs = np.array(xs)

names = ['WT', 'WT DMSO', 'WT LatA', 'Mut', 'Mut DMSO', 'Mut LatA']
nb_epochs=1500
cond = conds[1]
track_lens = []
np.mean(track_lens)
for cond in conds[:]:
    print(cond)
    paths = glob(datapath + r"\*\%s*.csv"%cond)
    track_list, frame_list, opt_metric_lists = atrack.read_table(paths, # path of the file to read or list of paths to read multiple files.
                                                           lengths = lengths, # number of positions per track accepted (take the first position if longer than max
                                                           dist_th = np.inf, # maximum distance allowed for consecutive positions 
                                                           frames_boundaries = [-np.inf, np.inf], # min and max frame values allowed for peak detection
                                                           fmt = 'csv', # format of the document to be red, 'csv' or 'pkl', one can also just specify a separator e.g. ' '. 
                                                           colnames = ['POSITION_X', 'POSITION_Y', 'FRAME', 'TRACK_ID'],  # if multiple columns are required to identify a track, the string used to identify the track ID can be replaced by a list of strings represening the column names e.g. ['TRACK_ID', 'Movie_ID']
                                                           opt_colnames = ['QUALITY'], # list of additional metrics to collect e.g. ['QUALITY', 'ID']
                                                           remove_no_disp = True,
                                                           split_long_tracks = True)
    track_lens.append(len(track_list))
    track_list = np.array(track_list).reshape((len(track_list)*5, 20, 2))
    print(len(track_list))

    track_list.shape
    directed_pd_params = atrack.Directed_fit(track_list, verbose = 0, Fixed_LocErr = False, Initial_params = {'LocErr': 0.025, 'd': 0.01, 'q': 0.02, 'l': 0.02}, nb_epochs = nb_epochs, output_type = 'dict', fitting_type = 'All')
    Brownian_pd_params = atrack.Brownian_fit(track_list, verbose = 0, Fixed_LocErr = False, Initial_params = {'LocErr': 0.025, 'd': 0.01, 'q': 0.02, 'l': 0.02}, nb_epochs = nb_epochs, output_type = 'dict', fitting_type = 'All')
    #confined_pd_params = atrack.Confined_fit(track_list, verbose = 0, Fixed_LocErr = False, Initial_params = {'LocErr': 0.02, 'd': 0.01, 'q': 0.0001, 'l': 0.01}, nb_epochs = 1000, output_type = 'dict', fitting_type = 'All')
    
    L0s = np.sum(Brownian_pd_params['Log_likelihood'].values.reshape((len(track_list)//5, 5)), 1)
    L2s = np.sum(directed_pd_params['Log_likelihood'].values.reshape((len(track_list)//5, 5)), 1)
    proba = np.exp(L0s -L2s)
    
    tracks = np.array(track_list).reshape((len(track_list)//5, 100, 2))
    plt.figure(figsize = (25, 25))
    plt.title(cond)
    lim = 0.6
    nb_rows = 15
    for i in range(nb_rows):
        for j in range(nb_rows):
            step = max(1, len(tracks)//nb_rows**2)
            ID = step * i*nb_rows+j
            track = tracks[ID]
            track = track - np.mean(track,0, keepdims = True) + [[lim*i, lim*j]]
            plt.plot(track[:,0], track[:,1], alpha = 1, color = plt.cm.brg(0.5*(proba[ID]>0.05)))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(cond + '.svg')
    
    all_L0s.append(L0s)
    all_L2s.append(L2s)    

# we compute the fraction of the tracks significantly directed according to our metric
fs = []
for cond, L0s, L2s in zip(conds, all_L0s, all_L2s):
    fs.append(np.mean(np.exp(L0s-L2s)<0.05))
fs = np.array(fs)


# Obtained values:
# fs = np.array([0.25319309, 0.10881378, 0.23282313, 0.23596939, 0.09451385, 0.21489971, 0.18415379, 0.19657966, 0.16184374, 0.20485813, 0.12234136, 0.22410465, 0.16055046, 0.16875   , 0.15019763])
# xs = array([1, 2, 0, 1, 2, 0, 4, 5, 3, 1, 2, 0, 4, 5, 3])
width = 0.65
plt.figure(figsize = (2,2.5))
vals = fs[xs ==1]
plt.bar(0, np.mean(vals), yerr = np.std(vals), width = width)
vals = fs[xs ==2]
plt.bar(1, np.mean(vals), yerr = np.std(vals), width = width)
vals = fs[xs ==4]
plt.bar(2, np.mean(vals), yerr = np.std(vals), width = width)
vals = fs[xs ==5]
plt.bar(3, np.mean(vals), yerr = np.std(vals), width = width)
plt.ylabel('Fraction of directed tracks')
plt.xticks([0,1,2,3], ["WT - LatA", "WT + LatA", "Mut - LatA", "Mut + LatA"], rotation = 70)
plt.xlim([-0.7, 3.7])
plt.ylim([0, 0.3])

# test of the significance of the the differences of directed motion fraction
import scipy
scipy.stats.ttest_ind(fs[xs == 1], fs[xs == 2]) # test WT DMSO vs WT LatA
scipy.stats.ttest_ind(fs[xs == 4], fs[xs == 5]) # test Mut DMSO vs Mut LatA



'''
Bacteria motion analysis
'''


path = r'path/to/the/bacteria/track/file/named/becteria_tracks.csv'

tracks, frames, opt_metrics = atrack.read_table(path, # path of the file to read or list of paths.
                                        lengths = np.arange(50,51), # number of positions per track accepted.
                                        dist_th = np.inf, # Maximum distance allowed for consecutive positions.
                                        frames_boundaries = [0, np.inf], # Minimum and maximum frames to consider.
                                        fmt = 'csv', # Format of the document to be red, 'csv' or 'pkl'.
                                        colnames = ['POSITION_X', 'POSITION_Y', 'FRAME', 'TRACK_ID'], 
                                        opt_colnames = [], # list of additional metrics to collect.
                                        remove_no_disp = True)
print('Number of tracks:', len(tracks))
tracks = np.array(tracks)


lim = 40.
plt.figure(figsize = (10,10))
for i in range(5):
    for j in range(5):
        #plt.subplot(3,3,i*3+j+1)
        track = tracks[i*5+j]
        plt.plot(track[:,0] - np.mean(track[:,0]) + lim*i, track[:,1] - np.mean(track[:,1]) +lim*j, color = 'k')
plt.gca().set_aspect('equal', adjustable='box')

Fixed_LocErr = False
nb_epochs = 1000
Initial_confined_params = {'LocErr': 0.02, 'd': 0.1, 'q': 0.01, 'l': 0.01}
Initial_directed_params = {'LocErr': 0.02, 'd': 0.1, 'q': 0.01, 'l': 0.01}
nb_dims = tracks.shape[2]
verbose = 1

# single-track approach
est_LocErrs, est_ds, est_qs, est_ls, LP = atrack.Confined_fit(tracks, verbose = verbose, Fixed_LocErr = Fixed_LocErr, Initial_params = Initial_confined_params, nb_epochs = nb_epochs, output_type = '')
est_LocErrs2, est_ds2, est_qs2, est_ls2, LP2, pred_kis = atrack.Directed_fit(tracks, verbose = verbose, Fixed_LocErr = Fixed_LocErr, Initial_params = Initial_directed_params, nb_epochs = nb_epochs, output_type = '')

mask = ((LP2 - LP) > np.log(1000)).astype(int)[:,0]

cs = ['r', 'b']

nb = 9

lim = 50.
plt.figure(figsize = (10,10))
for i in range(nb):
    for j in range(nb):
        #plt.subplot(3,3,i*3+j+1)
        track = tracks[i*nb+j]
        plt.plot(track[:,0] - np.mean(track[:,0]) + lim*i, track[:,1] - np.mean(track[:,1]) +lim*j, color = cs[mask[i*nb+j]])
plt.gca().set_aspect('equal', adjustable='box')

plt.figure()
plt.scatter(np.arange(len(LP)), (LP2 - LP))
plt.ylabel('Log likelihood difference')
plt.plot([0, 85], [-np.log(0.05/85), -np.log(0.05/85)], label = 'Threshold')


# population wise approach:
with tf.device('/cpu:0'):
    likelihoods, all_pd_params = atrack.multi_fit(tracks, verbose = 1, Fixed_LocErr = True, min_nb_states = 1, max_nb_states = 10, nb_epochs = 1000, batch_size = 2**11,
                   Initial_confined_params = {'LocErr': 0.1, 'd': 0.5, 'q': 0.01, 'l': 0.1},
                   Initial_directed_params = {'LocErr': 0.1, 'd': 0.5, 'q': 0.01, 'l': 0.1},
                   fitting_type = 'All')
likelihoods_values = likelihoods['Likelihood'].values
plt.figure()
plt.plot(likelihoods['number_of_states'], likelihoods_values - likelihoods_values[-1])
plt.ylabel('Normalized log likelihood') # [$\mathrm{\mu}$m]
plt.xlabel('Nubmer of states')

final_d = all_pd_params['5']['d'].values.astype(float)
final_l = all_pd_params['5']['l'].values.astype(float)
final_fractions = all_pd_params['5']['fraction'].values.astype(float)

plt.figure()
plt.scatter(final_l, final_d, s = 8)
plt.ylabel('Diffusion length [pixel]') # [$\mathrm{\mu}$m]
plt.xlabel('Velocity [pixel/time step]')


'''
Trapping analysis
'''

paths = glob(r"path/to/datasets/Trapping/*.csv")
paths = paths[1:] + paths[:1]

laser_intensities = np.array([15, 20, 30, 50, 60, 70, 80, 90, 100])

all_tracks = [] 
for path in paths:
    tracks, frames, opt_metrics = atrack.read_table(path, # path of the file to read or list of paths.
                                                    lengths = np.arange(6000,6001), # number of positions per track accepted.
                                                    dist_th = np.inf, # Maximum distance allowed for consecutive positions.
                                                    frames_boundaries = [-1, 100000], # Minimum and maximum frames to consider.
                                                    fmt = 'csv', # Format of the document to be red, 'csv' or 'pkl'.
                                                    colnames = ['POSITION_X', 'POSITION_Y', 'FRAME', 'TRACK_ID'],
                                                    opt_colnames = [], # list of additional metrics to collect.
                                                    remove_no_disp = False)
    all_tracks.append(tracks[0])

plt.figure(figsize = (10,3))
for k, track in enumerate(all_tracks):
    plt.plot(track[:200:,0] - np.mean(track[:200,0]) + 10*k, track[:200, 1])
plt.gca().set_aspect('equal', adjustable='box')
plt.xticks(np.arange(len(all_tracks))*10, laser_intensities)

all_tracks = np.array(all_tracks)

reshaped_tracks = all_tracks.reshape((9, 300, 20, 2))
reshaped_tracks.shape

log_probas = []
conf_factors = []
est_dif_lenghts = []
all_locErrs = []
all_qs = []

tracks = reshaped_tracks[0]

tracks.shape
plt.figure()
plt.plot(tracks[0, :, 0], tracks[0, :, 1])

fixed_params = True
for tracks in reshaped_tracks:
    est_LocErrs, est_ds, est_ls, est_qs, sum_LP = atrack.Confined_fit_multi(tracks, nb_states = 1, verbose = 1, Fixed_LocErr = False, Fixed_d = fixed_params, nb_epochs = 300, batch_size = 300, Initial_params = {'LocErr': [0.01], 'd': [0.55], 'q': [0.001], 'l': [0.2]}, fitting_type = 'All')
    
    #est_LocErrs, est_ds, est_ls, est_qs, sum_LP = atrack.Confined_fit_multi(tracks*0.052, nb_states = 1, verbose = 1, Fixed_LocErr = True, Fixed_d = True, nb_epochs = 200, batch_size = 300, Initial_params = {'LocErr': [0.0116], 'd': [0.0259], 'q': [0.01], 'l': [0.2]}, fitting_type = 'All')
    conf_factors.append(est_ls[0,0])
    est_dif_lenghts.append(est_ds[0,0])
    all_qs.append(est_qs)
    all_locErrs.append(est_LocErrs)
    print(conf_factors)
    print(est_dif_lenghts)

# Results:
#conf_factors = [0.279685449487432, 0.30944399377574605, 0.3665262091532544, 0.5501682269629026, 0.5834224488014728, 0.662859257921852, 0.7395640527509248, 0.8515406465115264, 0.9273056522535253]
#est_dif_lenghts = [0.4797824455621736, 0.47304140350395973, 0.46046943563199166, 0.4230178526820067, 0.41669294756656594, 0.40211160688141984, 0.3887059171958287, 0.3702508885527909, 0.35846862834426635]

est_dif_lenghts = np.array(est_dif_lenghts)
conf_factors = np.array(conf_factors)

plt.figure(figsize = (3.1,2.8))
plt.plot(laser_intensities, conf_factors, marker='x')
plt.xlabel('Laser intensity')
plt.ylabel('Confinement factor')
plt.ylim([0.2, 1])

plt.figure(figsize = (3.1,2.8))
pixel_size = 0.051
plt.plot(laser_intensities, pixel_size*4*est_dif_lenghts/(2*conf_factors)**0.5, marker='x')
plt.xlabel('Laser intensity')
plt.ylabel('Confinement radius (um)')


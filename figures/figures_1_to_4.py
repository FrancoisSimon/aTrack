# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 14:58:05 2026

@author: Franc
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 16:41:59 2023

@author: franc
"""

import numpy as np
from numba import njit, typed, prange, jit
from matplotlib import pyplot as plt
from matplotlib import cm
import time
import os
import seaborn as sns
import pandas as pd

import sys
sys.path.insert(0, r'path/to/atrack')
from atrack import Brownian_fit, Confined_fit, Directed_fit
import tensorflow as tf

@njit
def anomalous_diff_mixture(track_len=200,
                           nb_tracks = 100,
                    LocErr=0.02, # localization error in x, y and z (even if not used)
                    nb_states = 2,
                    Fs = np.array([0.5, 0.5]),
                    Ds = np.array([0.0, 0.05]),
                    nb_dims = 2,
                    velocities = np.array([0.1, 0.0]),
                    angular_Ds = np.array([0.0, 0.0]),
                    conf_forces = np.array([0.0, 0.2]),
                    conf_Ds = np.array([0.0, 0.0]),
                    conf_dists = np.array([0.0, 0.0]),
                    LocErr_std = 0,
                    dt = 0.02,
                    nb_sub_steps = 10):
    # diff + persistent motion + elastic confinement
    conf_sub_forces = conf_forces / nb_sub_steps
    sub_dt = dt / nb_sub_steps
   
    cum_Fs = np.zeros(nb_states)
    cum_Fs[0] = Fs[0]
    for state in range(1, nb_states):
        cum_Fs[state] = cum_Fs[state-1] + Fs[state]
   
    for kkk in range(nb_tracks):
        state = np.argmin(np.random.rand()>cum_Fs)
        D, velocity, angular_D, conf_sub_force, conf_D, conf_dist = (Ds[state], velocities[state], angular_Ds[state], conf_sub_forces[state], conf_Ds[state], conf_dists[state])
       
        positions = np.zeros((track_len * nb_sub_steps, nb_dims))
        disps = np.random.normal(0, np.sqrt(2*D*sub_dt), ((track_len) * nb_sub_steps - 1, nb_dims))
       
        anchor_positions = np.random.normal(0, np.sqrt(2*conf_D*sub_dt), ((track_len) * nb_sub_steps - 1, nb_dims))
        anchor_positions[0] = positions[0] + np.random.normal(0,conf_dist, nb_dims)
       
        for i in range(1, len(anchor_positions)):
            anchor_positions[i] += anchor_positions[i-1]
       
        d_angles = np.random.normal(0, 1, ((track_len) * nb_sub_steps)-1) * (2*angular_D*sub_dt)**0.5
        angles = np.zeros((track_len * nb_sub_steps-1))
        angles[0] = np.random.rand()*2*np.pi
        for i in range(1, len(d_angles)):
            angles[i] = angles[i-1] + d_angles[i]
       
        for i in range(len(positions)-1):
            angle = angles[i-1]
            pesistent_disp = np.array([np.cos(angle), np.sin(angle)]).T * velocity/nb_sub_steps
            positions[i+1] = positions[i] + pesistent_disp + disps[i]
            positions[i+1] = (1-conf_sub_force) *  positions[i+1] + conf_sub_force * anchor_positions[i]
       
        final_track = np.zeros((track_len, nb_dims))
        for i in range(track_len):
            final_track[i] = positions[i*nb_sub_steps]
       
        final_track += np.random.normal(0, LocErr, (track_len, nb_dims))
       
        if kkk ==0:
            final_tracks = typed.List([final_track])
        else:
            final_tracks.append(final_track)
    return final_tracks

"""        
def rainbow_plot(list_of_lists, ylabels, bins = None, input_type = 'values', heigth_ratio = 0.4, figsize = (5,5), density = True, same_x_y = False, grid = True, max_factor = 1.3):

    if input_type == 'values':
        if type(bins) == type(None):
            min_val = np.min(list_of_lists)
            max_val = np.max(list_of_lists)
            range_vals = max_val - min_val
            bins = np.linspace(min_val-0.05*range_vals, max_val+0.05*range_vals, 100)
        densities = []
        step = bins[1]-bins[0]
        for l in list_of_lists:
            d = np.histogram(l, bins = bins, density = density)[0]
            densities.append(list(d))
    elif input_type == 'histograms':
        step = bins[1]-bins[0]
        densities = list_of_lists
        if type(bins) == type(None):
            bins = np.arange(len(densities[0]))
        
    max_d = np.nanmax(densities)
    max_heigth = heigth_ratio * max_d
    nb_hists = len(list_of_lists)
    plt.figure(figsize = figsize)
    if same_x_y:
        plt.xticks(ylabels)
    for k in np.arange(nb_hists):
        d = densities[k]
        fraction = k / nb_hists
        plt.fill(bins[:-1]+0.5*step, d + max_heigth * k / nb_hists, color = cm.jet(fraction), zorder=3*nb_hists-(2*k+1))
        plt.plot(bins[:-1]+0.5*step, d + max_heigth * k / nb_hists, color = 'gray', zorder=3*nb_hists-(2*k))
    plt.yticks(np.arange(nb_hists)*max_heigth/nb_hists, ylabels)
    if grid:
        plt.grid()
    plt.ylims(k / nb_hists)
"""
    
def rainbow_plot(list_of_lists, ylabels, bins = None, input_type = 'values', heigth_ratio = 0.4, figsize = (5,5), density = True, same_x_y = False, grid = True, max_offset = 1.3, size_ratio = 3):
    '''
    Produces a rainbow plot for a list of lists of values

    Parameters
    ----------
    list_of_lists : TYPE
        If input_type = 'values': this must be a list of lists of values. If input_type = 'histograms', this must be a list of histograms, the histograms must have the same binning.
    ylabels : List
        labels : list of numbers or strings of the same length than list_of_lists.
    bins : TYPE, optional
        binnign used for the histograms. If input_type = 'values', bins = None will automatically find a binning.
    heigth_ratio : float in between 0 and 1
        fraction of the plot that contain the histogram offsets : heigth_ratio corresponds to the y offset of the last histogram. 
    figsize : TYPE, optional
        DESCRIPTION. The default is (5,5).
    density : Bool
        If True the histogram heights correspond to their density, if False the heights correspond to the number of counts.
    same_x_y : Bool
        if True ylabels will be used for both x and y ticks. 
    grid : Bool
        Show a grid or not
    max_factor : float
        offset between the last histogram maximum heigh and the upper limit of the plot in y, a higher max_factor increases the y upper limit 
    size_ratio : float
        a higher size_ratio increases the height of the histograms. 
    Returns
    -------
    None

    '''
    if input_type == 'values':
        if type(bins) == type(None):
            min_val = np.min(list_of_lists)
            max_val = np.max(list_of_lists)
            range_vals = max_val - min_val
            bins = np.linspace(min_val-0.05*range_vals, max_val+0.05*range_vals, 100)
        densities = []
        step = bins[1]-bins[0]
        for l in list_of_lists:
            d = np.histogram(l, bins = bins, density = density)[0]
            densities.append(list(d))
    elif input_type == 'histograms':
        step = bins[1]-bins[0]
        densities = list_of_lists
        if type(bins) == type(None):
            bins = np.arange(len(densities[0]))
    
    max_d = np.nanmax(densities[-1])
    nb_hists = len(list_of_lists)
    max_heigth = heigth_ratio * max_d * nb_hists
    plt.figure(figsize = figsize)
    if same_x_y:
        plt.xticks(ylabels)
    for k in np.arange(nb_hists):
        d = densities[k]
        fraction = k / nb_hists
        plt.fill(bins[:-1]+0.5*step, size_ratio*np.array(d) + max_heigth * k / nb_hists, color = cm.jet(fraction), zorder=3*nb_hists-(2*k+1))
        plt.plot(bins[:-1]+0.5*step, size_ratio*np.array(d) + max_heigth * k / nb_hists, color = 'gray', zorder=3*nb_hists-(2*k))
    plt.yticks(np.arange(nb_hists)*max_heigth/nb_hists, ylabels)
    if grid:
        plt.grid()
    plt.ylim([-0.5*max_d, max_d*(nb_hists-1+size_ratio*max_offset)])
#%% 
'''
Plot tracks, Fig 1c
'''
sav_dir = r'D:\anomalous/track_plots/'
sv = r'D:\anomalous\Fig_data/' # Directory to save the figure data in csv files

if not os.path.exists(sav_dir):
    os.makedirs(sav_dir)

l = 0.05

tracks = anomalous_diff_mixture(track_len=100,
                            nb_tracks = 10,
                            LocErr=0.02, # localization error in x, y and z (even if not used)
                            nb_states = 1,
                            Fs = np.array([1]),
                            Ds = np.array([0.25]),
                            nb_dims = 2,
                            velocities = np.array([0.0]),
                            angular_Ds = np.array([0.0]),
                            conf_forces = np.array([l]),
                            conf_Ds = np.array([0.0]),
                            conf_dists = np.array([0]),
                            LocErr_std = 0,
                            dt = 0.02,
                            nb_sub_steps = 20)

track = tracks[0]
lim = 2
plt.figure(figsize = (3,3))
plt.plot(track[:,0], track[:,1])
plt.xlim([np.mean(track[:,0])-lim, np.mean(track[:,0])+lim])
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(sav_dir + 'confined_track_100_steps_l=' + str(l) + '_3.svg', format = 'svg')




tracks = anomalous_diff_mixture(track_len=30,
                            nb_tracks = 10,
                            LocErr=0.02, # localization error in x, y and z (even if not used)
                            nb_states = 1,
                            Fs = np.array([1]),
                            Ds = np.array([0.0]),
                            nb_dims = 2,
                            velocities = np.array([0.02]),
                            angular_Ds = np.array([0.0]),
                            conf_forces = np.array([0]),
                            conf_Ds = np.array([0.0]),
                            conf_dists = np.array([0]),
                            LocErr_std = 0,
                            dt = 0.02,
                            nb_sub_steps = 20)

track = tracks[4]
lim = 0.2
plt.figure(figsize = (3,3))
plt.plot(track[:30,0], track[:30,1])
plt.xlim([np.mean(track[:,0])-lim, np.mean(track[:,0])+lim])
plt.gca().set_aspect('equal', adjustable='box')


sav_dir = r'D:\anomalous/track_plots/'
if not os.path.exists(sav_dir):
    os.makedirs(sav_dir)

l = 0.002
'''
linear
'''

tracks = anomalous_diff_mixture(track_len=50,
                           nb_tracks = 10,
                            LocErr=0.02, # localization error in x, y and z (even if not used)
                            nb_states = 1,
                            Fs = np.array([1]),
                            Ds = np.array([0.0]),
                            nb_dims = 2,
                            velocities = np.array([l]),
                            angular_Ds = np.array([0.0]),
                            conf_forces = np.array([0]),
                            conf_Ds = np.array([0.0]),
                            conf_dists = np.array([0]),
                            LocErr_std = 0,
                            dt = 0.02,
                            nb_sub_steps = 20)

for k in [1,2,3]:
    track = tracks[k]
    lim = 1
    plt.figure(figsize = (6,6))
    plt.plot(track[:,0], track[:,1])
    plt.xlim([np.mean(track[:,0])-lim, np.mean(track[:,0])+lim])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(sav_dir + 'linear_track_100_steps_v=' + str(l) + '_%s.svg'%k, format = 'svg')

for k in [1,2,3]:
    track = tracks[k]
    lim = 0.2
    plt.figure(figsize = (6,6))
    plt.plot(track[:,0], track[:,1])
    plt.xlim([np.mean(track[:,0])-lim, np.mean(track[:,0])+lim])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(sav_dir + 'linear_track_100_steps_v=' + str(l) + '_%s_zoomed.svg'%k, format = 'svg')


'''
linear + diffusive
'''

l = 0.3
tracks = anomalous_diff_mixture(track_len=50,
                                nb_tracks = 10,
                                LocErr=0.02, # localization error in x, y and z (even if not used)
                                nb_states = 1,
                                Fs = np.array([1]),
                                Ds = np.array([l**2/(2*0.02)]),
                                nb_dims = 2,
                                velocities = np.array([0.1]),
                                angular_Ds = np.array([0.0]),
                                conf_forces = np.array([0]),
                                conf_Ds = np.array([0.0]),
                                conf_dists = np.array([0]),
                                LocErr_std = 0,
                                dt = 0.02,
                                nb_sub_steps = 20)

for k in [1,2,3]:
    track = tracks[k]
    lim = 6
    plt.figure(figsize = (6,6))
    plt.plot(track[:,0], track[:,1])
    plt.xlim([np.mean(track[:,0])-lim, np.mean(track[:,0])+lim])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(sav_dir + 'linear0.1+diffusion_track_100_steps_d=' + str(l) + '_%s.svg'%k, format = 'svg')

'''
directed w/ changing orientation
'''

l = 0.3
tracks = anomalous_diff_mixture(track_len=100,
                                nb_tracks = 10,
                                LocErr=0.02, # localization error in x, y and z (even if not used)
                                nb_states = 1,
                                Fs = np.array([1]),
                                Ds = np.array([0.]),
                                nb_dims = 2,
                                velocities = np.array([0.1]),
                                angular_Ds = np.array([l**2/(2*0.02)]),
                                conf_forces = np.array([0]),
                                conf_Ds = np.array([0.0]),
                                conf_dists = np.array([0]),
                                LocErr_std = 0,
                                dt = 0.02,
                                nb_sub_steps = 20)

for k in [1,2,3]:
    track = tracks[k]
    lim = 4
    plt.figure(figsize = (6,6))
    plt.plot(track[:,0], track[:,1])
    plt.xlim([np.mean(track[:,0])-lim, np.mean(track[:,0])+lim])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(sav_dir + 'directed_0.1_changing_orientation_100_steps_d=' + str(l) + '_%s.svg'%k, format = 'svg')

#%%
'''
Analyzing confined tracks with varying confinement factors, fig 2b and fig 3b, 
'''

conf_vals = np.arange(0, 1.01 , 0.1)

all_tracks = []

for l in conf_vals:
    print(l)
    
    tracks = anomalous_diff_mixture(track_len=200,
                               nb_tracks = 10000,
                                LocErr=0.02, # localization error in x, y and z (even if not used)
                                nb_states = 1,
                                Fs = np.array([1]),
                                Ds = np.array([0.25]),
                                nb_dims = 2,
                                velocities = np.array([0.0]),
                                angular_Ds = np.array([0.0]),
                                conf_forces = np.array([l]),
                                conf_Ds = np.array([0.0]),
                                conf_dists = np.array([0]),
                                LocErr_std = 0,
                                dt = 0.02,
                                nb_sub_steps = 20)
    
    tracks = np.array(tracks)
    all_tracks.append(tracks)

all_tracks = np.array(all_tracks)

sav_dir = r'D:\anomalous/rainbow_conf/'
if not os.path.exists(sav_dir):
    os.makedirs(sav_dir)

np.save(sav_dir + 'tracks', all_tracks)
all_tracks = np.load(sav_dir + 'tracks.npy')

all_LocErrs = []
all_ds = []
all_qs = []
all_ls = []
all_LP  = []

for tracks in all_tracks:
    
    with tf.device('/CPU:0'):
        _, _, LP0 = Brownian_fit(tracks, Fixed_LocErr = True, output_type = '1', verbose = 1, nb_epochs = 1000)
        
        est_LocErrs, est_ds, est_qs, est_ls, LP = Confined_fit(tracks, Fixed_LocErr = True, output_type = '1', verbose = 1, nb_epochs = 1000)
    
    all_LocErrs.append(est_LocErrs)
    all_ds.append(est_ds)
    all_qs.append(est_qs)
    all_ls.append(-np.log(1-est_ls))
    #all_LP.append(LP - LP0)

len(all_LocErrs)
np.mean(est_ls)
np.std(est_ls)

np.mean(LP - LP0)
np.mean(-np.log(1-est_ls))
np.std(-np.log(1-est_ls))

tracks.shape
-np.log(1-0.5)

np.save(sav_dir + 'all_ds', all_ds)
np.save(sav_dir + 'all_qs', all_qs)
np.save(sav_dir + 'all_ls', all_ls)
np.save(sav_dir + 'all_LP', all_LP)



all_ds = np.load(sav_dir + 'all_ds.npy')
all_qs = np.load(sav_dir + 'all_qs.npy')
all_ls = np.load(sav_dir + 'all_ls.npy')
all_LP = np.load(sav_dir + 'all_LP.npy')

cor_all_ds = (1-np.exp(-all_ls))/all_ls*all_ds

names = ['ds.svg', 'qs.svg', 'ls.svg', 'LP.svg']
all_lists = [cor_all_ds, all_qs, all_ls, all_LP]
xlabels = ['Estimated diffusion length per step [$\mathrm{\mu m.\Delta t^{-1}}$]', 'Estimated diffusion length per step of the potential well [$\mathrm{\mu m.\Delta t^{-1}}$]', 'Estimated confinement factor', 'likelihood ratio (scale: log$_{10}$)']
all_bins =  [None, None, np.arange(-0.05, 1.301, 0.01), None]
sames = [False, False, True, False]

for hists, xlabel, bins, name, same in zip(all_lists, xlabels, all_bins, names, sames):
    
    rainbow_plot(hists, ylabels = np.round(conf_vals, 1), bins = bins, heigth_ratio = 0.9, figsize = (3.5,3.5), density = True, same_x_y = same, size_ratio = 2, max_offset = 0.7)
    plt.ylabel('True confinement factor')
    plt.xlabel(xlabel)
    #plt.tight_layout()
    plt.grid(zorder=-30)
    plt.savefig(sav_dir + name, format = 'svg')

rainbow_plot(all_LP/np.log(10), ylabels = np.round(conf_vals, 1), bins = None, heigth_ratio = 0.9, figsize = (3.5,3.5), density = True, same_x_y = same, size_ratio = 2, max_offset = 0.7)
plt.ylabel('True confinement factor')
plt.xlabel(xlabel)
plt.tight_layout()
plt.grid(zorder=-30)
plt.savefig(sav_dir + name, format = 'svg')

all_conf_radius = []
for ds, ls  in zip(cor_all_ds, all_ls):
    all_conf_radius.append(((ds*0.02)/ls)**0.5)

len(all_conf_radius)

true_conf_radius = ((0.1*0.02) / conf_vals)**0.5

rainbow_plot(all_conf_radius, ylabels = np.round(true_conf_radius, 3), bins = np.arange(0.03, 0.2, 0.002), heigth_ratio = 0.9, figsize = (3.5,3.5), density = True, same_x_y = same, size_ratio = 2, max_offset = 0.7)
plt.ylabel('True confinement radius [$\mathrm{\mu m}$]')
plt.xlabel('Estimated confinement radius [$\mathrm{\mu m}$]')
plt.tight_layout()
plt.grid(zorder=-30)
plt.savefig(sav_dir + 'conf_radius.svg', format = 'svg')


data = pd.DataFrame(cor_all_ds[:, :, 0].T, columns = conf_vals)
data.to_csv(sv + '3b-1.csv')




data.to_csv(sv + '3-sup1a-3.csv')


len(hists)

all_ds.shape
#%%
'''
Analyzing tracks in linear motion varying the velocity, fig 2b
'''

velocity_vals = np.arange(0, 0.101, 0.01)

(0.02*0.25*2)**0.5

all_LocErrs = []
all_ds = []
all_qs = []
all_ls = []
all_LP  = []

all_tracks = []

for l in velocity_vals:
    print(l)
    
    tracks = anomalous_diff_mixture(track_len=30,
                               nb_tracks = 10000,
                                LocErr=0.02, # localization error in x, y and z (even if not used)
                                nb_states = 1,
                                Fs = np.array([1]),
                                Ds = np.array([0.0]),
                                nb_dims = 2,
                                velocities = np.array([l]),
                                angular_Ds = np.array([0.0]),
                                conf_forces = np.array([0]),
                                conf_Ds = np.array([0.0]),
                                conf_dists = np.array([0]),
                                LocErr_std = 0,
                                dt = 0.02,
                                nb_sub_steps = 20)
    
    tracks = np.array(tracks)
    
    all_tracks.append(tracks)

sav_dir = r'D:\anomalous/rainbow_linear_short_tracks/'
if not os.path.exists(sav_dir):
    os.makedirs(sav_dir)

np.save(sav_dir + 'tracks', all_tracks)
all_tracks = np.load(sav_dir + 'tracks.npy')

for tracks in all_tracks:
    
    _, _, LP0 = Brownian_fit(tracks, verbose  = 1, output_type = 'values')
    
    est_LocErrs, est_ds, est_qs, est_ls, LP, mean_pred_vs = Directed_fit(tracks, Fixed_LocErr = True, verbose  = 1, output_type = 'values')
    
    all_LocErrs.append(est_LocErrs)
    all_ds.append(est_ds)
    all_qs.append(est_qs)
    all_ls.append(est_ls)
    all_LP.append(LP - LP0)

np.save(sav_dir + 'all_LocErrs', all_LocErrs)
np.save(sav_dir + 'all_ds', all_ds)
np.save(sav_dir + 'all_qs', all_qs)
np.save(sav_dir + 'all_ls', all_ls)
np.save(sav_dir + 'all_LP', all_LP)

all_ds = np.load(sav_dir + 'all_ds.npy')
all_qs = np.load(sav_dir + 'all_qs.npy')
all_ls = np.load(sav_dir + 'all_ls.npy')
all_LP = np.load(sav_dir + 'all_LP.npy')

all_lists = [all_ds, all_qs, all_ls, all_LP]
names = ['ds.svg', 'qs.svg', 'ls.svg', 'LP.svg']
xlabels = ['Estimated diffusion length [$\mathrm{\mu m.\Delta t^{-1}}$]', 'Estimated speed change per step [$\mathrm{\mu m.\Delta t^{-1}}$]', 'Estimated speed [$\mathrm{\mu m.\Delta t^{-1}}$]', 'likelihood ratio (scale: log$_{10}$)']
all_bins =  [None, None, np.arange(-0.005, 0.1101, 0.001),None]
same_xys = [False, False, True, False]

for hists, xlabel, bins, same, name in zip(all_lists, xlabels, all_bins, same_xys, names):
    rainbow_plot(hists, ylabels = np.round(velocity_vals, 2), bins = bins, heigth_ratio = 0.9, figsize = (3.5,3.5), density = True, same_x_y = same, size_ratio = 2, max_offset = 0.7)
    plt.ylabel('True speed [$\mathrm{\mu m.\Delta t^{-1}}$]')
    plt.xlabel(xlabel)
    #plt.tight_layout()
    plt.grid(zorder=-30)
    plt.savefig(sav_dir + name, format = 'svg')

rainbow_plot(hists/np.log(10), ylabels = np.round(velocity_vals, 2), bins = bins, heigth_ratio = 0.4, figsize = (3.5,3.5), density = True, same_x_y = same)
plt.ylabel('True speed [$\mathrm{\mu m.\Delta t^{-1}}$]')
plt.xlabel(xlabel)
plt.tight_layout()
plt.grid(zorder=-30)

plt.savefig(sav_dir + name, format = 'svg')

data = pd.DataFrame(data)

#%%
'''
Analyzing tracks in linear motion varying the velocity (smaller velocities), fig 4b and fig4-sup 1a
'''

velocity_vals = np.arange(0, 0.04001, 0.002)

all_LocErrs = []
all_ds = []
all_qs = []
all_ls = []
all_LP  = []

all_tracks = []

for l in velocity_vals:
    print(l)
    
    tracks = anomalous_diff_mixture(track_len=30,
                               nb_tracks = 10000,
                                LocErr=0.02, # localization error in x, y and z (even if not used)
                                nb_states = 1,
                                Fs = np.array([1]),
                                Ds = np.array([0.0]),
                                nb_dims = 2,
                                velocities = np.array([l]),
                                angular_Ds = np.array([0.0]),
                                conf_forces = np.array([0]),
                                conf_Ds = np.array([0.0]),
                                conf_dists = np.array([0]),
                                LocErr_std = 0,
                                dt = 0.02,
                                nb_sub_steps = 20)
    
    tracks = np.array(tracks)
    
    all_tracks.append(tracks)

sav_dir = r'D:\anomalous/rainbow_linear_short_tracks_low_v/'
if not os.path.exists(sav_dir):
    os.makedirs(sav_dir)
    
np.save(sav_dir + 'tracks', all_tracks)
all_tracks = np.load(sav_dir + 'tracks.npy')

for tracks in all_tracks:
    
    _, _, LP0 = Brownian_fit(tracks, nb_dims = 2, verbose  = 1, Initial_params = {'LocErr': 0.02, 'd': 0.002}, Fixed_LocErr = True, output_type = 'values')
    
    est_LocErrs, est_ds, est_qs, est_ls, LP, pred_kis = Directed_fit(tracks, Fixed_LocErr = True, verbose  = 1, Initial_params = {'LocErr': 0.02, 'd': 0.0000001, 'q': 0.0000001, 'l': 0.002}, output_type = 'values')
    
    all_LocErrs.append(est_LocErrs)
    all_ds.append(est_ds)
    all_qs.append(est_qs)
    all_ls.append(est_ls)
    all_LP.append(LP - LP0)

np.save(sav_dir + 'all_LocErrs', all_LocErrs)
np.save(sav_dir + 'all_ds', all_ds)
np.save(sav_dir + 'all_qs', all_qs)
np.save(sav_dir + 'all_ls', all_ls)
np.save(sav_dir + 'all_LP', all_LP)

all_ds = np.load(sav_dir + 'all_ds.npy')
all_qs = np.load(sav_dir + 'all_qs.npy')
all_ls = np.load(sav_dir + 'all_ls.npy')
all_LP = np.load(sav_dir + 'all_LP.npy')

all_lists = [all_ds[:11], all_qs[:11], all_ls[:11], all_LP[:11]]
names = ['ds.svg', 'qs.svg', 'ls.svg', 'LP.svg']
xlabels = ['Estimated diffusion length [$\mathrm{\mu m.\Delta t^{-1}}$]', 'Estimated speed change per step [$\mathrm{\mu m.\Delta t^{-1}}$]', 'Estimated speed [$\mathrm{\mu m.\Delta t^{-1}}$]', 'likelihood ratio (scale: log$_{10}$)']
all_bins =  [None, None, np.arange(-0.001, 0.0221, 0.0002),None]
same_xys = [False, False, True, False]

for hists, xlabel, bins, same, name in zip(all_lists, xlabels, all_bins, same_xys, names):
    rainbow_plot(hists, ylabels = np.round(velocity_vals[:11], 3), bins = bins, heigth_ratio = 0.85, figsize = (3.5,3.5), density = True, same_x_y = same, size_ratio = 5, max_offset = 0.7)
    plt.ylabel('True speed [$\mathrm{\mu m.\Delta t^{-1}}$]')
    plt.xlabel(xlabel)
    #plt.tight_layout()
    plt.grid(zorder=-30)
    plt.savefig(sav_dir + name, format = 'svg')

rainbow_plot(hists/np.log(10), ylabels = np.round(velocity_vals, 2), bins = bins, heigth_ratio = 0.4, figsize = (3.5,3.5), density = True, same_x_y = same)
plt.ylabel('True speed [$\mathrm{\mu m.\Delta t^{-1}}$]')
plt.xlabel(xlabel)
plt.tight_layout()
plt.grid(zorder=-30)

plt.savefig(sav_dir + name, format = 'svg')


sv = r'D:\anomalous\Fig_data/'
data = pd.DataFrame(all_ls[:11, :, 0].T, columns = velocity_vals[:11])
data.to_csv(sv + '4b.csv')

data = pd.DataFrame(all_LP[:11, :, 0].T, columns = velocity_vals[:11])
data.to_csv(sv + '4-sup1a-1.csv')

data = pd.DataFrame(all_ds[:11, :, 0].T, columns = velocity_vals[:11])
data.to_csv(sv + '4-sup1a-2.csv')

data = pd.DataFrame(all_qs[:11, :, 0].T, columns = velocity_vals[:11])
data.to_csv(sv + '4-sup1a-3.csv')




'''
rainbow plot Lb/Lc for diffusive tracks and for confined and directed tracks, fig 2a
'''

sav_dir = r'D:\anomalous/rainbow_bronwian_var_track_length/'

all_LocErrs = []
all_ds = []
all_qs = []
all_ls = []
all_LP  = []

#Brownian
all_tracks = []
track_lengths = [5, 7, 10, 20, 50, 100, 200, 400]

for track_length in track_lengths:
    print(track_length)
    
    tracks = anomalous_diff_mixture(track_len=track_length,
                               nb_tracks = 10000,
                                LocErr=0.02, # localization error in x, y and z (even if not used)
                                nb_states = 1,
                                Fs = np.array([1]),
                                Ds = np.array([0.25]),
                                nb_dims = 2,
                                velocities = np.array([0]),
                                angular_Ds = np.array([0.0]),
                                conf_forces = np.array([0]),
                                conf_Ds = np.array([0.0]),
                                conf_dists = np.array([0]),
                                LocErr_std = 0,
                                dt = 0.02,
                                nb_sub_steps = 20)
    
    tracks = np.array(tracks)
    
    all_tracks.append(tracks)

all_LP1  = []
all_LP2  = []

for tracks in all_tracks:
    
    _, _, LP0 = Brownian_fit(tracks, verbose  = 1, output_type = 'values')
    
    est_LocErrs, est_ds, est_qs, est_ls, LP1, mean_pred_vs = Directed_fit(tracks, Fixed_LocErr = True, verbose  = 1, output_type = 'values')
    est_LocErrs, est_ds, est_qs, est_ls, LP2 = Confined_fit(tracks, Fixed_LocErr = True, verbose  = 1, output_type = 'values')

    all_LP1.append(LP1 - LP0)
    all_LP2.append(LP2 - LP0)

inv_all_P1 = []
inv_all_P2 = []
for LP1, LP2 in zip(all_LP1, all_LP2):
    inv_all_P1.append(np.exp(-LP1))
    inv_all_P2.append(np.exp(-LP2))

rainbow_plot(inv_all_P1, ylabels = track_lengths, bins = np.arange(-0.1, 1.21, 0.02), heigth_ratio = 0.7, figsize = (3.5,3.5), density = True, same_x_y = False, size_ratio = 5, max_offset = 0.8)
rainbow_plot(inv_all_P2, ylabels = track_lengths, bins = np.arange(-0.1, 1.31, 0.02), heigth_ratio = 1, figsize = (3.5,3.5), density = True, same_x_y = False, size_ratio = 3, max_offset = 1.5)

#%%
'''

rainbow plot Lb/Lc for diffusive, confined and directed tracks, fig 2-supplement figures
'''

sav_dir = r'D:\anomalous/rainbow_bronwian_var_track_length/'

#Brownian tracks

all_tracks = []
track_lengths = [5, 7, 10, 20, 50, 100, 200, 400]

for track_length in track_lengths:
    print(track_length)
    
    tracks = anomalous_diff_mixture(track_len=track_length,
                               nb_tracks = 10000,
                                LocErr=0.02, # localization error in x, y and z (even if not used)
                                nb_states = 1,
                                Fs = np.array([1]),
                                Ds = np.array([0.25]),
                                nb_dims = 2,
                                velocities = np.array([0]),
                                angular_Ds = np.array([0.0]),
                                conf_forces = np.array([0.0]),
                                conf_Ds = np.array([0.0]),
                                conf_dists = np.array([0]),
                                LocErr_std = 0,
                                dt = 0.02,
                                nb_sub_steps = 20)
    
    tracks = np.array(tracks)
    
    all_tracks.append(tracks)

all_LP1  = []
all_LP2  = []

for tracks in all_tracks:
    
    _, _, LP0 = Brownian_fit(tracks, verbose  = 1, output_type = 'values')
    
    est_LocErrs, est_ds, est_qs, est_ls, LP1, mean_pred_vs = Directed_fit(tracks, Fixed_LocErr = True, verbose  = 1, output_type = 'values')
    est_LocErrs, est_ds, est_qs, est_ls, LP2 = Confined_fit(tracks, Fixed_LocErr = True, verbose  = 1, output_type = 'values')

    all_LP1.append(LP1 - LP0)
    all_LP2.append(LP2 - LP0)
    
    np.save(sav_dir + 'log_likelihood_ratio_brownian_track_directed_test.npy', all_LP1)
    np.save(sav_dir + 'log_likelihood_ratio_brownian_track_confined_test.npy', all_LP2)

all_LP1 = np.load(sav_dir + 'log_likelihood_ratio_brownian_track_directed_test.npy')
all_LP2 = np.load(sav_dir + 'log_likelihood_ratio_brownian_track_confined_test.npy')

inv_all_P1 = []
inv_all_P2 = []
for LP1, LP2 in zip(all_LP1, all_LP2):
    inv_all_P1.append(np.exp(-LP1))
    inv_all_P2.append(np.exp(-LP2))

rainbow_plot(inv_all_P1, ylabels = track_lengths, bins = np.arange(-0.1, 1.21, 0.02), heigth_ratio = 0.7, figsize = (3.5,3.5), density = True, same_x_y = False, size_ratio = 5, max_offset = 0.8)
rainbow_plot(inv_all_P2, ylabels = track_lengths, bins = np.arange(-0.1, 1.31, 0.02), heigth_ratio = 0.7, figsize = (3.5,3.5), density = True, same_x_y = False, size_ratio = 5, max_offset = 0.8)

rainbow_plot(all_LP1, ylabels = track_lengths, bins = np.arange(-0.5, 5, 0.1), heigth_ratio = 0.7, figsize = (3.5,3.5), density = True, same_x_y = False, size_ratio = 5, max_offset = 0.8)
rainbow_plot(all_LP2, ylabels = track_lengths, bins = np.arange(-0.5, 5, 0.1), heigth_ratio = 0.7, figsize = (3.5,3.5), density = True, same_x_y = False, size_ratio = 5, max_offset = 0.8)

#Confined tracks

sav_dir = r'D:\anomalous/rainbow_bronwian_var_track_length/'

all_tracks = []
track_lengths = [5, 7, 10, 20, 50, 100, 200, 400]

0.1/0.25**0.5 # confinement radius

all_LP1  = []
all_LP2  = []

for track_length in track_lengths:
    print(track_length)
    
    tracks = anomalous_diff_mixture(track_len=track_length,
                               nb_tracks = 10000,
                                LocErr=0.02, # localization error in x, y and z (even if not used)
                                nb_states = 1,
                                Fs = np.array([1]),
                                Ds = np.array([0.25]),
                                nb_dims = 2,
                                velocities = np.array([0]),
                                angular_Ds = np.array([0.0]),
                                conf_forces = np.array([0.25]),
                                conf_Ds = np.array([0.0]),
                                conf_dists = np.array([0]),
                                LocErr_std = 0,
                                dt = 0.02,
                                nb_sub_steps = 20)
    
    tracks = np.array(tracks)
    
    _, _, LP0 = Brownian_fit(tracks, verbose  = 0, output_type = 'values')
    #est_LocErrs1, est_ds1, est_qs1, est_ls1, LP1 = Directed_fit(tracks, nb_dims)
    
    est_LocErrs, est_ds, est_qs, est_ls, LP1, mean_pred_vs = Directed_fit(tracks, Fixed_LocErr = True, verbose  = 0, output_type = 'values')
    est_LocErrs, est_ds, est_qs, est_ls, LP2 = Confined_fit(tracks, Fixed_LocErr = True, verbose  = 0, output_type = 'values')

    all_LP1.append(LP1 - LP0)
    all_LP2.append(LP2 - LP0)
    
    np.save(sav_dir + 'log_likelihood_ratio_confined_track_directed_test.npy', all_LP1)
    np.save(sav_dir + 'log_likelihood_ratio_confined_track_confined_test.npy', all_LP2)

all_LP1 = np.load(sav_dir + 'log_likelihood_ratio_confined_track_directed_test.npy')
all_LP2 = np.load(sav_dir + 'log_likelihood_ratio_confined_track_confined_test.npy')

inv_all_P1 = []
inv_all_P2 = []
for LP1, LP2 in zip(all_LP1, all_LP2):
    inv_all_P1.append(np.exp(-LP1))
    inv_all_P2.append(np.exp(-LP2))

rainbow_plot(inv_all_P1, ylabels = track_lengths, bins = np.arange(-0.1, 1.21, 0.02), heigth_ratio = 0.7, figsize = (3.5,3.5), density = True, same_x_y = False, size_ratio = 5, max_offset = 0.8)
plt.savefig(sav_dir + 'pvalue_confined_tracks_directed_test.svg', format = 'svg')
rainbow_plot(inv_all_P2, ylabels = track_lengths, bins = np.arange(-0.1, 1.31, 0.02), heigth_ratio = 0.7, figsize = (3.5,3.5), density = True, same_x_y = False, size_ratio = 5, max_offset = 0.8)
plt.savefig(sav_dir + 'pvalue_confined_tracks_confined_test.svg', format = 'svg')

rainbow_plot(all_LP1, ylabels = track_lengths, bins = np.arange(-0.5, 5, 0.1), heigth_ratio = 0.7, figsize = (3.5,3.5), density = True, same_x_y = False, size_ratio = 5, max_offset = 0.8)
plt.savefig(sav_dir + 'log_likeliood_ratio_confined_tracks_directed_test.svg', format = 'svg')
rainbow_plot(all_LP2, ylabels = track_lengths, bins = None, heigth_ratio = 0.5, figsize = (3.5,3.5), density = True, same_x_y = False, size_ratio = 2, max_offset = 0.5)
plt.savefig(sav_dir + 'log_likeliood_ratio_confined_tracks_confined_test.svg', format = 'svg')





#Directed tracks

sav_dir = r'D:\anomalous/rainbow_bronwian_var_track_length/'

track_lengths = [5, 7, 10, 20, 50, 100, 200, 400]

0.1/0.25**0.5 # confinement radius

all_LP1  = []
all_LP2  = []

for track_length in track_lengths:
    print(track_length)
    
    tracks = anomalous_diff_mixture(track_len=track_length,
                               nb_tracks = 10000,
                                LocErr=0.02, # localization error in x, y and z (even if not used)
                                nb_states = 1,
                                Fs = np.array([1]),
                                Ds = np.array([0.0]),
                                nb_dims = 2,
                                velocities = np.array([0.02]),
                                angular_Ds = np.array([0.0]),
                                conf_forces = np.array([0.0]),
                                conf_Ds = np.array([0.0]),
                                conf_dists = np.array([0]),
                                LocErr_std = 0,
                                dt = 0.02,
                                nb_sub_steps = 20)
    
    tracks = np.array(tracks)
    
    _, _, LP0 = Brownian_fit(tracks, verbose  = 0, output_type = 'values')
    
    est_LocErrs, est_ds, est_qs, est_ls, LP1, mean_pred_vs = Directed_fit(tracks, Fixed_LocErr = True, verbose  = 0, output_type = 'values')
    est_LocErrs, est_ds, est_qs, est_ls, LP2 = Confined_fit(tracks, Fixed_LocErr = True, verbose  = 0, output_type = 'values')
    
    all_LP1.append(LP1 - LP0)
    all_LP2.append(LP2 - LP0)
    
    np.save(sav_dir + 'log_likelihood_ratio_directed_track_directed_test.npy', all_LP1)
    np.save(sav_dir + 'log_likelihood_ratio_directed_track_confined_test.npy', all_LP2)

all_LP1 = np.load(sav_dir + 'log_likelihood_ratio_directed_track_directed_test.npy')
all_LP2 = np.load(sav_dir + 'log_likelihood_ratio_directed_track_confined_test.npy')

inv_all_P1 = []
inv_all_P2 = []
for LP1, LP2 in zip(all_LP1, all_LP2):
    inv_all_P1.append(np.exp(-LP1))
    inv_all_P2.append(np.exp(-LP2))

rainbow_plot(inv_all_P1, ylabels = track_lengths, bins = None, heigth_ratio = 0.9, figsize = (3.5,3.5), density = True, same_x_y = False, size_ratio = 5, max_offset = 0.8)
plt.savefig(sav_dir + 'pvalue_directed_tracks_directed_test.svg', format = 'svg')
rainbow_plot(inv_all_P2, ylabels = track_lengths, bins = np.arange(-0.1, 1.31, 0.02), heigth_ratio = 0.7, figsize = (3.5,3.5), density = True, same_x_y = False, size_ratio = 5, max_offset = 0.8)
plt.savefig(sav_dir + 'pvalue_directed_tracks_confined_test.svg', format = 'svg')

rainbow_plot(all_LP1, ylabels = track_lengths, bins = None, heigth_ratio = 0.6, figsize = (3.5,3.5), density = True, same_x_y = False, size_ratio = 3, max_offset = 0.8)
plt.savefig(sav_dir + 'log_likeliood_ratio_directed_tracks_directed_test.svg', format = 'svg')
rainbow_plot(all_LP2, ylabels = track_lengths, bins = np.arange(-0.5, 5, 0.1), heigth_ratio = 0.9, figsize = (3.5,3.5), density = True, same_x_y = False, size_ratio = 3, max_offset = 0.8)
plt.savefig(sav_dir + 'log_likeliood_ratio_directed_tracks_confined_test.svg', format = 'svg')


#%%
'''
Varying the track length of confined and directed tracks, fig 2a
'''
#Confined

sav_dir = r'D:\anomalous/rainbow_bronwian_var_track_length_long_tracks/'
if not os.path.exists(sav_dir):
    os.makedirs(sav_dir)

all_tracks = []
track_lengths = [5, 50, 100, 150, 200, 250, 300, 350, 400]

0.1/0.25**0.5 # confinement radius

all_LP1  = []
all_LP2  = []

for track_length in track_lengths:
    print(track_length)
    
    tracks = anomalous_diff_mixture(track_len=track_length,
                               nb_tracks = 10000,
                                LocErr=0.02, # localization error in x, y and z (even if not used)
                                nb_states = 1,
                                Fs = np.array([1]),
                                Ds = np.array([0.25]),
                                nb_dims = 2,
                                velocities = np.array([0]),
                                angular_Ds = np.array([0.0]),
                                conf_forces = np.array([0.25]),
                                conf_Ds = np.array([0.0]),
                                conf_dists = np.array([0]),
                                LocErr_std = 0,
                                dt = 0.02,
                                nb_sub_steps = 20)
    
    tracks = np.array(tracks)
    
    _, _, LP0 = Brownian_fit(tracks, verbose  = 0, output_type = 'values')
    
    est_LocErrs, est_ds, est_qs, est_ls, LP2 = Confined_fit(tracks, Fixed_LocErr = True, verbose  = 0, output_type = 'values')

    all_LP2.append(LP2 - LP0)
    
    np.save(sav_dir + 'log_likelihood_ratio_confined_track_confined_test.npy', all_LP2)

inv_all_P2 = []
for LP1, LP2 in zip(all_LP1, all_LP2):
    inv_all_P2.append(np.exp(-LP2))

rainbow_plot(all_LP2, ylabels = track_lengths, bins = np.arange(-5, 60, 1), heigth_ratio = 0.9, figsize = (3.5,3.5), density = True, same_x_y = False, size_ratio = 2, max_offset = 0.8)
plt.ylabel('Log likelihood ratio')
plt.xlabel('Number of time points')
plt.grid(zorder=-30)
plt.savefig(sav_dir + 'log_likeliood_ratio_confined_tracks_confined_test.svg', format = 'svg')


#Directed
sav_dir = r'D:\anomalous/rainbow_bronwian_var_track_length_short_tracks/'
if not os.path.exists(sav_dir):
    os.makedirs(sav_dir)
    
track_lengths = [4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

all_LP1  = []
all_LP2  = []

for track_length in track_lengths:
    print(track_length)
    
    tracks = anomalous_diff_mixture(track_len=track_length,
                               nb_tracks = 10000,
                                LocErr=0.02, # localization error in x, y and z (even if not used)
                                nb_states = 1,
                                Fs = np.array([1]),
                                Ds = np.array([0.0]),
                                nb_dims = 2,
                                velocities = np.array([0.02]),
                                angular_Ds = np.array([0.0]),
                                conf_forces = np.array([0.0]),
                                conf_Ds = np.array([0.0]),
                                conf_dists = np.array([0]),
                                LocErr_std = 0,
                                dt = 0.02,
                                nb_sub_steps = 20)
    
    tracks = np.array(tracks)
    
    _, _, LP0 = Brownian_fit(tracks, verbose  = 0, output_type = 'values')
    
    est_LocErrs, est_ds, est_qs, est_ls, LP1, mean_pred_vs = Directed_fit(tracks, Fixed_LocErr = True, verbose  = 0, output_type = 'values')
    
    all_LP1.append(LP1 - LP0)
    
    np.save(sav_dir + 'log_likelihood_ratio_directed_track_directed_test.npy', all_LP1)

inv_all_P1 = []
for LP1, LP2 in zip(all_LP1, all_LP2):
    inv_all_P1.append(np.exp(-LP1))

rainbow_plot(all_LP1, ylabels = track_lengths, bins = None, heigth_ratio = 0.9, figsize = (3.5,3.5), density = True, same_x_y = False, size_ratio = 2, max_offset = 0.8)
plt.xlabel('Log likelihood ratio')
plt.ylabel('Number of time points')
plt.grid(zorder=-30)
plt.savefig(sav_dir + 'log_likeliood_ratio_directed_tracks_directed_test.svg', format = 'svg')



#%%


'''
Vary both confinement factor and track length of confined tracks, fig 3c and fig 3-sup fig1b,c
'''
sav_dir = r'D:\anomalous/confined_var_track_len_conf_factor/'
if not os.path.exists(sav_dir):
    os.makedirs(sav_dir)
    
all_tracks = []

track_lengths = [10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 400, 600]
conf_factors = np.array([0.  , 0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 , 0.55, 0.6, 0.7, 0.8, 0.9 , 1.  ]) # confinement radius

all_LP  = []
all_ds = []
all_qs = []
all_ls = []

for track_length in track_lengths:
    for conf_factor in conf_factors:
        print(track_length, conf_factor)
        
        tracks = anomalous_diff_mixture(track_len=track_length,
                                   nb_tracks = 10000,
                                    LocErr=0.02, # localization error in x, y and z (even if not used)
                                    nb_states = 1,
                                    Fs = np.array([1]),
                                    Ds = np.array([0.25]),
                                    nb_dims = 2,
                                    velocities = np.array([0]),
                                    angular_Ds = np.array([0.0]),
                                    conf_forces = np.array([conf_factor]),
                                    conf_Ds = np.array([0.0]),
                                    conf_dists = np.array([0]),
                                    LocErr_std = 0,
                                    dt = 0.02,
                                    nb_sub_steps = 20)
        
        tracks = np.array(tracks)
        
        _, _, LP0 = Brownian_fit(tracks, verbose  = 0, output_type = 'values')
        
        est_LocErrs, est_ds, est_qs, est_ls, LP = Confined_fit(tracks, Fixed_LocErr = True, verbose  = 0, output_type = 'values')
        
        all_LP.append(LP - LP0)
        all_ds.append(est_ds)
        all_qs.append(est_qs)
        all_ls.append(est_ls)
        
        np.save(sav_dir + 'all_LP.npy', all_LP)
        np.save(sav_dir + 'all_ds.npy', all_ds)
        np.save(sav_dir + 'all_qs.npy', all_qs)
        np.save(sav_dir + 'all_ls.npy', all_ls)

all_ds = np.load(sav_dir + 'all_ds.npy')
all_qs = np.load(sav_dir + 'all_qs.npy')
all_ls = np.load(sav_dir + 'all_ls.npy')
all_LP = np.load(sav_dir + 'all_LP.npy')

mean_all_LP = np.mean(all_LP.reshape(len(track_lengths), len(conf_factors), 10000), 2).T
mean_pvalue = np.mean(np.exp(-all_LP.reshape(len(track_lengths), len(conf_factors), 10000)), 2).T

plt.figure(figsize = (4,3.3))
sns.heatmap(mean_all_LP/np.log(10), vmin = 0, vmax = 3, cmap = 'jet_r', xticklabels = track_lengths,  yticklabels = conf_factors)
plt.gca().invert_yaxis()
plt.xlabel('Number of time points')
plt.ylabel('Confinement factor')
plt.gca().set_aspect('equal', adjustable='box')
#plt.tight_layout()
plt.savefig(sav_dir + 'heatmap_log10_likeliood_ratio2.svg', format = 'svg')

plt.figure(figsize = (4,3.3))
sns.heatmap(mean_pvalue, vmin = 0, vmax = 0.2, cmap = 'PuRd', xticklabels = track_lengths,  yticklabels = conf_factors)
plt.gca().invert_yaxis()
plt.xlabel('Number of time points')
plt.ylabel('Confinement factor')
plt.gca().set_aspect('equal', adjustable='box')
#plt.tight_layout()
plt.savefig(sav_dir + 'heatmap_pvalue_likeliood_ratio2.svg', format = 'svg')


corr_all_ls = -np.log(1-all_ls)
mean_all_ls = np.mean(corr_all_ls.reshape(len(track_lengths), len(conf_factors), 10000), 2).T

plt.figure(figsize = (3.9,3.1))
sns.heatmap(mean_all_ls, vmin = 0, vmax = 1, cmap = 'jet', xticklabels = track_lengths,  yticklabels = conf_factors)
plt.gca().invert_yaxis()
plt.xlabel('Number of time points')
plt.ylabel('Confinement factor')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(sav_dir + 'heatmap_corrected_ls.svg', format = 'svg')

mean_all_ds = np.mean(all_ds.reshape(len(track_lengths), len(conf_factors), 10000), 2).T

plt.figure(figsize = (3.9,3.1))
sns.heatmap(mean_all_ds, vmin = 0, vmax = 0.2, cmap = 'BrBG', xticklabels = track_lengths,  yticklabels = conf_factors)
plt.gca().invert_yaxis()
plt.xlabel('Number of time points')
plt.ylabel('Confinement factor')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(sav_dir + 'heatmap_ds.svg', format = 'svg')

cor_all_ds = (1-np.exp(-corr_all_ls))/corr_all_ls * all_ds
cor_all_ds = all_ls/corr_all_ls * all_ds
mean_cor_all_ds = np.mean(cor_all_ds.reshape(len(track_lengths), len(conf_factors), 10000), 2).T

plt.figure(figsize = (3.9,3.1))
sns.heatmap(mean_cor_all_ds, vmin = 0, vmax = 0.2, cmap = 'PuOr', xticklabels = track_lengths,  yticklabels = conf_factors)
plt.gca().invert_yaxis()
plt.xlabel('Number of time points')
plt.ylabel('Confinement factor')
plt.gca().set_aspect('equal', adjustable='box')
#plt.tight_layout()
plt.savefig(sav_dir + 'heatmap_corrected_ds.svg', format = 'svg')

mean_all_qs = np.mean(all_qs.reshape(len(track_lengths), len(conf_factors), 10000), 2).T

plt.figure(figsize = (3.9,3.1))
sns.heatmap(mean_all_qs, vmin = 0, vmax = 0.02, cmap = 'jet', xticklabels = track_lengths,  yticklabels = conf_factors)
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('Number of time points')
plt.ylabel('Confinement factor')
#plt.tight_layout()
plt.savefig(sav_dir + 'heatmap_qs.svg', format = 'svg')


all_conf_rads = (cor_all_ds*0.02/corr_all_ls)**0.5
mean_all_conf_radius = np.mean(all_conf_rads.reshape(len(track_lengths), len(conf_factors), 10000), 2).T

true_conf_rads = (0.1*0.02/conf_factors)**0.5

plt.figure(figsize = (3.9,3.1))
sns.heatmap(mean_all_conf_radius, vmin = 0, vmax = 0.25, cmap = 'jet', xticklabels = track_lengths,  yticklabels = np.round(true_conf_rads, 3))
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('Number of time points')
plt.ylabel('Confinement radius [$\mathrm{\mu m}$]')
#plt.tight_layout()
plt.savefig(sav_dir + 'heatmap_conf_radius.svg', format = 'svg')




plt.figure(figsize = (3.9,3.1))
sns.heatmap(mean_all_conf_radius - true_conf_rads[:, None], vmin = -0.02, vmax = 0.02, cmap = 'PuOr', xticklabels = track_lengths,  yticklabels = np.round(true_conf_rads, 3))
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('Number of time points')
plt.ylabel('Confinement radius [$\mathrm{\mu m}$]')
#plt.tight_layout()
plt.savefig(sav_dir + 'heatmap_conf_radius_bias.svg', format = 'svg')


corr_all_ls = -np.log(1-all_ls)
rainbow_plot(corr_all_ls.reshape(len(track_lengths), len(conf_factors), 10000)[:, 5], ylabels = track_lengths, bins = np.arange(0,0.501,0.01), heigth_ratio = 0.9, figsize = (3.5,3.5), density = True, same_x_y = False, size_ratio = 2, max_offset = 0.8)
plt.xlabel('Estimated confinement factor')
plt.ylabel('Number of time points')
plt.grid(zorder=-30)
plt.savefig(sav_dir + 'rainbow_confinement_factor025_var_track_len.svg', format = 'svg')

rainbow_plot(all_LP.reshape(len(track_lengths), len(conf_factors), 10000)[-1], ylabels = conf_factors, bins = None, heigth_ratio = 1, figsize = (3.5,3.5), density = True, same_x_y = False, size_ratio = 2, max_offset = 2)
rainbow_plot(all_LP.reshape(len(track_lengths), len(conf_factors), 10000)[:, 9], ylabels = track_lengths, bins = None, heigth_ratio = 1, figsize = (3.5,3.5), density = True, same_x_y = False, size_ratio = 2, max_offset = 2)



#%%

'''
Vary both confinement radius and track length of confined tracks, fig 3d and fig 3-sup fig1e
'''

sav_dir = r'D:\anomalous/confined_var_track_len_conf_factor_2/'
if not os.path.exists(sav_dir):
    os.makedirs(sav_dir)
    

track_lengths = [10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 400, 600, 1000]
rads = np.array([0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3])

0.25*0.02/rads**2

all_LP  = []
all_ds = []
all_qs = []
all_ls = []

for track_length in track_lengths:
    for r in rads:
        print(track_length, r)
        conf_factor = 0.25*0.02/r**2
        
        tracks = anomalous_diff_mixture(track_len=track_length,
                                   nb_tracks = 10000,
                                    LocErr=0.02, # localization error in x, y and z (even if not used)
                                    nb_states = 1,
                                    Fs = np.array([1]),
                                    Ds = np.array([0.25]),
                                    nb_dims = 2,
                                    velocities = np.array([0]),
                                    angular_Ds = np.array([0.0]),
                                    conf_forces = np.array([conf_factor]),
                                    conf_Ds = np.array([0.0]),
                                    conf_dists = np.array([0]),
                                    LocErr_std = 0,
                                    dt = 0.02,
                                    nb_sub_steps = 50)
        
        tracks = np.array(tracks)
        
        _, _, LP0 = Brownian_fit(tracks, nb_dims = 2, verbose  = 0)
        
        est_LocErrs, est_ds, est_qs, est_ls, LP = Confined_fit(tracks, Fixed_LocErr = True, verbose  = 0, output_type = 'values')
        
        all_LP.append(LP - LP0)
        all_ds.append(est_ds)
        all_qs.append(est_qs)
        all_ls.append(est_ls)
        
        np.save(sav_dir + 'all_LP.npy', all_LP)
        np.save(sav_dir + 'all_ds.npy', all_ds)
        np.save(sav_dir + 'all_qs.npy', all_qs)
        np.save(sav_dir + 'all_ls.npy', all_ls)

all_ds = np.load(sav_dir + 'all_ds.npy')
all_qs = np.load(sav_dir + 'all_qs.npy')
all_ls = np.load(sav_dir + 'all_ls.npy')
all_LP = np.load(sav_dir + 'all_LP.npy')

conf_factors = 0.25*0.02/rads**2

label_rads = []
for r in rads:
    label_rads.append((str(r) + '00')[:4])

corr_all_ls = -np.log(1-all_ls)
mean_all_ls = np.mean(corr_all_ls.reshape(len(track_lengths), len(rads), 10000), 2).T

plt.figure(figsize = (4.1,2.8))
sns.heatmap(mean_all_ls, vmin = 0, vmax = 1, cmap = 'jet', xticklabels = track_lengths,  yticklabels = label_rads)
plt.gca().invert_yaxis()
plt.xlabel('Number of time points')
plt.ylabel('Confinement radius [$\mathrm{\mu m}$]')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(sav_dir + 'heatmap_corrected_ls.svg', format = 'svg')

plt.figure(figsize = (4.1,2.8))
sns.heatmap((mean_all_ls - conf_factors[:,None])/conf_factors[:,None], vmin = -0.5, vmax = 0.5, cmap = 'PuOr', xticklabels = track_lengths,  yticklabels = label_rads)
plt.gca().invert_yaxis()
plt.xlabel('Number of time points')
plt.ylabel('Confinement radius [$\mathrm{\mu m}$]')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(sav_dir + 'heatmap_corrected_ls_bais.svg', format = 'svg')



mean_all_ds = np.mean(all_ds.reshape(len(track_lengths), len(rads), 10000), 2).T

plt.figure(figsize = (4.1,2.8))
sns.heatmap(mean_all_ds, vmin = 0, vmax = 0.2, cmap = 'BrBG', xticklabels = track_lengths,  yticklabels = label_rads)
plt.gca().invert_yaxis()
plt.xlabel('Number of time points')
plt.ylabel('Confinement radius [$\mathrm{\mu m}$]')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(sav_dir + 'heatmap_ds.svg', format = 'svg')

cor_all_ds = (1-np.exp(-corr_all_ls))/corr_all_ls * all_ds
cor_all_ds = all_ls/corr_all_ls * all_ds
mean_cor_all_ds = np.mean(cor_all_ds.reshape(len(track_lengths), len(rads), 10000), 2).T

plt.figure(figsize = (4.1,2.8))
sns.heatmap(mean_cor_all_ds, vmin = 0, vmax = 0.2, cmap = 'PuOr', xticklabels = track_lengths,  yticklabels = label_rads)
plt.gca().invert_yaxis()
plt.xlabel('Number of time points')
plt.ylabel('Confinement radius [$\mathrm{\mu m}$]')
plt.gca().set_aspect('equal', adjustable='box')
#plt.tight_layout()
plt.savefig(sav_dir + 'heatmap_corrected_ds.svg', format = 'svg')

mean_all_qs = np.mean(all_qs.reshape(len(track_lengths), len(rads), 10000), 2).T

plt.figure(figsize = (4.1,2.8))
sns.heatmap(mean_all_qs, vmin = 0, vmax = 0.02, cmap = 'jet', xticklabels = track_lengths,  yticklabels = label_rads)
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('Number of time points')
plt.ylabel('Confinement radius [$\mathrm{\mu m}$]')
#plt.tight_layout()
plt.savefig(sav_dir + 'heatmap_qs.svg', format = 'svg')


all_conf_rads = (cor_all_ds*0.02/corr_all_ls)**0.5
all_conf_rads = cor_all_ds/(2*corr_all_ls)**0.5

mean_all_conf_radius = np.mean(all_conf_rads.reshape(len(track_lengths), len(rads), 10000), 2).T

mean_all_conf_radius.shape
len(label_rads)
mean_all_conf_radius[-1]

plt.figure(figsize = (4.1,2.8))
sns.heatmap(mean_all_conf_radius, vmin = 0, vmax = 0.3, cmap = 'jet', xticklabels = track_lengths,  yticklabels = label_rads)
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('Number of time points')
plt.ylabel('Confinement radius [$\mathrm{\mu m}$]')
#plt.tight_layout()
plt.savefig(sav_dir + 'heatmap_conf_radius.svg', format = 'svg')


plt.figure(figsize = (4.1,2.8))
sns.heatmap((mean_all_conf_radius - rads[:, None])/rads[:, None], vmin = -0.3, vmax = 0.3, cmap = 'PuOr', xticklabels = track_lengths,  yticklabels = label_rads)
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('Number of time points')
plt.ylabel('Confinement radius [$\mathrm{\mu m}$]')
#plt.tight_layout()
plt.savefig(sav_dir + 'heatmap_conf_radius_rel_bias.svg', format = 'svg')


plt.figure(figsize = (4.1,2.8))
sns.heatmap((mean_all_conf_radius - rads[:, None]), vmin = -0.1, vmax = 0.1, cmap = 'PuOr', xticklabels = track_lengths,  yticklabels = label_rads)
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('Number of time points')
plt.ylabel('Confinement radius [$\mathrm{\mu m}$]')
#plt.tight_layout()
plt.savefig(sav_dir + 'heatmap_conf_radius_bias.svg', format = 'svg')


mean_all_conf_radius.shape
rads[:, None].shape

corr_all_ls = -np.log(1-all_ls)
rainbow_plot(corr_all_ls.reshape(len(track_lengths), len(label_rads), 10000)[:, 5], ylabels = track_lengths, bins = np.arange(0,0.501,0.01), heigth_ratio = 0.9, figsize = (3.5,3.5), density = True, same_x_y = False, size_ratio = 2, max_offset = 0.8)
plt.xlabel('Estimated confinement factor')
plt.ylabel('Number of time points')
plt.grid(zorder=-30)
plt.savefig(sav_dir + 'rainbow_confinement_factor025_var_track_len.svg', format = 'svg')

corr_all_ls = -np.log(1-all_ls)
rainbow_plot(corr_all_ls.reshape(len(track_lengths), len(label_rads), 10000)[12, :], ylabels = conf_factors, bins = np.arange(0,3,0.01), heigth_ratio = 0.9, figsize = (3.5,3.5), density = True, same_x_y = False, size_ratio = 2, max_offset = 0.8)
plt.xlabel('Estimated confinement factor')
plt.ylabel('Number of time points')
plt.grid(zorder=-30)
plt.savefig(sav_dir + 'rainbow_confinement_factor025_var_track_len.svg', format = 'svg')

corr_all_ls = -np.log(1-all_ls)
rainbow_plot(all_conf_rads.reshape(len(track_lengths), len(rads), 10000)[12], ylabels = rads, bins = np.arange(0,0.4,0.005), heigth_ratio = 0.7, figsize = (3.5,3.5), density = True, same_x_y = False, size_ratio = 1.5, max_offset = 0.8)
plt.xlabel('Estimated confinement radius')
plt.ylabel('Number of time points')
plt.grid(zorder=-30)
plt.savefig(sav_dir + 'rainbow_confinement_radius_var_radius.svg', format = 'svg')


Fracs = np.mean(all_LP.reshape(len(track_lengths), len(label_rads), 10000) > np.log(20), 2).T

plt.figure(figsize = (4.1,2.8))
sns.heatmap(Fracs, vmin = 0, vmax = 1, cmap = 'jet', xticklabels = track_lengths,  yticklabels = label_rads)
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('Number of time points')
plt.ylabel('Confinement radius [$\mathrm{\mu m}$]')
#plt.tight_layout()
plt.savefig(sav_dir + 'heatmap_Fraction_95perc.svg', format = 'svg')








'''
Varying both the track length and the velocity of directed tracks, fig 4c and supplement 1b
'''

sav_dir = r'D:\anomalous/directed_var_track_len_velocity/'
if not os.path.exists(sav_dir):
    os.makedirs(sav_dir)
    
track_lengths = np.array([4, 5, 6, 8, 10, 15,  20,  30,  45, 70, 100, 200, 300])
velocities = np.array([0, 0.002, 0.004, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 1])

all_LP  = []
all_ds = []
all_qs = []
all_ls = []

for track_length in track_lengths:
    for velocity in velocities:
        print(track_length, velocity)
        
        tracks = anomalous_diff_mixture(track_len = int(track_length),
                                                    nb_tracks = 10000,
                                                    LocErr=0.02, # localization error in x, y and z (even if not used)
                                                    nb_states = 1,
                                                    Fs = np.array([1]),
                                                    Ds = np.array([0.0]),
                                                    nb_dims = 2,
                                                    velocities = np.array([velocity]),
                                                    angular_Ds = np.array([0.0]),
                                                    conf_forces = np.array([0]),
                                                    conf_Ds = np.array([0.0]),
                                                    conf_dists = np.array([0]),
                                                    LocErr_std = 0,
                                                    dt = 0.02,
                                                    nb_sub_steps = 20)
        
        tracks = np.array(tracks, dtype = 'float32')
        
        _, _, LP0 = Brownian_fit(tracks, verbose  = 0, output_type = 'values')
        
        est_LocErrs, est_ds, est_qs, est_ls, LP, mean_pred_vs = Directed_fit(tracks, Fixed_LocErr = True, verbose  = 0, output_type = 'values')
        
        all_LP.append(LP - LP0)
        all_ds.append(est_ds)
        all_qs.append(est_qs)
        all_ls.append(est_ls)
        
        np.save(sav_dir + 'all_LP.npy', all_LP)
        np.save(sav_dir + 'all_ds.npy', all_ds)
        np.save(sav_dir + 'all_qs.npy', all_qs)
        np.save(sav_dir + 'all_ls.npy', all_ls)

all_ds = np.load(sav_dir + 'all_ds.npy')
all_qs = np.load(sav_dir + 'all_qs.npy')
all_ls = np.load(sav_dir + 'all_ls.npy')
all_LP = np.load(sav_dir + 'all_LP.npy')


mean_all_LP = np.mean(all_LP.reshape(len(track_lengths), len(velocities), 10000), 2).T
mean_pvalue = np.mean(np.exp(-all_LP.reshape(len(track_lengths), len(velocities), 10000)), 2).T

plt.figure(figsize = (3.7,3.3))
sns.heatmap(mean_all_LP/np.log(10), vmin = 0, vmax = 3, cmap = 'jet_r', xticklabels = track_lengths,  yticklabels = velocities)
plt.gca().invert_yaxis()
plt.xlabel('Number of time points')
plt.ylabel('Velocity [$\mathrm{\mu m.\Delta t^{-1}}$]')
plt.gca().set_aspect('equal', adjustable='box')
#plt.tight_layout()
plt.savefig(sav_dir + 'heatmap_log10_likeliood_ratio.svg', format = 'svg')

plt.figure(figsize = (3.7,3.3))
sns.heatmap(mean_pvalue, vmin = 0, vmax = 0.2, cmap = 'PuRd', xticklabels = track_lengths,  yticklabels = velocities)
plt.gca().invert_yaxis()
plt.xlabel('Number of time points')
plt.ylabel('Velocity [$\mathrm{\mu m.\Delta t^{-1}}$]')
plt.gca().set_aspect('equal', adjustable='box')
#plt.tight_layout()
plt.savefig(sav_dir + 'heatmap_pvalue_likeliood_ratio.svg', format = 'svg')

mean_all_ls = np.mean(all_ls.reshape(len(track_lengths), len(velocities), 10000), 2).T

plt.figure(figsize = (3.9,3.3))
sns.heatmap((mean_all_ls - velocities[:,None])/velocities[:,None], vmin = -0.2, vmax = 0.2, cmap = 'PuOr', xticklabels = track_lengths,  yticklabels = velocities)
plt.gca().invert_yaxis()
plt.xlabel('Number of time points')
plt.ylabel('True velocity [$\mathrm{\mu m.\Delta t^{-1}}$]')
plt.gca().set_aspect('equal', adjustable='box')
#plt.gca().ticklabel_format(axis='both', style='plain', useOffset=False)
#plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
plt.savefig(sav_dir + 'heatmap_ls.svg', format = 'svg')


mean_all_ds = np.mean(all_ds.reshape(len(track_lengths), len(velocities), 10000), 2).T

plt.figure(figsize = (3.7,3.3))
sns.heatmap(mean_all_ds, vmin = 0, vmax = 0.01, cmap = 'jet', xticklabels = track_lengths,  yticklabels = velocities)
plt.gca().invert_yaxis()
plt.xlabel('Number of time points')
plt.ylabel('Velocity [$\mathrm{\mu m.\Delta t^{-1}}$]')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(sav_dir + 'heatmap_ds.svg', format = 'svg')

mean_all_qs = np.mean(all_qs.reshape(len(track_lengths), len(velocities), 10000), 2).T

plt.figure(figsize = (4.1,3.3))
sns.heatmap(mean_all_qs, vmin = 0, vmax = 0.01, cmap = 'jet', xticklabels = track_lengths,  yticklabels = velocities)
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('Number of time points')
plt.ylabel('Velocity [$\mathrm{\mu m.\Delta t^{-1}}$]')
#plt.tight_layout()
plt.savefig(sav_dir + 'heatmap_qs.svg', format = 'svg')

rainbow_plot(all_ls.reshape(len(track_lengths), len(velocities), 10000)[6], ylabels = velocities, bins = None, heigth_ratio = 0.9, figsize = (3.5,3.5), density = True, same_x_y = False, size_ratio = 2, max_offset = 0.8)
plt.xlabel('Estimated velocity [$\mathrm{\mu m.\Delta t^{-1}}$]')
plt.ylabel('True velocity [$\mathrm{\mu m.\Delta t^{-1}}$]')
plt.grid(zorder=-30)
plt.savefig(sav_dir + 'rainbow_velocities_track_len=20.svg', format = 'svg')




'''
Varying the speed on the potential well of confined tracks, fig 3e and fig 3 sup 1g
'''

conf_d_vals = np.arange(0, 0.0501, 0.005)

(0.02*0.25*2)**0.5
dt = 0.02
all_tracks = []

for d in conf_d_vals:
    
    tracks = anomalous_diff_mixture(track_len=400,
                               nb_tracks = 10000,
                                LocErr=0.02, # localization error in x, y and z (even if not used)
                                nb_states = 1,
                                Fs = np.array([1]),
                                Ds = np.array([0.25]),
                                nb_dims = 2,
                                velocities = np.array([0.0]),
                                angular_Ds = np.array([0.0]),
                                conf_forces = np.array([0.25]),
                                conf_Ds = np.array([d**2/(2*dt)]),
                                conf_dists = np.array([0]),
                                LocErr_std = 0,
                                dt = dt,
                                nb_sub_steps = 20)
    
    tracks = np.array(tracks)
    all_tracks.append(tracks)
    

sav_dir = r'D:\anomalous/track_plots/'
if not os.path.exists(sav_dir):
    os.makedirs(sav_dir)

v = 0.05

tracks = anomalous_diff_mixture(track_len=400,
                           nb_tracks = 10,
                            LocErr=0.02, # localization error in x, y and z (even if not used)
                            nb_states = 1,
                            Fs = np.array([1]),
                            Ds = np.array([0.25]),
                            nb_dims = 2,
                            velocities = np.array([0.0]),
                            angular_Ds = np.array([0.0]),
                            conf_forces = np.array([0.25]),
                            conf_Ds = np.array([v**2/(2*0.02)]),
                            conf_dists = np.array([0]),
                            LocErr_std = 0,
                            dt = 0.02,
                            nb_sub_steps = 20)

k = 2
track = tracks[k]
lim = 1.5
plt.figure(figsize = (3,3))
plt.plot(track[:,0], track[:,1])
plt.xlim([np.mean(track[:,0])-lim, np.mean(track[:,0])+lim])
plt.ylim([np.mean(track[:,1])-lim, np.mean(track[:,1])+lim])
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(sav_dir + 'confined_moving_well_track_400_steps_q=' + str(v) + '_%s.svg'%k, format = 'svg')


sav_dir = r'D:\anomalous/rainbow_conf_var_conf_speed/'
if not os.path.exists(sav_dir):
    os.makedirs(sav_dir)

np.save(sav_dir + 'tracks', all_tracks)
all_tracks = np.load(sav_dir + 'tracks.npy')

all_LocErrs = []
all_ds = []
all_qs = []
all_ls = []
all_LP  = []

for tracks in all_tracks:
    _, _, LP0 = Brownian_fit(tracks, fixed_LocErr = True, output_type = 'values')
    est_LocErrs, est_ds, est_qs, est_ls, LP = Confined_fit(tracks, Fixed_LocErr = True, output_type = 'values', )
    
    all_LocErrs.append(est_LocErrs)
    all_ds.append(est_ds)
    all_qs.append(est_qs)
    all_ls.append(-np.log(1-est_ls))
    all_LP.append(LP - LP0)

np.save(sav_dir + 'all_ds', all_ds)
np.save(sav_dir + 'all_qs', all_qs)
np.save(sav_dir + 'all_ls', all_ls)
np.save(sav_dir + 'all_LP', all_LP)

all_ds = np.load(sav_dir + 'all_ds.npy')
all_qs = np.load(sav_dir + 'all_qs.npy')
all_ls = np.load(sav_dir + 'all_ls.npy')
all_LP = np.load(sav_dir + 'all_LP.npy')

cor_all_ds = (1-np.exp(-all_ls))/all_ls*all_ds

names = ['ds.svg', 'qs.svg', 'ls.svg', 'LP.svg']
all_lists = [cor_all_ds, all_qs, all_ls, all_LP]
xlabels = ['Estimated diffusion length per step [$\mathrm{\mu m.\Delta t^{-1}}$]', 'Estimated diffusion length of the potential well [$\mathrm{\mu m.\Delta t^{-1}}$]', 'Estimated confinement factor', 'likelihood ratio (scale: log$_{10}$)']
all_bins =  [None, None, np.arange(-0.05, 1.301, 0.01), None, ]
sames = [False, False, False, False]

for hists, xlabel, bins, name, same in zip(all_lists, xlabels, all_bins, names, sames):
    
    rainbow_plot(hists, ylabels = np.round(conf_d_vals, 3), bins = bins, heigth_ratio = 0.9, figsize = (2.8,3.5), density = True, same_x_y = same, size_ratio = 2, max_offset = 0.7)
    plt.ylabel('True diffusion length of the potential well [$\mathrm{\mu m}$]')
    plt.xlabel(xlabel)
    #plt.tight_layout()
    plt.grid(zorder=-30)
    plt.savefig(sav_dir + name, format = 'svg')


true_conf_radius = (0.1*0.02/0.25)**0.5
all_conf_radius = (cor_all_ds*0.02/all_ls)**0.5

rainbow_plot(all_conf_radius, ylabels = np.round(conf_d_vals, 3), bins = np.arange(0,0.15, 0.005), heigth_ratio = 0.9, figsize = (2.8,3.5), density = True, same_x_y = False, size_ratio = 2, max_offset = 0.7)
plt.ylabel('True diffusion length of the potential well [$\mathrm{\mu m}$]')
plt.xlabel('Estimated confinement radius [$\mathrm{\mu m}$]')
#plt.tight_layout()
plt.grid(zorder=-30)
plt.vlines(true_conf_radius, -100, 10000, zorder=100, linestyle = ':', color = (0.5,0.5,0.5), linewidth = 3)
plt.savefig(sav_dir + 'conf_radius.svg', format = 'svg')


rainbow_plot(all_qs, ylabels = np.round(conf_d_vals, 3), bins = np.arange(-0.01,0.07, 0.001), heigth_ratio = 0.9, figsize = (2.8,3.5), density = True, same_x_y = True, size_ratio = 2, max_offset = 0.7)
plt.ylabel('True diffusion length of the potential well [$\mathrm{\mu m}$]')
plt.xlabel('Estimated diffusion length of the potential well [$\mathrm{\mu m.\Delta t^{-1}}$]')
#plt.tight_layout()
plt.grid(zorder=-30)
plt.xticks(np.arange(-0.0,0.065, 0.01), rotation  = 90)

plt.savefig(sav_dir + 'qs.svg', format = 'svg')


#%%

'''
Directed motion with varying orientation changes, fig 4f
'''

orientation_vals = np.arange(0, 1.21, 0.1)
(0.02*0.25*2)**0.5

0.1**2/(2*0.02)

all_LocErrs = []
all_ds = []
all_qs = []
all_ls = []
all_LP  = []

all_tracks = []

for l in orientation_vals:
    print(l)
    
    tracks = anomalous_diff_mixture(track_len=200,
                               nb_tracks = 10000,
                                LocErr=0.02, # localization error in x, y and z (even if not used)
                                nb_states = 1,
                                Fs = np.array([1]),
                                Ds = np.array([0.0]),
                                nb_dims = 2,
                                velocities = np.array([0.1]), # 0.1 for the original data 
                                angular_Ds = np.array([l**2/(2*0.02)]),
                                conf_forces = np.array([0]),
                                conf_Ds = np.array([0.0]),
                                conf_dists = np.array([0]),
                                LocErr_std = 0,
                                dt = 0.02,
                                nb_sub_steps = 20)
    
    tracks = np.array(tracks)
    all_tracks.append(tracks)

val_names = ['0', '$\pi / 10$', '$2\pi / 10$', '$3\pi / 10$', '$4\pi / 10$', '$5\pi / 10$', '$6\pi / 10$', '$7\pi / 10$', '$8\pi / 10$', '$9\pi / 10$', '$pi$']
val_names = ['0', '$\pi / 10$', '$2/ 10\pi$', '$3/ 10\pi$', '$4/ 10\pi$', '$5/ 10\pi$', '$6/ 10\pi$', '$7/ 10\pi$', '$8/ 10\pi$', '$9/ 10\pi$', '$pi$']

sav_dir = r'D:\anomalous/rainbow_variable_orientation/'
if not os.path.exists(sav_dir):
    os.makedirs(sav_dir)
np.save(sav_dir + 'tracks', all_tracks)

all_tracks = np.load(sav_dir + 'tracks.npy')

for tracks in all_tracks:
    
    _, _, LP0 = Brownian_fit(tracks, verbose  = 0, output_type = 'values')
    #est_LocErrs1, est_ds1, est_qs1, est_ls1, LP1 = Directed_fit(tracks, nb_dims)
    
    est_LocErrs, est_ds, est_qs, est_ls, LP, mean_pred_vs = Directed_fit(tracks, Fixed_LocErr = True, verbose  = 0, output_type = 'values')
    
    all_LocErrs.append(est_LocErrs)
    all_ds.append(est_ds)
    all_qs.append(est_qs)
    all_ls.append(est_ls)
    print(all_ls, all_qs, all_ds)
    all_LP.append(LP - LP0)

np.save(sav_dir + 'all_LocErrs', all_LocErrs)
np.save(sav_dir + 'all_ds', all_ds)
np.save(sav_dir + 'all_qs', all_qs)
np.save(sav_dir + 'all_ls', all_ls)
np.save(sav_dir + 'all_LP', all_LP)

all_ds = np.load(sav_dir + 'all_ds.npy')
all_qs = np.load(sav_dir + 'all_qs.npy')
all_ls = np.load(sav_dir + 'all_ls.npy')
all_LP = np.load(sav_dir + 'all_LP.npy')

all_cor_ratio = -np.log(1-all_qs/(2*0.02))

all_lists = [all_ds, 2*np.arcsin(2**0.5*all_cor_ratio), all_ls, all_LP]
names = ['ds.svg', 'qs.svg', 'ls.svg', 'LP.svg']
xlabels = ['Estimated diffusion length [$\mathrm{\mu m.\Delta t^{-1}}$]', 'Estimated change of orientation per step [$\mathrm{\mu m.\Delta t^{-1}}$]', 'Estimated speed [$\mathrm{\mu m.\Delta t^{-1}}$]', 'likelihood ratio (scale: log$_{10}$)']
all_bins =  [None, None, None, None]
same_xys = [False, False, False, False]

for hists, xlabel, bins, same, name in zip(all_lists, xlabels, all_bins, same_xys, names):
    rainbow_plot(hists, ylabels = np.round(orientation_vals, 1), bins = bins, heigth_ratio = 0.9, figsize = (3.5,3.5), density = True, same_x_y = same, size_ratio = 2, max_offset = 0.7)
    plt.ylabel('True change of orientation per step [$\mathrm{\mu m.\Delta t^{-1}}$]')
    plt.xlabel(xlabel)
    #plt.tight_layout()
    plt.grid(zorder=-30)
    plt.savefig(sav_dir + name, format = 'svg')

rainbow_plot(hists/np.log(10), ylabels = np.round(orientation_vals, 1), bins = bins, heigth_ratio = 0.4, figsize = (3.5,3.5), density = True, same_x_y = same)
plt.ylabel('True speed [$\mathrm{\mu m.\Delta t^{-1}}$]')
plt.xlabel(xlabel)
plt.tight_layout()
plt.grid(zorder=-30)



'''
Motion with both a linear component and a diffusive component varying the diffusion length, fig 4g and fig 4 sup 2b
'''

sav_dir = r'D:\anomalous/directed_plus_diffusive_var_track_len_diffusion/'
if not os.path.exists(sav_dir):
    os.makedirs(sav_dir)
    
track_lengths = np.array([20, 30, 50, 70, 100, 150,  200, 300, 400, 600])
ds = np.array([0, 0.005, 0.01, 0.02, 0.04, 0.06, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1])

all_LP  = []
all_ds = []
all_qs = []
all_ls = []

for track_length in track_lengths:
    for d in ds:
        print(track_length, d)
        
        tracks = anomalous_diff_mixture(track_len = int(track_length),
                                                    nb_tracks = 10000,
                                                    LocErr=0.02, # localization error in x, y and z (even if not used)
                                                    nb_states = 1,
                                                    Fs = np.array([1]),
                                                    Ds = np.array([d**2/(2*0.02)]),
                                                    nb_dims = 2,
                                                    velocities = np.array([0.1]),
                                                    angular_Ds = np.array([0.0]),
                                                    conf_forces = np.array([0]),
                                                    conf_Ds = np.array([0.0]),
                                                    conf_dists = np.array([0]),
                                                    LocErr_std = 0,
                                                    dt = 0.02,
                                                    nb_sub_steps = 5)
        
        tracks = np.array(tracks, dtype = 'float32')
        
        _, _, LP0 = Brownian_fit(tracks, verbose  = 0, output_type = 'values')
        #est_LocErrs1, est_ds1, est_qs1, est_ls1, LP1 = Directed_fit(tracks, nb_dims)
        
        est_LocErrs, est_ds, est_qs, est_ls, LP, mean_pred_vs = Directed_fit(tracks, Fixed_LocErr = True, verbose  = 0, output_type = 'values')
        
        all_LP.append(LP - LP0)
        all_ds.append(est_ds)
        all_qs.append(est_qs)
        all_ls.append(est_ls)
        
        np.save(sav_dir + 'all_LP.npy', all_LP)
        np.save(sav_dir + 'all_ds.npy', all_ds)
        np.save(sav_dir + 'all_qs.npy', all_qs)
        np.save(sav_dir + 'all_ls.npy', all_ls)

all_ds = np.load(sav_dir + 'all_ds.npy')
all_qs = np.load(sav_dir + 'all_qs.npy')
all_ls = np.load(sav_dir + 'all_ls.npy')
all_LP = np.load(sav_dir + 'all_LP.npy')

mean_all_LP = np.mean(all_LP.reshape(len(track_lengths), len(ds), 10000), 2).T
mean_pvalue = np.mean(np.exp(-all_LP.reshape(len(track_lengths), len(ds), 10000)), 2).T

plt.figure(figsize = (3.7,3.3))
sns.heatmap(mean_all_LP/np.log(10), vmin = 0, vmax = 3, cmap = 'jet_r', xticklabels = track_lengths,  yticklabels =  np.round(ds, 3))
plt.gca().invert_yaxis()
plt.xlabel('Number of time points')
plt.ylabel('diffusion length [$\mathrm{\mu m.\Delta t^{-1}}$]')
plt.gca().set_aspect('equal', adjustable='box')
#plt.tight_layout()
plt.savefig(sav_dir + 'heatmap_log10_likeliood_ratio2.svg', format = 'svg')

plt.figure(figsize = (3.7,3.3))
sns.heatmap(mean_pvalue, vmin = 0, vmax = 0.2, cmap = 'PuRd', xticklabels = track_lengths,  yticklabels = np.round(ds, 3))
plt.gca().invert_yaxis()
plt.xlabel('Number of time points')
plt.ylabel('diffusion length [$\mathrm{\mu m.\Delta t^{-1}}$]')
plt.gca().set_aspect('equal', adjustable='box')
#plt.tight_layout()
plt.savefig(sav_dir + 'heatmap_pvalue_likeliood_ratio2.svg', format = 'svg')

mean_all_ds = np.mean(all_ds.reshape(len(track_lengths), len(ds), 10000), 2).T

plt.figure(figsize = (3.7,3.3))
sns.heatmap(mean_all_ds - ds[:, None], vmin = -0.02, vmax = 0.02, cmap = 'PuOr', xticklabels = track_lengths,  yticklabels =  np.round(ds, 3))
plt.gca().invert_yaxis()
plt.xlabel('Number of time points')
plt.ylabel('diffusion length [$\mathrm{\mu m.\Delta t^{-1}}$]')
plt.gca().set_aspect('equal', adjustable='box')
#plt.tight_layout()
plt.savefig(sav_dir + 'heatmap_ds_bias.svg', format = 'svg')

mean_all_ls = np.mean(all_ls.reshape(len(track_lengths), len(ds), 10000), 2).T

plt.figure(figsize = (3.7,3.3))
sns.heatmap(mean_all_ls, vmin = 0, vmax = 0.2, cmap = 'PuOr', xticklabels = track_lengths,  yticklabels =  np.round(ds, 3))
plt.gca().invert_yaxis()
plt.xlabel('Number of time points')
plt.ylabel('diffusion length [$\mathrm{\mu m.\Delta t^{-1}}$]')
plt.gca().set_aspect('equal', adjustable='box')
#plt.tight_layout()
plt.savefig(sav_dir + 'heatmap_ls.svg', format = 'svg')

mean_all_qs = np.mean(all_qs.reshape(len(track_lengths), len(ds), 10000), 2).T

plt.figure(figsize = (3.7,3.3))
sns.heatmap(mean_all_qs, vmin = 0, vmax = 0.02, cmap = 'jet', xticklabels = track_lengths,  yticklabels =  np.round(ds, 3))
plt.gca().invert_yaxis()
plt.xlabel('Number of time points')
plt.ylabel('diffusion length [$\mathrm{\mu m.\Delta t^{-1}}$]')
plt.gca().set_aspect('equal', adjustable='box')
#plt.tight_layout()
plt.savefig(sav_dir + 'heatmap_qs.svg', format = 'svg')

mean_err_ds = np.mean(np.abs(all_ds.reshape(len(track_lengths), len(ds), 10000)-ds[None,:, None]), 2).T

plt.figure(figsize = (3.7,3.3))
sns.heatmap(mean_err_ds, vmin = 0, vmax = 0.02, cmap = 'jet', xticklabels = track_lengths,  yticklabels =  np.round(ds, 3))
plt.gca().invert_yaxis()
plt.xlabel('Number of time points')
plt.ylabel('diffusion length [$\mathrm{\mu m.\Delta t^{-1}}$]')
plt.gca().set_aspect('equal', adjustable='box')
#plt.tight_layout()
plt.savefig(sav_dir + 'heatmap_MAE_ds.svg', format = 'svg')

mean_err_ls = np.mean(np.abs(all_ls.reshape(len(track_lengths), len(ds), 10000)-0.1), 2).T

plt.figure(figsize = (3.7,3.3))
sns.heatmap(mean_err_ls, vmin = 0, vmax = 0.05, cmap = 'jet', xticklabels = track_lengths,  yticklabels =  np.round(ds, 3))
plt.gca().invert_yaxis()
plt.xlabel('Number of time points')
plt.ylabel('diffusion length [$\mathrm{\mu m.\Delta t^{-1}}$]')
plt.gca().set_aspect('equal', adjustable='box')
#plt.tight_layout()
plt.savefig(sav_dir + 'heatmap_MAE_ls.svg', format = 'svg')




'''
Changing orientation varying velocity and speed of the change of orientation, fig 4e and fig 4 sup 2a
'''

sav_dir = r'D:\anomalous/directed_variable_changes_orientation_speed/'
if not os.path.exists(sav_dir):
    os.makedirs(sav_dir)
    
thetas = np.array([0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2])
vs = np.array([0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2])

all_LP  = []
all_ds = []
all_qs = []
all_ls = []

for theta in thetas:
    for v in vs:
        print(theta, v)
        
        tracks = anomalous_diff_mixture(track_len = int(200),
                                                    nb_tracks = 10000,
                                                    LocErr=0.02, # localization error in x, y and z (even if not used)
                                                    nb_states = 1,
                                                    Fs = np.array([1]),
                                                    Ds = np.array([0.]),
                                                    nb_dims = 2,
                                                    velocities = np.array([v]),
                                                    angular_Ds = np.array([theta**2/(2*0.02)]),
                                                    conf_forces = np.array([0]),
                                                    conf_Ds = np.array([0.0]),
                                                    conf_dists = np.array([0]),
                                                    LocErr_std = 0,
                                                    dt = 0.02,
                                                    nb_sub_steps = 5)
        
        tracks = np.array(tracks, dtype = 'float32')
        
        _, _, LP0 = Brownian_fit(tracks, verbose  = 0, output_type = 'values')
        #est_LocErrs1, est_ds1, est_qs1, est_ls1, LP1 = Directed_fit(tracks, nb_dims)
        
        est_LocErrs, est_ds, est_qs, est_ls, LP, mean_pred_vs = Directed_fit(tracks, Fixed_LocErr = True, verbose  = 0, output_type = 'values')
        
        all_LP.append(LP - LP0)
        all_ds.append(est_ds)
        all_qs.append(est_qs)
        all_ls.append(est_ls)
        
        np.save(sav_dir + 'all_LP.npy', all_LP)
        np.save(sav_dir + 'all_ds.npy', all_ds)
        np.save(sav_dir + 'all_qs.npy', all_qs)
        np.save(sav_dir + 'all_ls.npy', all_ls)

all_ds = np.load(sav_dir + 'all_ds.npy')
all_qs = np.load(sav_dir + 'all_qs.npy')
all_ls = np.load(sav_dir + 'all_ls.npy')
all_LP = np.load(sav_dir + 'all_LP.npy')

mean_all_LP = np.mean(all_LP.reshape(len(thetas), len(vs), 10000), 2).T
mean_pvalue = np.mean(np.exp(-all_LP.reshape(len(thetas), len(vs), 10000)), 2).T

plt.figure(figsize = (4,2.2))
sns.heatmap(mean_all_LP/np.log(10), vmin = 0, vmax = 3, cmap = 'jet_r', xticklabels = thetas,  yticklabels =  np.round(vs, 3))
plt.gca().invert_yaxis()
plt.xlabel('Rotational diffusion speed [$\mathrm{Rad m.\Delta t^{-1}}$]')
plt.ylabel('Velocity [$\mathrm{\mu m.\Delta t^{-1}}$]')
plt.gca().set_aspect('equal', adjustable='box')
#plt.tight_layout()
plt.savefig(sav_dir + 'heatmap_log10_likeliood_ratio2.svg', format = 'svg')

plt.figure(figsize = (4,2.2))
sns.heatmap(mean_pvalue, vmin = 0, vmax = 0.2, cmap = 'PuRd', xticklabels = thetas,  yticklabels = np.round(vs, 3))
plt.gca().invert_yaxis()
plt.xlabel('Rotational diffusion speed [$\mathrm{Rad m.\Delta t^{-1}}$]')
plt.ylabel('Velocity [$\mathrm{\mu m.\Delta t^{-1}}$]')
plt.gca().set_aspect('equal', adjustable='box')
#plt.tight_layout()
plt.savefig(sav_dir + 'heatmap_pvalue_likeliood_ratio2.svg', format = 'svg')

mean_all_ds = np.mean(all_ds.reshape(len(thetas), len(vs), 10000), 2).T

plt.figure(figsize = (4,2.2))
sns.heatmap(mean_all_ds, vmin = -0.0, vmax = 0.03, cmap = 'jet', xticklabels = thetas,  yticklabels =  np.round(vs, 3))
plt.gca().invert_yaxis()
plt.xlabel('Rotational diffusion speed [$\mathrm{Rad m.\Delta t^{-1}}$]')
plt.ylabel('Velocity [$\mathrm{\mu m.\Delta t^{-1}}$]')
plt.gca().set_aspect('equal', adjustable='box')
#plt.tight_layout()
plt.savefig(sav_dir + 'heatmap_ds.svg', format = 'svg')

mean_all_ls = np.mean(all_ls.reshape(len(thetas), len(vs), 10000), 2).T

plt.figure(figsize = (4,2.2))
sns.heatmap(mean_all_ls, vmin = 0, vmax = 0.2, cmap = 'jet', xticklabels = thetas,  yticklabels =  np.round(vs, 3))
plt.gca().invert_yaxis()
plt.xlabel('Rotational diffusion speed [$\mathrm{Rad m.\Delta t^{-1}}$]')
plt.ylabel('Velocity [$\mathrm{\mu m.\Delta t^{-1}}$]')
plt.gca().set_aspect('equal', adjustable='box')
#plt.tight_layout()
plt.savefig(sav_dir + 'heatmap_ls.svg', format = 'svg')

plt.figure(figsize = (4,2.2))
sns.heatmap(mean_all_ls - vs[:,None], vmin = -0.1, vmax = 0.1, cmap = 'PuOr', xticklabels = thetas,  yticklabels =  np.round(vs, 3))
plt.gca().invert_yaxis()
plt.xlabel('Rotational diffusion speed [$\mathrm{Rad m.\Delta t^{-1}}$]')
plt.ylabel('Velocity [$\mathrm{\mu m.\Delta t^{-1}}$]')
plt.gca().set_aspect('equal', adjustable='box')
#plt.tight_layout()
plt.savefig(sav_dir + 'heatmap_ls_bias.svg', format = 'svg')

plt.figure(figsize = (4,2.2))
sns.heatmap((mean_all_ls - vs[:,None])/vs[:,None], vmin = -0.5, vmax = 0.5, cmap = 'PuOr', xticklabels = thetas,  yticklabels =  np.round(vs, 3))
plt.gca().invert_yaxis()
plt.xlabel('Rotational diffusion speed [$\mathrm{Rad m.\Delta t^{-1}}$]')
plt.ylabel('Velocity [$\mathrm{\mu m.\Delta t^{-1}}$]')
plt.gca().set_aspect('equal', adjustable='box')
#plt.tight_layout()
plt.savefig(sav_dir + 'heatmap_ls_relative_bias.svg', format = 'svg')

mean_all_qs = np.mean(all_qs.reshape(len(thetas), len(vs), 10000), 2).T
all_cor_ratio = -np.log(1-mean_all_qs/(2*mean_all_ls))
mean_all_thetas = 2*np.arcsin(2**0.5*all_cor_ratio)

plt.figure(figsize = (4,2.2))
sns.heatmap(mean_all_thetas, vmin = 0, vmax = 1.5, cmap = 'jet', xticklabels = thetas,  yticklabels =  np.round(vs, 3))
plt.gca().invert_yaxis()
plt.xlabel('Rotational diffusion speed [$\mathrm{Rad m.\Delta t^{-1}}$]')
plt.ylabel('Velocity [$\mathrm{\mu m.\Delta t^{-1}}$]')
plt.gca().set_aspect('equal', adjustable='box')
#plt.tight_layout()
plt.savefig(sav_dir + 'heatmap_thetas.svg', format = 'svg')

plt.figure(figsize = (4,2.2))
sns.heatmap(mean_all_thetas - thetas[None], vmin = -0.3, vmax = 0.3, cmap = 'PuOr', xticklabels = thetas,  yticklabels =  np.round(vs, 3))
plt.gca().invert_yaxis()
plt.xlabel('Rotational diffusion speed [$\mathrm{Rad m.\Delta t^{-1}}$]')
plt.ylabel('Velocity [$\mathrm{\mu m.\Delta t^{-1}}$]')
plt.gca().set_aspect('equal', adjustable='box')
#plt.tight_layout()
plt.savefig(sav_dir + 'heatmap_thetas_bias.svg', format = 'svg')

plt.figure(figsize = (4,2.2))
sns.heatmap((mean_all_thetas - thetas[None])/thetas[None], vmin = -0.1, vmax = 0.1, cmap = 'PuOr', xticklabels = thetas,  yticklabels =  np.round(vs, 3))
plt.gca().invert_yaxis()
plt.xlabel('Rotational diffusion speed [$\mathrm{Rad m.\Delta t^{-1}}$]')
plt.ylabel('Velocity [$\mathrm{\mu m.\Delta t^{-1}}$]')
plt.gca().set_aspect('equal', adjustable='box')
#plt.tight_layout()
plt.savefig(sav_dir + 'heatmap_thetas_relative_bias.svg', format = 'svg')

mean_err_ds = np.mean(all_ds.reshape(len(thetas), len(vs), 10000), 2).T

plt.figure(figsize = (4,2.2))
sns.heatmap(mean_err_ds, vmin = 0, vmax = 0.02, cmap = 'jet', xticklabels = thetas,  yticklabels =  np.round(vs, 3))
plt.gca().invert_yaxis()
plt.xlabel('Rotational diffusion speed [$\mathrm{Rad m.\Delta t^{-1}}$]')
plt.ylabel('Velocity [$\mathrm{\mu m.\Delta t^{-1}}$]')
plt.gca().set_aspect('equal', adjustable='box')
#plt.tight_layout()
plt.savefig(sav_dir + 'heatmap_ds.svg', format = 'svg')

mean_err_ls = np.mean(np.abs(all_ls.reshape(len(thetas), len(vs), 10000)-vs[:,None]), 2).T

plt.figure(figsize = (4,2.2))
sns.heatmap(mean_err_ls, vmin = 0, vmax = 0.2, cmap = 'jet', xticklabels = thetas,  yticklabels =  np.round(vs, 3))
plt.gca().invert_yaxis()
plt.xlabel('Rotational diffusion speed [$\mathrm{Rad m.\Delta t^{-1}}$]')
plt.ylabel('Velocity [$\mathrm{\mu m.\Delta t^{-1}}$]')
plt.gca().set_aspect('equal', adjustable='box')
#plt.tight_layout()
plt.savefig(sav_dir + 'heatmap_MAE_ls.svg', format = 'svg')

mean_err_ls = np.mean(np.abs(all_ls.reshape(len(thetas), len(vs), 10000)-vs[:,None])/vs[:,None], 2).T

plt.figure(figsize = (4,2.2))
sns.heatmap(mean_err_ls, vmin = 0, vmax = 0.5, cmap = 'jet', xticklabels = thetas,  yticklabels =  np.round(vs, 3))
plt.gca().invert_yaxis()
plt.xlabel('Rotational diffusion speed [$\mathrm{Rad m.\Delta t^{-1}}$]')
plt.ylabel('Velocity [$\mathrm{\mu m.\Delta t^{-1}}$]')
plt.gca().set_aspect('equal', adjustable='box')
#plt.tight_layout()
plt.savefig(sav_dir + 'heatmap_MRE_ls.svg', format = 'svg')









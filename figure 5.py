# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:45:32 2023

@author: franc
"""

import numpy as np
from numba import njit, typed, prange, jit
from matplotlib import pyplot as plt
from matplotlib import cm
import time
import os
import seaborn as sns

import sys
sys.path.insert(0, r'path/to/atrack')
from atrack import Confined_fit_multi, Directed_fit_multi, Brownian_fit_multi, multi_fit
import tensorflow as tf

@njit
def anomalous_diff_mixture(track_len=640,
                           nb_tracks = 100,
                    LocErr=0.02, # localization error in x, y and z (even if not used)
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
    
    nb_states = len(velocities)
    if not np.all(np.array([len(Fs), len(Ds), len(velocities), len(angular_Ds), len(conf_forces), len(conf_Ds), len(conf_dists)]) == nb_states):
        raise ValueError('Fs, Ds, velocities, angular_Ds, conf_forces, conf_Ds and conf_dists must all be 1D arrays of the same number of element (one element per state)')
    # diff + persistent motion + elastic confinement
    conf_sub_forces = conf_forces / nb_sub_steps
    sub_dt = dt / nb_sub_steps
   
    cum_Fs = np.zeros(nb_states)
    cum_Fs[0] = Fs[0]
    for state in range(1, nb_states):
        cum_Fs[state] = cum_Fs[state-1] + Fs[state]
    
    all_states = np.zeros(nb_tracks)
    
    for kkk in range(nb_tracks):
        state = np.argmin(np.random.rand()>cum_Fs)
        all_states[kkk] = state
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
    return final_tracks, all_states

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


'''
simulate a population of directed tracks and test effect of the dataset on the population model predicted parameters:
Fig 5 sup 1
'''

sav_dir = r'D:\anomalous/population_nb_tracks_directed/'
if not os.path.exists(sav_dir):
    os.makedirs(sav_dir)

all_est_LocErrs, all_est_ds, all_est_qs, all_est_ls, all_est_LPs = [[], [], [], [], []]
nb_tracks = [10, 30, 100, 300, 1000, 3000, 10000]

all_est_LocErrs=[]
all_est_ds = list(np.load(sav_dir + 'all_ds.npy'))
all_est_qs = list(np.load(sav_dir + 'all_qs.npy'))
all_est_ls = list(np.load(sav_dir + 'all_ls.npy'))
all_est_LPs = list(np.load(sav_dir + 'all_LPs.npy'))

len(all_est_LPs)

for i in range(50):
    print('rep:', i)
    
    tracks, true_states = anomalous_diff_mixture(track_len=50,
                                nb_tracks = 10000,
                                LocErr=0.02, # localization error in x, y and z (even if not used)
                                Fs = np.array([1]),
                                Ds = np.array([0.0]),
                                nb_dims = 2,
                                velocities = np.array([0.02]),
                                angular_Ds = np.array([0.1]),
                                conf_forces = np.array([0]),
                                conf_Ds = np.array([0.0]),
                                conf_dists = np.array([0]),
                                LocErr_std = 0,
                                dt = 0.02,
                                nb_sub_steps = 10)
    
    tracks = np.array(tracks)
    tf.keras.backend.clear_session()
    
    for n in nb_tracks:
        print('rep:', i, 'n:', n)
        
        est_LocErrs, est_ds, est_qs, est_ls, LP = Directed_fit_multi(tracks[:n], nb_states = 1, verbose = 0, Fixed_LocErr = True, nb_epochs = 300, Initial_params = {'LocErr': [0.02], 'd': [0.005], 'q': [0.01], 'l': [0.02]})
        
        est_LocErrs0, est_ds0, LP0 = Brownian_fit_multi(tracks[:n], nb_states = 1, verbose = 0, Fixed_LocErr = True, nb_epochs = 300, Initial_params = {'LocErr': [0.02], 'd': [0.005], 'q': [0.01], 'l': [0.02]})
        
        all_est_ds.append(est_ds)
        all_est_qs.append(est_qs)
        all_est_ls.append(est_ls)
        all_est_LPs.append(-np.sum(LP-LP0))
    
    np.save(sav_dir + 'all_ds', all_est_ds)
    np.save(sav_dir + 'all_qs', all_est_qs)
    np.save(sav_dir + 'all_ls', all_est_ls)
    np.save(sav_dir + 'all_LPs', all_est_LPs)

all_est_ds = np.load(sav_dir + 'all_ds.npy')
all_est_qs = np.load(sav_dir + 'all_qs.npy')
all_est_ls = np.load(sav_dir + 'all_ls.npy')
all_est_LPs = np.load(sav_dir + 'all_LPs.npy')

ds = np.reshape(all_est_ds, (50, 7)).T
qs = np.reshape(all_est_qs, (50, 7)).T
ls = np.reshape(all_est_ls, (50, 7)).T
LPs = np.reshape(all_est_LPs, (50, 7)).T


plt.figure(figsize = (3,3))
plt.plot(nb_tracks, np.mean((ds - 0)**2, 1)**0.5)
plt.plot(nb_tracks, np.std(ds, 1))
#plt.plot(nb_tracks, np.mean(ds**2, 1)**0.5-0)
plt.legend(['Error', 'Standard deviation'])
plt.xscale('log')
plt.xlabel('Number of tracks')
plt.ylabel('Diffusion speed metrics [$\mathrm{\mu m.\Delta t^{-1}}$]')
plt.savefig(sav_dir + 'diffusion_speed_error.svg')

plt.figure(figsize = (3,3))
plt.plot(nb_tracks, np.mean((ls*2**0.5 - 0.02)**2, 1)**0.5)
plt.plot(nb_tracks, np.std(ls*2**0.5, 1))
plt.plot(nb_tracks, np.abs(np.mean(ls*2**0.5, 1)-0.02))
plt.legend(['Error', 'Standard deviation', 'Bias'])
plt.xscale('log')
plt.xlabel('Number of tracks')
plt.ylabel('Velocity metrics [$\mathrm{\mu m.\Delta t^{-1}}$]')
plt.savefig(sav_dir + 'velocity_error.svg', format = 'svg')

ls*2**0.5

plt.figure(figsize = (3,3))
plt.plot(nb_tracks, np.mean(LPs, 1))
plt.scatter(nb_tracks, np.mean(LPs, 1))
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of tracks')
plt.ylabel('Log likelihood')
plt.savefig(sav_dir + 'LP.svg', format = 'svg')




'''
simulate a population of directed tracks and test effect of the dataset on the population model predicted parameters:
Fig 5 sup 1
'''


sav_dir = r'D:\anomalous/population_nb_tracks_confined/'
if not os.path.exists(sav_dir):
    os.makedirs(sav_dir)

all_est_LocErrs, all_est_ds, all_est_qs, all_est_ls, all_est_LPs = [[], [], [], [], []]

#all_est_LocErrs, all_est_ds, all_est_qs, all_est_ls = [list(np.load(sav_dir + 'all_LocErrs.npy')), list(np.load(sav_dir + 'all_ds.npy')), list(np.load(sav_dir + 'all_qs.npy')), list(np.load(sav_dir + 'all_ls.npy'))]
#len(all_est_LocErrs)

all_est_LocErrs=[]
all_est_ds = list(np.load(sav_dir + 'all_ds.npy'))
all_est_qs = list(np.load(sav_dir + 'all_qs.npy'))
all_est_ls = list(np.load(sav_dir + 'all_ls.npy'))
all_est_LPs = list(np.load(sav_dir + 'all_LPs.npy'))

(len(all_est_ds))/7
203/7

nb_tracks = [10, 30, 100, 300, 1000, 3000, 10000]

for i in range(50):
    tracks, true_states = anomalous_diff_mixture(track_len=50,
                                nb_tracks = 10000,
                                LocErr=0.02, # localization error in x, y and z (even if not used)
                                Fs = np.array([1]),
                                Ds = np.array([0.25]),
                                nb_dims = 2,
                                velocities = np.array([0.00]),
                                angular_Ds = np.array([0.0]),
                                conf_forces = np.array([0.2]),
                                conf_Ds = np.array([0.0]),
                                conf_dists = np.array([0]),
                                
                                LocErr_std = 0,
                                dt = 0.02,
                                nb_sub_steps = 20)
    
    tracks = np.array(tracks)
    
    tf.keras.backend.clear_session()
    
    for n in nb_tracks[:]:
        print('rep:', i, 'n:', n)
        est_LocErrs, est_ds, est_qs, est_ls, LP = Confined_fit_multi(tracks[:n], nb_states = 1, verbose = 0, Fixed_LocErr = True, nb_epochs = 300, Initial_params = {'LocErr': [0.02], 'd': [0.1], 'q': [0.01], 'l': [0.02]})
        est_LocErrs0, est_ds0, LP0 = Brownian_fit_multi(tracks[:n], nb_states = 1, verbose = 0, Fixed_LocErr = True, nb_epochs = 300, Initial_params = {'LocErr': [0.02], 'd': [0.1], 'q': [0.01], 'l': [0.02]})
        
        all_est_LocErrs.append(est_LocErrs)
        all_est_ds.append(est_ds)
        all_est_qs.append(est_qs)
        all_est_ls.append(est_ls)
        all_est_LPs.append(np.sum(LP-LP0))
    
    np.save(sav_dir + 'all_LocErrs', all_est_LocErrs)
    np.save(sav_dir + 'all_ds', all_est_ds)
    np.save(sav_dir + 'all_qs', all_est_qs)
    np.save(sav_dir + 'all_ls', all_est_ls)
    np.save(sav_dir + 'all_LPs', all_est_LPs)

all_est_ds = np.load(sav_dir + 'all_ds.npy')
all_est_qs = np.load(sav_dir + 'all_qs.npy')
all_est_ls = np.load(sav_dir + 'all_ls.npy')
all_est_LPs = np.load(sav_dir + 'all_LPs.npy')

ds = np.reshape(all_est_ds, (50, 7)).T
qs = np.reshape(all_est_qs, (50, 7)).T
ls = np.reshape(all_est_ls, (50, 7)).T
LPs = np.reshape(all_est_LPs, (50, 7)).T

plt.figure(figsize = (3,3))
plt.plot(nb_tracks, np.mean((ds - 0.1)**2, 1)**0.5)
plt.plot(nb_tracks, np.std(ds, 1))
plt.plot(nb_tracks, np.abs(np.mean(ds**2, 1)**0.5-0.1))
plt.legend(['Error', 'Standard deviation', 'Bias'])
plt.xscale('log')
plt.xlabel('Number of tracks')
plt.ylabel('Diffusion speed metrics [$\mathrm{\mu m.\Delta t^{-1}}$]')
plt.savefig(sav_dir + 'diffusion_speed_error.svg')

plt.figure(figsize = (3,3))
plt.plot(nb_tracks, np.mean((ls - 0.2)**2, 1)**0.5)
plt.plot(nb_tracks, np.std(ls, 1))
plt.plot(nb_tracks, np.abs(np.mean(ls, 1)-0.2))
plt.legend(['Error', 'Standard deviation', 'bias'])
plt.xscale('log')
plt.xlabel('Number of tracks')
plt.ylabel('Velocity metrics [$\mathrm{\mu m.\Delta t^{-1}}$]')
plt.savefig(sav_dir + 'velocity_error.svg', format = 'svg')

plt.figure(figsize = (3,3))
plt.plot(nb_tracks, -np.mean(LPs, 1))
plt.scatter(nb_tracks, -np.mean(LPs, 1))
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of tracks')
plt.ylabel('Log likelihood')
plt.savefig(sav_dir + 'LP.svg', format = 'svg')





'''
Determine the number of states in a population of tracks, fig 5b and sup 1
'''


sav_dir = r'D:\anomalous/population_get_nb_states/'
if not os.path.exists(sav_dir):
    os.makedirs(sav_dir)

all_likelihoods = []

nb_tracks = 50*4**np.arange(6)
for n in nb_tracks:
    print(n, 'rep:', i)
    
    tracks, true_states = anomalous_diff_mixture(track_len=300,
                                nb_tracks = n,
                                LocErr=0.02, # localization error in x, y and z (even if not used)
                                Fs = np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
                                Ds = np.array([0., 0.09, (0.05)**2/(2*0.02), 0.16, 0.16]),
                                nb_dims = 2,
                                velocities = np.array([0.01, 0.0, 0.008, 0.0, 0.]),
                                angular_Ds = np.array([0.02, 0.0, 0., 0., 0.]),
                                conf_forces = np.array([0., 0.05, 0., 0.2, 0.0]),
                                conf_Ds = np.array([0.0, 0., 0., 0.0225, 0.0]),
                                conf_dists = np.array([0., 0., 0., 0., 0.]),
                                LocErr_std = 0,
                                dt = 0.02,
                                nb_sub_steps = 10)
    
    tracks = np.array(tracks)
    tf.keras.backend.clear_session()
    
    likelihoods, all_pd_params = multi_fit(tracks, verbose = 1, Fixed_LocErr = True, min_nb_states = 1, 
                        max_nb_states = 10, nb_epochs = 400, batch_size = 1000,
                        Initial_confined_params = {'LocErr': 0.02, 'd': 0.1, 'q': 0.01, 'l': 0.1},
                        Initial_directed_params = {'LocErr': 0.02, 'd': 0.1, 'q': 0.01, 'l': 0.1},
                        fitting_type = 'All')
    
    all_likelihoods.append(likelihoods['Likelihood'].values)

plt.figure()
for likelihoods in all_likelihoods:
    plt.plot(np.arange(1,8), likelihoods - likelihoods[-1])
plt.legend(nb_tracks)

# Plot tracks
cs = [[0, 1, 1],  [0,0,1], [0,1,0], [1,0,1], [1,0.5,0]]
# dir, fixed conf, dir + dif, mov conf, diff

plt.figure(figsize = (15, 15))
lim = 2.5 # MreB
#lim = 0.5
nb_rows = 6
offset = 0
for i in range(nb_rows):
    for j in range(nb_rows):
        track = tracks[offset+i*nb_rows+j]
        cur_state_preds = true_states[offset+i*nb_rows+j].astype(int)
        track = track - np.mean(track,0 , keepdims = True) + [[lim*i, lim*j]]
        plt.plot(track[:,0], track[:,1], '--k', alpha = 0.5)
        plt.scatter(track[:,0], track[:,1] , c = cs[cur_state_preds], s = 7)
        plt.scatter(track[0,0], track[0,1] , c = 'k', s = 10, marker = 'x')
plt.gca().set_aspect('equal', adjustable='box')



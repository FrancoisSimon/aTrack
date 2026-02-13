# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 18:26:17 2026

@author: franc
"""


from fbm import FBM
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
#import andi_datasets
#from andi_datasets.datasets_challenge import challenge_theory_dataset

import sys
sys.path.insert(0, r'D:\Downloads\randi-main') # download randi at https://github.com/argunaykut/randi
from utils import data_norm, data_reshape, many_net_uhd, my_atan

sys.path.insert(0, r'C:\Users\franc\OneDrive\Bureau\Anomalous\scripts') # add the path to the atrack script (https://github.com/FrancoisSimon/aTrack)
import atrack

import numpy as np
from numba import njit, typed, prange, jit

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



'''
Andi model on fbm data
'''
n = 100
nb_tracks = 10000
nb_dims = 2

randi_dists = []

exps = [0.5,1,1.5]
coefs = [0.44, 0.14, 0.044] # coefficients used to get tracks of similar magnitudes

for c, exp in zip(coefs, exps):
    
    f = FBM(n=n-1, hurst=exp/2, length=1, method='daviesharte')
    
    fbm_samples = []
    for i in range(nb_tracks):
        fbm_samples.append(np.concatenate((f.fbm()[:,None], f.fbm()[:,None]), axis = 1).T*0.1/c)
    
    dataset = np.array(fbm_samples).reshape(nb_tracks, nb_dims * n)
    
    #dataset = np.concatenate((fbm_sample0[None], fbm_sample1[None], fbm_sample2[None]), axis = 0).reshape(nb_tracks, nb_dims * n)
    dataset = data_norm(dataset, dim=2, task=1)
    
    net = load_model(r'D:\Downloads\randi-main\nets\inference_nets\2d/inference_2D_125.h5')
    bs = net.layers[0].input_shape[-1]
    
    dataset_rs = data_reshape(dataset,bs=bs,dim=1)
    
    preds = net.predict(dataset_rs)
    randi_dists.append(preds)
    
    print('predicted exponents', preds.flatten())

# plot the distributions of the metric
plt.figure(figsize = (5,3))
for i in range(3):
    plt.hist(randi_dists[i], np.linspace(0, 2., 101), alpha = 0.5)
plt.xlabel('Estimated anomalous coefficient')
plt.ylabel('Counts')
plt.legend(['0.5', '1.0', '1.5'], title = 'Anomalous coefficient')

# print the best accuracy to distinguish confined motion from brownian motion
ks = []
for i in np.linspace(0.3, 1.7, 101):
    ks.append(np.mean(randi_dists[0]<i)/2 + np.mean(randi_dists[1]>i)/2)
print(np.max(ks))

# print the best accuracy to distinguish codirected motion from brownian motion
ks = []
for i in np.linspace(0.3, 1.7, 101):
    ks.append(np.mean(randi_dists[1]<i)/2 + np.mean(randi_dists[2]>i)/2)
print(np.max(ks))





'''
aTrack model on fbm data
'''
verbose = 1
Fixed_LocErr = False
Initial_confined_params = {'LocErr': 0.002, 'd': 0.05, 'q': 0.01, 'l': 0.01}
Initial_directed_params = {'LocErr': 0.002, 'd': 0.05, 'q': 0.01, 'l': 0.01}

dists = []


for c, exp in zip(coefs, exps):
    
    f = FBM(n=n-1, hurst=exp/2, length=1, method='daviesharte')
    
    
    fbm_samples = []
    for i in range(nb_tracks):
        fbm_samples.append(np.concatenate((f.fbm()[:,None], f.fbm()[:,None]), axis = 1)*0.1/c)
    
    dataset = np.array(fbm_samples)
    
    est_LocErrs0, est_ds0, est_qs0, est_ls0, LP0 = atrack.Confined_fit(dataset, verbose = verbose, Fixed_LocErr = Fixed_LocErr, Initial_params = Initial_confined_params, nb_epochs = 400, output_type = 'values')
    est_LocErrs2, est_ds2, est_qs2, est_ls2, LP2, pred_kis = atrack.Directed_fit(dataset, verbose = verbose, Fixed_LocErr = Fixed_LocErr, Initial_params = Initial_directed_params, nb_epochs = 400, output_type = 'values')
    
    LPs = LP2 - LP0
    
    dists.append(LPs)
    
    print('predicted exponents', preds.flatten())
    
# plot the distributions of the metric
plt.figure(figsize = (5,3))
for i in range(3):
    plt.hist(dists[i], np.linspace(-10, 40, 101), alpha = 0.5)
plt.xlabel('Estimated anomalous coefficient')
plt.ylabel('Counts')
plt.legend(['0.5', '1.0', '1.5'], title = 'Anomalous coefficient')

# print the best accuracy to distinguish confined motion from brownian motion
ks = []
for i in np.linspace(-30, 30, 101):
    ks.append(np.mean(dists[0]<i)/2 + np.mean(dists[1]>i)/2)
print(np.max(ks))

# print the best accuracy to distinguish directed motion from brownian motion
ks = []
for i in np.linspace(-30, 30, 101):
    ks.append(np.mean(dists[1]<i)/2 + np.mean(dists[2]>i)/2)
print(np.max(ks))

'''
Plot fbm tracks
'''
exps = [0.5,1,1.5]
coefs = [0.44, 0.14, 0.044]

sav_dir = 'D:/anomalous/Randi/'
names = ['fbm_0.5', 'fbm_1.0', 'fbm_1.5']
lims = [1,1,3]

for k, c, exp in zip(np.arange(3), coefs, exps):
    
    f = FBM(n=n-1, hurst=exp/2, length=1, method='daviesharte')
    
    fbm_samples = []
    for i in range(10):
        fbm_samples.append(np.concatenate((f.fbm()[:,None], f.fbm()[:,None]), axis = 1)*0.1/c)
    
    tracks = np.array(fbm_samples)
    
    lim = lims[k]
    plt.figure(figsize = (10,10))
    for i in range(3):
        for j in range(3):
            #plt.subplot(3,3,i*3+j+1)
            track = tracks[i*3+j]
            plt.plot(track[:,0] - np.mean(track[:,0]) + lim*i, track[:,1] - np.mean(track[:,1]) +lim*j, color = 'k')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(sav_dir + names[k] + '.svg', format = 'svg')



'''
Randi vs aTrack on physical simulation data
'''

n = 100

tracks, all_states = anomalous_diff_mixture(track_len=n,
                            nb_tracks = 10000,
                            LocErr=0.02, # localization error in x, y and z (even if not used)
                            Fs = np.array([1]),
                            Ds = np.array([0.25]),
                            nb_dims = 2,
                            velocities = np.array([0.0]),
                            angular_Ds = np.array([0.0]),
                            conf_forces = np.array([0.2]),
                            conf_Ds = np.array([0.0]),
                            conf_dists = np.array([0.0]),
                            LocErr_std = 0,
                            dt = 0.02,
                            nb_sub_steps = 10)

track_lists = [np.array(tracks)]

tracks, all_states = anomalous_diff_mixture(track_len=n,
                           nb_tracks = 10000,
                            LocErr=0.02, # localization error in x, y and z (even if not used)
                            Fs = np.array([1]),
                            Ds = np.array([0.25]),
                            nb_dims = 2,
                            velocities = np.array([0.0]),
                            angular_Ds = np.array([0.0]),
                            conf_forces = np.array([0.0]),
                            conf_Ds = np.array([0.0]),
                            conf_dists = np.array([0.0]),
                            LocErr_std = 0,
                            dt = 0.02,
                            nb_sub_steps = 10)


track_lists.append(np.array(tracks))

tracks, all_states = anomalous_diff_mixture(track_len=n,
                           nb_tracks = 10000,
                            LocErr=0.02, # localization error in x, y and z (even if not used)
                            Fs = np.array([1]),
                            Ds = np.array([0.25]),
                            nb_dims = 2,
                            velocities = np.array([0.1]),
                            angular_Ds = np.array([0.1]),
                            conf_forces = np.array([0.0]),
                            conf_Ds = np.array([0.0]),
                            conf_dists = np.array([0.0]),
                            LocErr_std = 0,
                            dt = 0.02,
                            nb_sub_steps = 10)

track_lists.append(np.array(tracks))

tracks, all_states = anomalous_diff_mixture(track_len=n,
                           nb_tracks = 10000,
                            LocErr=0.02, # localization error in x, y and z (even if not used)
                            Fs = np.array([1]),
                            Ds = np.array([0.0]),
                            nb_dims = 2,
                            velocities = np.array([0.1]),
                            angular_Ds = np.array([0.1]),
                            conf_forces = np.array([0.0]),
                            conf_Ds = np.array([0.0]),
                            conf_dists = np.array([0.0]),
                            LocErr_std = 0,
                            dt = 0.02,
                            nb_sub_steps = 10)

tracks = np.array(tracks)
track_lists.append(np.array(tracks))

sav_dir = 'D:/anomalous/Randi/'
names = ['Conf', 'Diff', 'Lin+Diff', 'Lin']
lims = [2,2,10,10]


'''
plot physical tracks
'''
for k, tracks in enumerate(track_lists):
    lim = lims[k]
    plt.figure(figsize = (10,10))
    for i in range(3):
        for j in range(3):
            #plt.subplot(3,3,i*3+j+1)
            track = tracks[i*3+j]
            plt.plot(track[:,0] - np.mean(track[:,0]) + lim*i, track[:,1] - np.mean(track[:,1]) +lim*j, color = 'k')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(sav_dir + names[k] + '.svg', format = 'svg')


nb_tracks = 10000
nb_dims = 2

randi_dists = []

for tracks in track_lists:
    dataset = tracks
    #.transpose([0,2,1])
    #dataset = dataset.reshape(nb_tracks, nb_dims * n)
    
    #dataset = np.concatenate((fbm_sample0[None], fbm_sample1[None], fbm_sample2[None]), axis = 0).reshape(nb_tracks, nb_dims * n)
    dataset = data_norm(dataset, dim=2, task=1)
    
    net = load_model(r'D:\Downloads\randi-main\nets\inference_nets\2d/inference_2D_125.h5')
    bs = net.layers[0].input_shape[-1]
    
    dataset_rs = data_reshape(dataset,bs=bs,dim=1)
    
    preds = net.predict(dataset_rs)
    randi_dists.append(preds)
    
    print('predicted exponents', preds.flatten())

# plot the distributions of the metric
plt.figure(figsize = (5,3))
for i in range(4):
    plt.hist(randi_dists[i], np.linspace(0.2, 0.4, 101), alpha = 0.5)
plt.xlabel('Estimated anomalous coefficient')
plt.ylabel('Counts')
plt.legend(['Confined', 'Diffusive', 'Directed + Diffusive', 'Directed'])

# print the best accuracy to distinguish confined motion from brownian motion
ks = []
for i in np.linspace(0.05, 1.95, 1901):
    ks.append(np.mean(randi_dists[0]<i)/2 + np.mean(randi_dists[1]>i)/2)
print(np.max(ks))

# print the best accuracy to distinguish directed + diffusive motion from brownian motion
ks = []
for i in np.linspace(0.05, 1.95, 1901):
    ks.append(np.mean(randi_dists[1]<i)/2 + np.mean(randi_dists[2]>i)/2)
print(np.max(ks))

# print the best accuracy to distinguish directed motion from brownian motion
ks = []
for i in np.linspace(0.05, 1.95, 1901):
    ks.append(np.mean(randi_dists[1]<i)/2 + np.mean(randi_dists[3]>i)/2)
print(np.max(ks))



verbose = 1
Fixed_LocErr = False
Initial_confined_params = {'LocErr': 0.002, 'd': 0.05, 'q': 0.01, 'l': 0.01}
Initial_directed_params = {'LocErr': 0.002, 'd': 0.05, 'q': 0.01, 'l': 0.01}

dists = []

for tracks in track_lists:
    
    #dataset = np.concatenate((fbm_sample0[None], fbm_sample1[None], fbm_sample2[None]), axis = 0).reshape(nb_tracks, nb_dims * n)
    
    est_LocErrs0, est_ds0, est_qs0, est_ls0, LP0 = atrack.Confined_fit(tracks, verbose = verbose, Fixed_LocErr = Fixed_LocErr, Initial_params = Initial_confined_params, nb_epochs = 400, output_type = 'values')
    #est_LocErrs1, est_ds1, LP1 = Brownian_fit(dataset, nb_dims, verbose = verbose, Fixed_LocErr = Fixed_LocErr, Initial_params = Initial_confined_params, nb_epochs = 400)
    est_LocErrs2, est_ds2, est_qs2, est_ls2, LP2, pred_kis = atrack.Directed_fit(tracks, verbose = verbose, Fixed_LocErr = Fixed_LocErr, Initial_params = Initial_directed_params, nb_epochs = 400, output_type = 'values')
    
    #argmaxs = np.argmax(np.concatenate((LP0[None, :, 0], LP1[None, :, 0]+3, LP2[None, :, 0])), 0)
    #np.mean(argmaxs==0)
    #pred_kis.shape
    
    #LPs = LP2 - LP1
    #LPs[LP0>LP1] = LP1[LP0>LP1] - LP0[LP0>LP1]
    LPs = LP2 - LP0
    #np.mean(np.sum(pred_kis**2, 2), 1)**0.5
    
    dists.append(LPs)
    
    print('predicted exponents', preds.flatten())
    
# plot the distributions of the metric
plt.figure(figsize = (8,3))
for i in range(4):
    plt.hist(dists[i], np.linspace(-10, 250, 261), alpha = 0.5)
plt.xlabel('Log likelihood')
plt.ylabel('Counts')

# print the best accuracy to distinguish confined motion from brownian motion
ks = []
for i in np.linspace(-30, 30, 101):
    ks.append(np.mean(dists[0]<i)/2 + np.mean(dists[1]>i)/2)
print(np.max(ks))

# print the best accuracy to distinguish directed + diffusive motion from brownian motion
ks = []
for i in np.linspace(-30, 30, 101):
    ks.append(np.mean(dists[1]<i)/2 + np.mean(dists[2]>i)/2)
print(np.max(ks))

# print the best accuracy to distinguish directed motion from brownian motion
ks = []
for i in np.linspace(-30, 30, 101):
    ks.append(np.mean(dists[1]<i)/2 + np.mean(dists[3]>i)/2)
print(np.max(ks))



'''
Test of atrack on tracks confined by fixed boundaries instead of a potential well
'''
from numba import njit, typed, prange, jit

@njit
def anomalous_diff_mixture_hard_conf(track_len=640,
                           nb_tracks = 100,
                    LocErr=0.02, # localization error in x, y and z (even if not used)
                    Fs = np.array([1.]),
                    Ds = np.array([0.25]),
                    nb_dims = 2,
                    velocities = np.array([0.0]),
                    angular_Ds = np.array([0.0]),
                    conf_forces = np.array([0.0]),
                    conf_Ds = np.array([0.0]),
                    conf_dists = np.array([0.0]),
                    conf_radii = np.array([0.1]),
                    LocErr_std = 0,
                    dt = 0.02,
                    nb_sub_steps = 40):
    
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
        D, velocity, angular_D, conf_sub_force, conf_D, conf_dist, conf_radius = (Ds[state], velocities[state], angular_Ds[state], conf_sub_forces[state], conf_Ds[state], conf_dists[state], conf_radii[state])
       
        positions = np.zeros((track_len * nb_sub_steps, nb_dims))
        
        init_x = np.inf
        init_y = np.inf
        while (init_x**2 + init_y**2)**0.5 > conf_radius:
            init_x = 2*np.random.rand()*conf_radius - conf_radius
            init_y = 2*np.random.rand()*conf_radius - conf_radius
            
        positions[0] = [init_x, init_y]
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
            angle = angles[i]
            pesistent_disp = np.array([np.cos(angle), np.sin(angle)]).T * velocity/nb_sub_steps
            positions[i+1] = positions[i] + pesistent_disp + disps[i]
            positions[i+1] = (1-conf_sub_force) *  positions[i+1] + conf_sub_force * anchor_positions[i]
            center_dist = (positions[i+1, 0]**2 + positions[i+1, 1]**2)**0.5
            if  center_dist >  conf_radius:
                positions[i+1] = positions[i+1] * conf_radius / center_dist
            
        final_track = np.zeros((track_len, nb_dims))
        for i in range(track_len):
            final_track[i] = positions[i*nb_sub_steps]
       
        final_track += np.random.normal(0, LocErr, (track_len, nb_dims))
       
        if kkk ==0:
            final_tracks = typed.List([final_track])
        else:
            final_tracks.append(final_track)
    return final_tracks, all_states

sav_dir = 'D:/anomalous/strict_confinement/'

verbose = 1
Fixed_LocErr = False
Initial_confined_params = {'LocErr': 0.02, 'd': 0.15, 'q': 0.01, 'l': 0.01}
Initial_directed_params = {'LocErr': 0.02, 'd': 0.15, 'q': 0.01, 'l': 0.01}

all_LPs = []
all_ds = []
all_ls = []

radii = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1, 1.5, 2, 3, 5]

for r in radii:
        
    tracks, all_states = anomalous_diff_mixture_hard_conf(track_len=200,
                               nb_tracks = 10000,
                                LocErr=0.02, # localization error in x, y and z (even if not used)
                                Fs = np.array([1.]),
                                Ds = np.array([0.25]),
                                nb_dims = 2,
                                velocities = np.array([0.0]),
                                angular_Ds = np.array([0.0]),
                                conf_forces = np.array([0.0]),
                                conf_Ds = np.array([0.0]),
                                conf_dists = np.array([0.0]),
                                conf_radii = np.array([r]),
                                LocErr_std = 0,
                                dt = 0.02,
                                nb_sub_steps = 40)
    
    tracks = np.array(tracks)
    
    #dataset = np.concatenate((fbm_sample0[None], fbm_sample1[None], fbm_sample2[None]), axis = 0).reshape(nb_tracks, nb_dims * n)
    
    est_LocErrs0, est_ds0, est_qs0, est_ls0, LP0 = atrack.Confined_fit(tracks, verbose = verbose, Fixed_LocErr = Fixed_LocErr, Initial_params = Initial_confined_params, nb_epochs = 1000, output_type = 'values')
    #est_LocErrs1, est_ds1, LP1 = Brownian_fit(dataset,
    
    est_LocErrs2, est_ds2, est_qs2, est_ls2, LP2, pred_kis = atrack.Directed_fit(tracks, verbose = verbose, Fixed_LocErr = Fixed_LocErr, Initial_params = Initial_directed_params, nb_epochs = 1000, output_type = 'values')
    
    all_ds.append(est_ds0)
    all_ls.append(est_ls0)
    all_LPs.append(LP0-LP2)
    print(r, np.mean(LP0-LP2))
    
    np.save(sav_dir + 'all_ds', all_ds)
    np.save(sav_dir + 'all_ls', all_ls)
    np.save(sav_dir + 'all_LPs', all_LPs)

all_ds = np.load(sav_dir + 'all_ds.npy')
all_ls = np.load(sav_dir + 'all_ls.npy')
all_LPs = np.load(sav_dir + 'all_LPs.npy')

fracts = np.mean(all_LPs > np.log(20), (1,2))

plt.figure(figsize = (3,3))
plt.plot(radii, fracts)
plt.ylabel('Fraction of significantly confined tracks')
plt.xlabel('Confinement radius [$\mathrm{\mu m}$]')
plt.grid()
plt.xscale('log')


np.mean(LP0)

all_ds = np.array(all_ds)
all_ls = np.array(all_ls)
all_LPs = np.array(all_LPs)

mean_LPs = np.mean(all_LPs, (1,2))

plt.figure(figsize = (3,3))
plt.plot(radii, mean_LPs)
plt.ylabel('Log likelihood difference')
plt.xlabel('Confinement radius [$\mathrm{\mu m}$]')
plt.grid()
plt.xscale('log')

mean_ds = np.mean(all_ds, (1,2))

plt.figure(figsize = (3,3))
plt.plot(radii, mean_ds)
plt.ylabel('Diffusion speed [$\mathrm{\mu m.\Delta t^{-1}}$]')
plt.xlabel('Confinement radius [$\mathrm{\mu m}$]')
plt.grid()
plt.xscale('log')

mean_all_ls = np.mean(all_ls, (1,2))

plt.figure(figsize = (3,3))
plt.plot(radii, mean_all_ls)
plt.ylabel('Non-corrected confinement factor')
plt.xlabel('Confinement radius [$\mathrm{\mu m}$]')
plt.grid()
plt.xscale('log')

corr_all_ls = -np.log(1-all_ls)
mean_corr_all_ls = np.mean(corr_all_ls, (1,2))

plt.figure(figsize = (3,3))
plt.plot(radii, mean_corr_all_ls)
plt.ylabel('Corrected confinement factor')
plt.xlabel('Confinement radius [$\mathrm{\mu m}$]')
plt.grid()
plt.xscale('log')

cor_all_ds = (1-np.exp(-corr_all_ls))/corr_all_ls * all_ds
all_conf_rads = (cor_all_ds*0.02/corr_all_ls)**0.5
mean_conf_rads = np.mean(all_conf_rads, (1,2))
std_conf_rads = np.std(all_conf_rads, (1,2))

plt.figure(figsize = (3,3))
plt.plot(radii, mean_conf_rads*3)
plt.plot(radii, radii)
plt.fill_between(radii, np.clip(mean_conf_rads*3 - 3*std_conf_rads, 0, 10), mean_conf_rads*3 + 3*std_conf_rads, color = 'b', alpha = 0.2)
plt.legend(['Estimated radius', 'True radius'])
plt.ylabel('Estimated confinement radius [$\mathrm{\mu m}$]')
plt.xlabel('True confinement radius [$\mathrm{\mu m}$]')
plt.grid()
plt.xscale('log')
plt.yscale('log')

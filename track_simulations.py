
import numpy as np
from numba import njit, typed, prange, jit

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
                           nb_sub_steps = 10,
                           nb_blurring_sub_steps = 1):
    
    #nb_blurring_sub_steps = min(max(int(np.ceil(0*nb_sub_steps)), 1), nb_sub_steps)
    
    # diff + persistent motion + elastic confinement
    conf_sub_forces = conf_forces / nb_sub_steps
    sub_dt = dt / nb_sub_steps
    sub_step_wise_loc_err = LocErr*nb_blurring_sub_steps**0.5

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
        
        positions += np.random.normal(0, sub_step_wise_loc_err, (track_len*nb_sub_steps, nb_dims))
        final_track = np.zeros((track_len, nb_dims))
        for i in range(track_len):
            #final_track[i] = np.mean(positions[i*nb_sub_steps:i*nb_sub_steps+nb_blurring_sub_steps], axis=0)
            #final_track[i] = positions[i*nb_sub_steps]
            for j in range(len(final_track[i])):
                final_track[i, j] = np.mean(positions[i*nb_sub_steps:i*nb_sub_steps+nb_blurring_sub_steps, j])
        
        #if nb_blurring_sub_steps==1:
        #    final_track += np.random.normal(0, LocErr, (track_len, nb_dims))
       
        if kkk == 0:
            final_tracks = typed.List([final_track])
            states = typed.List([state])
        else:
            final_tracks.append(final_track)
            states.append(state)
    return final_tracks, states



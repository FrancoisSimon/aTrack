aTrack
-------

aTrack is a method to detemine the state of motion of particles. The stand-alone version of aTrack is available at [https://zenodo.org/records/10994586](https://zenodo.org/records/11075336).
For more information about the Stand-alone version of aTrack, see the [Wiki](https://github.com/FrancoisSimon/aTrack/wiki) section of this GitHub page.

This readme file focuses on the python implementation of aTrack.

# Dependencies

- numpy
- tensorflow
- sklearn
- pandas
  
# Installation (from pip)

(needs to be run in anaconda prompt for anaconda users on windows)

## Install dependencies

pip install scikit-learn "tensorflow<2.11" pandas

## Install aTrack

`pip install aTrack` (not working yet)

Alternatively, you can left-click on `code`, then `download zip`. Unzip the downloaded folder. In python, set the working directory to the location of the unzipped folder. The package is named atrack.

`os.chdir(r'C:\Users\username\path\aTrack-main')
import atrack
`

## Input file format

aTrack needs csv files with rows that represent the peaks of the tracks and columns that must contain the following headers: ['POSITION_X', 'POSITION_Y', 'FRAME', 'TRACK_ID']. The columns 'POSITION_X' and 'POSITION_Y' must contain floats that represent the positions in the different dimensions of your track. 'FRAME' must contain floats or integers ordered with regard to the time points of the tracks. 'TRACK_ID' must contain integers that specify to which track the position belongs to. N/A values are not allowed.

# Installation from this GitHub repository

## From Unix/Mac:

## From Windows using anaconda prompt:

`pip install aTrack` (not working yet)

# Tutorial

For an extensive tutorial, see the jupyter notebook at https://github.com/FrancoisSimon/aTrack/blob/main/aTrack_Tutorial.ipynb .

# Usage
## Main functions

### Brownian_fit

Fit single tracks to a diffusion model with localization error. If memory issues occur, split your data set into multiple arrays and perform a fitting on each array separately.

Arguments:
- `tracks` : 3-dimension array of tracks of same length. dims: track ID, time point, space coordinates (x, y, ..).
- `verbose` : tensorflow model fit verbose. The default is 0.
- `Fixed_LocErr` : Fix the localization error to its initial value if True. The default is True.
- `nb_epochs` : Number of epochs for the model fitting. The default is 400.
- `Initial_params` : Dictionary of the initial parameters. The values for each key must be a list of floats of length `nb_states`.
        The default is {'LocErr': [0.02, 0.022, 0.022], 'd': [0.1, 0.12, 0.12], 'q': [0.01, 0.012, 0.012], 'l': [0.01, 0.02, 0.012]}.
        The parameters represent the localization error, the diffusion length per step, the change per step of the x, y speed of the
        directed motion and the standard deviation of the initial speed of the particle. 

Outputs:
- `pd_params`: Log likelihood and model parameters in pandas format
        Log_likelihood: log probability of each track according to the model.
        LocErr: Estiamted localization errors for each state 
        d: Estimated d, the diffusion length per step of the particle, for each state (the diffusion coefficient $D = d^2/(2dt)$
        est_qs: Estimated diffusion lengths per step of the potential well for each state.
        est_ls: Estiamted confinement factor of each particle.

### anomalous.Directed_fit

Fit single tracks to a model with diffusion plus directed motion while still considering localization error. If memory issues occur, split your data set into multiple arrays and perform a fitting on each array separately.

Arguments:
- `tracks` : 3-dimension array of tracks of same length. dims: track ID, time point, space coordinates (x, y, ..).
- `verbose` : tensorflow model fit verbose. The default is 0.
- `Fixed_LocErr` : Fix the localization error to its initial value if True. The default is True.
- `nb_epochs` : Number of epochs for the model fitting. The default is 400.
- `Initial_params` : Dictionary of the initial parameters. The values for each key must be a list of floats of length `nb_states`.
        The default is {'LocErr': [0.02, 0.022, 0.022], 'd': [0.1, 0.12, 0.12], 'q': [0.01, 0.012, 0.012], 'l': [0.01, 0.02, 0.012]}.
        The parameters represent the localization error, the diffusion length per step, the change per step of the x, y speed of the
        directed motion and the standard deviation of the initial speed of the particle. 

Outputs:
- `pd_params`: Log likelihood and model parameters in pandas format.
        Log_likelihood: log probability of each track according to the model.
        LocErr: Estiamted localization errors for each state .
        d: Estimated d, the diffusion length per step of the particle, for each state (the diffusion coefficient $D = d^2/(2dt)$.
        q: Estimated diffusion lengths per step of the potential well for each state.
        l: Estiamted standard deviation of the initial speed of the particle.
        mean_speed: Predicted average speed of the particle along the whole track (as opposed to l which represents the speed at the first time point).
    
### anomalous.Confined_fit

Fit single tracks to a model with diffusion plus confinement while still considering localization error. If memory issues occur, split your data set into multiple arrays and perform a fitting on each array separately.

Arguments:
- `tracks`: 3-dimension array of tracks of same length. dims: track ID, time point, space coordinates (x, y, ..).
- `verbose`: tensorflow model fit verbose. The default is 0.
- `Fixed_LocErr`: Fix the localization error to its initial value if True. The default is True.
- `nb_epochs`: Number of epochs for the model fitting. The default is 400.
- `Initial_params`: Dictionary of the initial parameters. The values for each key must be a list of floats of length `nb_states`.
        The default is {'LocErr': [0.02, 0.022, 0.022], 'd': [0.1, 0.12, 0.12], 'q': [0.01, 0.012, 0.012], 'l': [0.01, 0.02, 0.012]}.
        The parameters represent the localization error, the diffusion length per step, the change per step of the x, y speed of the
        directed motion and the standard deviation of the initial speed of the particle. 

Outputs:
- `pd_params`: Log likelihood and model parameters in pandas format.
        Log_likelihood: log probability of each track according to the model.
        LocErr: Estiamted localization errors for each state .
        d: Estimated d, the diffusion length per step of the particle, for each state (the diffusion coefficient $D = d^2/(2dt)$.
        est_qs: Estimated diffusion lengths per step of the potential well for each state.
        est_ls: Estiamted confinement factor of each particle.

### anomalous.multi_fit

Fit models with multiple states and vary the number of states to determine which number of states is best suited to the data set and to retrieve the multi-state model parameters. More precisely, in a first fitting step, we estimate the parameters of individual tracks. We then cluster tracks with close parameters using a Gaussian mixture model to form `max_nb_states` states whose parameters are the average of the parameters of their tracks. Then, multi-state fitting is performed on the full data set. the log likelihood is computed and stored and the state with the lowest impact on the likelihood is removed. The number of states is further reduced until the number of states of the model reaches the value `min_nb_states`.

Arguments:
- `tracks`: Numpy array of tracks of dims (track, time point, spatial axis).
- `verbose`: if 1, the function prints the model summary and the fitting infos. The default is 1.
- `Fixed_LocErr`: If True fixes the the localization error based on a prior estimate, this can be important to do if there is no immobile state. The default is True.
- `max_nb_states`: Number of states used for the clustering. The number of states is iteratively reduced until reaching `min_nb_states`.
- `min_nb_states`: Initial number of states used for the clustering.
- `nb_epochs`: number of epochs for the model fitting. The default is 1000.
- `batch_size`: Number of tracks considered per batch to avoid memory issues when dealing with big data sets.
- `Initial_confined_params`: Initial guess for the first step of the method. The default is {'LocErr': 0.02, 'd': 0.1, 'q': 0.01, 'l': 0.01}.
- `Initial_directed_params`: The default is {'LocErr': 0.02, 'd': 0.1, 'q': 0.01, 'l': 0.01}.

Outputs:
- `likelihoods`: pandas array of the likelihoods of the model for the different nunbers of states.
- `all_pd_params`: Dictionnary with keys 'nb_states' containing pandas arrays summarizing all the parameters of the model for the number of states specified by the key. The columns are 'state', 'LocErr', 'd', 'q', 'l', 'type', 'fraction'. `state`: integer representing each state of the model with a number of states = 'nb_states'. `LocErr`: localization error, `d`: diffusion length per step which relates to the diffusion coefficient $D$ and time step $\Delta t$ according to $D = \sqrt{2D\Delta t}$. `q`: evolution parameter which corresponds to the diffusion length of the potential well when the state is confined and which corresponds to the change per time step of the velocity when the state is directed (see `type` to know if the state is confined or directed). `l`: force of the anomalous diffusion parameter, in case of confinement it corresponds to the fraction of the distance between the particle and the center of the potential well crossed during 1 time step, in case of directed motion, it corresponds to the velocity of the directed motion (at the first time step of the tracks).  `type`: Either 'Conf' if the state represents confined motion or 'Dir' if the state represents directed motion. `fraction`: Fraction of the particles in this state.

## Extra functions

### Function `anomalous.read_table`

The function anomalous.read_table can be used to upload files in csv format (also work for other table formats).

Arguments:
- `path`: path to the data in the trackmate xml format.
- `lengths`: track lengths considered (default value = np.arange(100,101)).
- `dist_th`: maximal distance between consecutive time points (default value = p.inf).
- `frames_boundaries `: list of first and last frames to consider (default value = [-np.inf, np.inf]).
- `fmt`: format of the document to be red, can be 'csv' or 'pkl'. One can also simply specify a separator in case of another table format, e.g. ';' if colums are separated by ';'.
- `colnames`: list of the header names used in the table file to load corresponding to the coordianates, the frame and track ID for each peak. The first elements must be the coordinates headers (1 to 3 for 1D to 3D), the penultimate must be the frame header and the last element must be the track ID of the peak (default value = ['POSITION_X', 'POSITION_Y', 'FRAME', 'TRACK_ID']).
- `opt_colnames`: List of additional metrics to collect from the file, e.g. ['QUALITY', 'ID'],(default value = []).
- `remove_no_disp`: If True, removes tracks that show absolutly no displacements as most likely arising from wrong peak detection.

Outputs:
- `all_tracks`: dictionary describing the tracks with track length as keys (number of time positions, e.g. '23') of 3D arrays: dim 0 = track, dim 1 = time position, dim 2 = x, y position.
- `all_frames`: dictionary descibing the frame numbers of each peak of all tracks with track length as keys (number of time positions, e.g. '23') of 2D arrays: dim 0 = track, dim 1 = time position.
- `optional_metrics`: dictionary describing the optional metrics specified by opt_colnames for each peak of all tracks with the same format as the other outputs, track length as keys (number of time positions, e.g. '23') of 3D arrays: dim 0 = track, dim 1 = time position, dim 2 = optional metrics (same length as the length of the list opt_colnames).


## Caveats

# License
This program is released under the GNU General Public License version 3 or upper (GPLv3+).

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Parallelization
GPU parallelization is made available thanks to the tensorflow library. When analyzing small data sets, using CPU computing may be faster. To do so, one can use the following command `with tf.device('/CPU:0'):`.
`

# Deploying (developer only)

One can create an exectuable softare from the python code using:
`pyinstaller --onedir --hidden-import=scipy.special._cdflib path\GUI_2windows.py`
If anyone can make the Graphical interface work for linux or Mac Os. I will be happy to share this version as well on [Zenodo](https://zenodo.org/records/10994586).

# Authors
Francois Simon


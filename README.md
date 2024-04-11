aTrack
-------

This repository contains the necessary scripts to run the method aTrack, a method to detemine the state of motion of particles.

See the Wiki section for more information on how to install and use aTrack.

https://pypi.org/project/extrack/

# Dependencies

- numpy
- tensorflow
- sklearn
- pandas
  
# Installation (from pip)

(needs to be run in anaconda prompt for anaconda users on windows)

## Install dependencies

pip install scikit-learn tensorflow pandas

## Install aTrack

to do: `pip install aTrack`

## Input file format

aTrack needs csv files with row representing the peaks of the tracks the following columns: ['POSITION_X', 'POSITION_Y', 'FRAME', 'TRACK_ID']
'POSITION_X' and 'POSITION_Y' msut contain floats,
'FRAME' must contain floats or integers ordered with regard to the time,
'TRACK_ID' must contain integers that specify to which track the position belongs to
N/A values are not allowed

# Installation from this GitHub repository

## From Unix/Mac:


## From Windows using anaconda prompt:


# Creating an executable file from the python code

`pyinstaller --onefile --hidden-import=scipy.special._cdflib GUI_2windows.py`

# Tutorial


**Document here how to open a Jupyter notebook**

# Usage
## Main functions

## Extra functions



## Caveats

# References

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

# Deploying (developer only)

# Authors
Francois Simon


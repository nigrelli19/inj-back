import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import seaborn as sns
import json
import h5py
import scipy

GEMIT_X = 0.71e-9
GEMIT_Y = 2.1e-12

def load_h5_to_dict(file_path):
    data_dict = {}
    
    with h5py.File(file_path, 'r') as h5f:
        for key in h5f.keys():
            # Load each dataset and store in the dictionary
            data_dict[key] = h5f[key][:]
    
    return data_dict

def normalized_coordiantes(twiss, track_particle, monitor_name = '0'):
    '''
    Function to manualy calculate the normalized coordiantes
    '''
    if monitor_name == '0':
        monitor_name = 'ip.1_aper'

    if isinstance(twiss, pd.DataFrame):
        betx = twiss.loc[twiss['name'] == monitor_name, 'betx'].values[0]
        bety = twiss.loc[twiss['name'] == monitor_name, 'bety'].values[0]
        alfx = twiss.loc[twiss['name'] == monitor_name, 'alfx'].values[0]
        alfy = twiss.loc[twiss['name'] == monitor_name, 'alfy'].values[0]
    else:
        betx = twiss['betx', monitor_name]
        bety = twiss['bety', monitor_name]
        alfx = twiss['alfx', monitor_name]
        alfy = twiss['alfy', monitor_name]

    if isinstance(track_particle, dict):

        X_norm = np.array(track_particle['x']) / np.sqrt(betx* GEMIT_X)
        Y_norm = np.array(track_particle['y']) / np.sqrt(bety* GEMIT_Y)

        Px_norm = (alfx * np.array(track_particle['x']) + betx*np.array(track_particle['px'])) / np.sqrt(betx * GEMIT_X)
        Py_norm = (alfy * np.array(track_particle['y']) + bety*np.array(track_particle['py'])) / np.sqrt(bety * GEMIT_Y) 
    
    elif isinstance(track_particle, np.lib.npyio.NpzFile):
        
        X_norm = (track_particle['x']) / np.sqrt(betx * GEMIT_X)
        Y_norm = (track_particle['y']) / np.sqrt(bety * GEMIT_Y)

        Px_norm = (alfx * track_particle['x']+ betx*track_particle['px']) / np.sqrt(betx * GEMIT_X)
        Py_norm = (alfy * track_particle['y'] + bety*track_particle['py']) / np.sqrt(bety * GEMIT_Y)

    else:

        X_norm = (track_particle.x) / np.sqrt(betx * GEMIT_X)
        Y_norm = (track_particle.y) / np.sqrt(bety * GEMIT_Y)

        Px_norm = (alfx * track_particle.x + betx*track_particle.px) / np.sqrt(betx * GEMIT_X)
        Py_norm = (alfy * track_particle.y + bety*track_particle.py) / np.sqrt(bety * GEMIT_Y)    

    #print(np.sqrt(betx * GEMIT_X))
    return X_norm, Y_norm, Px_norm, Py_norm

output_dir = 'test_circ_with_coll'  #'test_DA_1'
#df_part = pd.read_hdf(os.path.join(output_dir,"part.hdf"), key = "particles")
#merged_data = load_h5_to_dict(os.path.join(output_dir,'merged_data_monitor_prim_coll.h5'))
merged_data = load_h5_to_dict(os.path.join(output_dir,'merged_data_monitor_inject.h5'))
#merged_data = load_h5_to_dict(os.path.join(output_dir,'merged_data_monitor_kick2.h5'))

num_particles = int(len(merged_data['x']))
num_turns = int(len(merged_data['x'].T))
X = 1000*merged_data['x'].reshape(num_particles, num_turns).T
Y = 1000*merged_data['y'].reshape(num_particles, num_turns).T
PX = merged_data['px'].reshape(num_particles, num_turns).T
PY = merged_data['py'].reshape(num_particles, num_turns).T

twiss_dir = 'test_circ_with_coll'
twiss = pd.read_json(os.path.join(twiss_dir,f'twiss_params.json'), orient='split')

X_norm, Y_norm, Px_norm, Py_norm = normalized_coordiantes(twiss, merged_data, 'finj.4')
#X_norm, Y_norm, Px_norm, Py_norm = normalized_coordiantes(twiss, merged_data, 'monitor_kick2')
#X_norm, Y_norm, Px_norm, Py_norm = normalized_coordiantes(twiss, merged_data, 'qi5.4..7')
#X_norm, Y_norm, Px_norm, Py_norm = normalized_coordiantes(twiss, merged_data, 'tcp.h.b1')

#turns = range(0,num_turns)  
#turns = range(0, 20)
#turns = [0, 8,9,10,11,50, 100, 150, 200, 250,  300, 350,  400, 450, 499]
turns = [0, 9, 10, 11, 12, 13, 14, 15,50 ]

# Set up the figure
fig, ax = plt.subplots(figsize=(6, 6))

ax.set_xlim(-10, 10)
ax.set_ylim(-2, 2)
ax.set_xlabel(r'x [mm]')
ax.set_ylabel(r'y [mm]')

ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
#ax.set_xlabel(r'y [$\sigma$]')
#ax.set_ylabel(r'py [$\sigma$]')
ax.set_xlabel(r'x [$\sigma$]')
#ax.set_ylabel(r'px [$\sigma$]')
ax.set_ylabel(r'y [$\sigma$]')

ax.grid(True, linestyle='--', alpha=0.5)
scat = ax.scatter([], [], s=0.5)

# Function to generate each frame and save it as a PNG
for turn in turns:
    print(f"Saving frame for turn {turn}")
    ax.set_title(f"Turn {turn}")
    #scat.set_offsets(np.c_[X[turn], Y[turn]])
    scat.set_offsets(np.c_[X_norm.T[turn], Y_norm.T[turn]])
    #scat.set_offsets(np.c_[Y_norm.T[turn], Py_norm.T[turn]])
    #scat.set_offsets(np.c_[X_norm.T[turn], Px_norm.T[turn]])
    
    # Save the current frame as a PNG
    plt.savefig(os.path.join(output_dir,f"particle_turn_{turn:03d}_inj.png"), dpi=300)

plt.close(fig)
print("All frames saved as PNG files.")
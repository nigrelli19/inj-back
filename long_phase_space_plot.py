import os
import json
import h5py
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
#%matplotlib inline
import xtrack as xt
import xcoll as xc


def load_h5_to_dict(file_path):
    data_dict = {}
    
    with h5py.File(file_path, 'r') as h5f:
        for key in h5f.keys():
            # Load each dataset and store in the dictionary
            data_dict[key] = h5f[key][:]
    
    return data_dict

output_dir = 'dataset/all_inj_2.5k'
merged_data = load_h5_to_dict(os.path.join(output_dir,'merged_data_monitor_inject.h5'))

num_particles = int(len(merged_data['x']))
num_turns = int(len(merged_data['x'].T))
X = merged_data['x'].reshape(num_particles, num_turns).T
Y = merged_data['y'].reshape(num_particles, num_turns).T
PX = merged_data['px'].reshape(num_particles, num_turns).T
PY = merged_data['py'].reshape(num_particles, num_turns).T
zeta = merged_data['zeta'].reshape(num_particles, num_turns).T
delta = merged_data['delta'].reshape(num_particles, num_turns).T

line =xt.Line.from_json('lines_&_coll/tapered_z_b1_thin_2.25pm.seq.json')
df_line = line.to_pandas()
tw = line.twiss()

first_RF_name = 'ca1.1'
number_of_RF_in_row = len(df_line[df_line['element_type'] == 'Cavity']) # 6

# line[''].lag gives the phase seen by the reference particle in degrees.
phi_s_rad = (line[first_RF_name].lag+line[first_RF_name].lag_taper)*np.pi/180

a1 = (line.particle_ref.q0 * number_of_RF_in_row*line[first_RF_name].voltage / (2*np.pi*line[first_RF_name].frequency * tw.circumference * tw.p0c/sp.constants.c))

a2 = 2*np.pi*line[first_RF_name].frequency / sp.constants.c

H_ufp = a1*(np.sin(phi_s_rad)*(2*phi_s_rad - np.pi) + np.cos(phi_s_rad))

H_z_usp = lambda qq, pqq: a1*(np.sin(phi_s_rad)*a2*qq - np.cos(phi_s_rad-a2*qq)) - 0.5*tw.slip_factor*pqq**2 - H_ufp

qq_vals = np.linspace(-0.2, 0.4, 500)  # Longitudinal displacement
pqq_vals = np.linspace(-0.017722, 0.017722, 500)  # Momentum deviation (LEAVE L=IT LIKE THIS TO SEE THE SEPARATRIX)
QQ, PQQ = np.meshgrid(qq_vals, pqq_vals)

# Compute Hamiltonian values
H_vals = H_z_usp(QQ, PQQ)

# Define Contour Levels
contour_levels = np.linspace(np.min(H_vals), np.max(H_vals), 10)

# Create the Plot
plt.figure(figsize=(8, 6))

# Plot Hamiltonian Contours
for level in contour_levels:
    linestyle = "solid" if level >= 0 else "dotted"
    plt.contour(QQ, PQQ, H_vals, levels=[level], colors="black", linestyles=linestyle)

turns = np.arange(0, num_turns)

# Flatten arrays and assign turn-based colors
sc = plt.scatter(
    zeta.flatten(), 
    delta.flatten(), 
    c=np.repeat(turns, zeta.shape[1]),  # Assign colors based on turn indices
    cmap='viridis', 
    s=5,  # Marker size
    edgecolors='none'
)

# Labels & Formatting
plt.xlabel(r"$\zeta$")
plt.ylabel(r"$\delta$")
plt.title(r"Hamiltonian Contours with Particle Motion")

# Axis Limits (adjust based on your data)
plt.ylim(-0.015, 0.015)
plt.xlim(-0.2, 0.3)

# Add Color Bar for Turns
cbar = plt.colorbar(sc)
cbar.set_label('Turn')

plt.grid(True)
plt.savefig(os.path.join(output_dir, 'zeta_vs_delta.png'))
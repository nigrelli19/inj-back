import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

def load_h5_to_dict(file_path):
    """Load an HDF5 file into a dictionary."""
    data_dict = {}
    with h5py.File(file_path, 'r') as h5f:
        for key in h5f.keys():
            data_dict[key] = h5f[key][:]
    return data_dict

def compute_emittance(X, PX):
    """Compute the emittance given position and momentum arrays."""
    mean_x2 = np.mean(X**2, axis=1)
    mean_xp2 = np.mean(PX**2, axis=1)
    mean_x_xp = np.mean(X * PX, axis=1)
    return np.sqrt(mean_x2 * mean_xp2 - mean_x_xp**2)

# User Input for file paths
dataset_paths = [
    input("Enter the path for dataset : "),
    input("Enter the path for dataset 2: ")
]

# Store emittance results
emittance_results = {}

for idx, path in enumerate(dataset_paths):
    merged_data = load_h5_to_dict(os.path.join(path, 'merged_data_monitor_prim_coll.h5'))
    
    num_particles = merged_data['x'].shape[0]
    num_turns = merged_data['x'].shape[1]
    
    X, Y = merged_data['x'].T, merged_data['y'].T
    PX, PY = merged_data['px'].T, merged_data['py'].T
    zeta, delta = merged_data['zeta'].T, merged_data['delta'].T
    
    emittance_results[f'dataset_{idx+1}'] = {
        'emitt_X': compute_emittance(X, PX),
        'emitt_Y': compute_emittance(Y, PY),
        'emitt_Z': compute_emittance(zeta, delta),
        'turns': np.arange(num_turns)
    }

# Plot results
plt.figure(figsize=(10, 8))
colors = ['blue', 'orange']
labels = ['Dataset 1', 'Dataset 2']

for idx, key in enumerate(emittance_results.keys()):
    data = emittance_results[key]
    plt.scatter(data['turns'], data['emitt_Y'] * 1e12, s=5,  label=f'{labels[idx]} - Vertical')
    plt.scatter(data['turns'], data['emitt_X'] * 1e9, s=5, marker='x', label=f'{labels[idx]} - Horizontal')
    plt.scatter(data['turns'], data['emitt_Z'] * 1e6, s=5, marker='s', label=f'{labels[idx]} - Longitudinal')

plt.xlabel("Turn", fontsize=12)
plt.ylabel(r"$\varepsilon_{\text{RMS}}$", fontsize=12)
plt.title("Emittance Growth Comparison", fontsize=14)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()


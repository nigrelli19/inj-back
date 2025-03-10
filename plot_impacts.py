import os
import re
import glob
import json
import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import warn
#%matplotlib inline

import xtrack as xt
import xcoll as xc
POW_TOT =  17.5e6 # J

def create_bins(sigma, min_val, max_val, steps):
    bins = []
    for factor, step in steps:
        start = min_val + factor * sigma
        stop = max_val + factor * sigma
        bins.append(np.linspace(start, stop, step))
    return np.concatenate(bins)

def merge_csv_files_with_offset(root_directory, file_name="plots/data.csv"):
    """
    Merges CSV files from nested directories, keeps only specified columns, and adds a cumulative offset to 'id_before',
    ensuring files are processed in numerical order.
    
    Parameters:
    root_directory (str): The root directory containing all nested directories with CSV files.
    file_name (str): The relative path pattern to the CSV files within each nested directory.
                     Default is "plots/data.csv".
    
    Returns:
    pd.DataFrame: A single DataFrame containing data from all found CSV files.
    """
    # Construct the file path pattern for all files in subdirectories
    search_pattern = os.path.join(root_directory, "**", file_name)
    
    # Recursively find all matching CSV files
    csv_files = glob.glob(search_pattern, recursive=True)
    
    # Sort files by job number (extracted from the filename pattern) to ensure they are in numerical order
    csv_files = sorted(csv_files, key=lambda x: int(re.search(r'Job\.(\d+)', x).group(1)))
    
    # List to hold each DataFrame
    dataframes = []
    cumulative_offset = 0  # To keep track of the cumulative offset for id_before

    # Loop through each found file and append its DataFrame to the list
    for file in csv_files:
        # Define columns to keep
        columns_to_keep = ['turn', 'collimator', 'interaction_type', 'id_before', 's_before',
                           'x_before', 'px_before', 'y_before', 'py_before', 'zeta_before',
                           'delta_before', 'energy_before']
        
        # Read the CSV file with only the required columns to optimize memory usage
        df = pd.read_csv(file, usecols=columns_to_keep)
        
        # Apply the cumulative offset to 'id_before'
        df['id_before'] += cumulative_offset
        
        # Update cumulative offset for the next file
        cumulative_offset = df['id_before'].max() + 1  # Add 1 to ensure unique ids across files

        # Append the DataFrame to the list
        dataframes.append(df)
    
    # Concatenate all DataFrames into one
    merged_df = pd.concat(dataframes, ignore_index=True)

    return merged_df

def generate_impacts(line, particles, ref_part, num_turns, seed, bdsim_config, output_dir): 
    '''
    Function to produce impacts on jaws per turn    
    '''
    impacts = xc.InteractionRecord.start(line=line)

    xc.Geant4Engine.start(line=line,
                                particle_ref=ref_part,  
                                seed=seed,
                                bdsim_config_file=bdsim_config)

    line.scattering.enable()
    # Track (saving turn-by-turn data)
    for turn in range(num_turns):
        print(f'Start turn {turn}, Survivng particles: {particles._num_active_particles}')
        if turn == 0 and particles.start_tracking_at_element < 0:
            line.track(particles, num_turns=1)
        else:
            line.track(particles, num_turns=1)

        if particles._num_active_particles == 0:
            print(f'All particles lost by turn {turn}, teminating.')
            break

    line.scattering.disable()
    line.discard_tracker()
    xc.Geant4Engine.stop()
    impacts.stop()

    df = impacts.to_pandas()
    df.to_csv(os.path.join(output_dir,'impacts_line.csv'), index=False)

    return df

def plot_energy2d(data, coll_plane, bins = None,**kwargs):
    # Binning the data
    if coll_plane == 'v':
        x = 'y_before'
        y = 'x_before'
    elif coll_plane == 'h':
        y = 'y_before'
        x = 'x_before'
    else:
        raise ValueError("coll_plane must be 'v' or 'h'")
    
    if bins is None:
        bins_x = np.linspace(0, 0.02, 1000)  # Default linear bins for x
        bins_y = np.linspace(-0.03, 0.03, 1000)    # Adjust as needed for y
    elif isinstance(bins, tuple):
        bins_x, bins_y = bins
    else:
        bins_x = bins_y = bins

    weights = data['energy_before'] * POW_TOT / data['energy_before'].sum()
    hist, x_edges, y_edges = np.histogram2d(
        data[x]*1000, data[y]*1000, bins=[bins_x*1000, bins_y*1000], weights=weights
    )
    # Plot the energy heatmap
    img = plt.imshow(
        hist.T, 
        origin='lower',
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        cmap='plasma',
        aspect='auto'
    )

    # Add colorbar
    cbar = plt.colorbar(img, label='Energy lost [J]',anchor=(0.5,0.5))
    #cbar.ax.yaxis.set_label_position('right')  # Move label to the left
    #cbar.ax.yaxis.tick_right()  # Move ticks to the left
    #cbar.ax.yaxis.set_ticks_position('right')


def main(output_dir, coll_type, coll_plane):

    df_impacts = pd.read_csv(os.path.join(output_dir,'impacts_line.csv'))
    df_impacts= df_impacts.rename(columns={'id_before': 'particle_id'})

    df_part = pd.read_hdf(os.path.join(output_dir,'part_merged.hdf'), key = "particles") 
    df_part['part_mass_ratio'] = df_part['charge_ratio'] / df_part['chi']
    df_part['part_mass'] = df_part['part_mass_ratio'] * df_part['mass0']
    p0c = df_part['p0c']
    df_part['energy'] = np.sqrt(((df_part['delta'] + 1) * p0c * df_part['part_mass_ratio'])**2 + df_part['part_mass']**2)

    # Count secondary particles
    secnd_part = df_part[df_part['parent_particle_id'] != df_part['particle_id']]
    print(f'Number of secondary particles: {len(secnd_part)}')

    # Counts of particle that hit the collimator many times
    multi_hit_mask = (df_impacts.duplicated(subset='particle_id', keep=False))
    multi_hit = df_impacts[multi_hit_mask]
    print(f'Number of particles that have multiple hits: {len(multi_hit)}')
    multi_hit_diff_coll = multi_hit[~multi_hit.duplicated(subset='collimator', keep=False)]
    print(f'Of which {len(multi_hit_diff_coll)} hit different collimators.')

    # Extraction of information about losses in the aperture
    # Select all the particle id that hit the collimator, either stopping or not
    coll_hit_part_id = df_impacts['particle_id'].unique()
    # Select particles NOT in the list of coll hit
    df_part_not_in_impacts = df_part[~(df_part['particle_id'].isin(coll_hit_part_id))]
    # Select particles not lost in collimators != -333
    # TO DO: CHECK THIS -337 CODE
    df_part_not_impacts_aper = df_part_not_in_impacts[(df_part_not_in_impacts['state'] !=  -333) & (df_part_not_in_impacts['at_turn'] != 500)] 
    # Select secondary particles
    aper_loss_from_secnd = df_part_not_impacts_aper[df_part_not_impacts_aper['particle_id'] != df_part_not_impacts_aper['parent_particle_id']]
    aper_loss_from_prim = df_part_not_impacts_aper[df_part_not_impacts_aper['particle_id'] == df_part_not_impacts_aper['parent_particle_id']]
    aper_loss = df_part[(df_part['state'] != (-333)) & (df_part['at_turn'] != 500)]
    print(f'Total number of particles lost in the apertures:{len(aper_loss)}')
    print(f'Of this, {len(df_part_not_impacts_aper)} are first interactions.')
    print(f'{(len(aper_loss_from_secnd)*100/len(df_part_not_impacts_aper))} % of this is due to secondary particles.')
    print(f'So, {(len(aper_loss_from_prim)*100/len(df_part_not_impacts_aper))} % is due to primaries.')


    #df_impacts = df_impacts[df_impacts['collimator'] == 'tcp.h.b1']
    coll = coll_type + '.' + coll_plane
    df_single_hit = df_impacts[~multi_hit_mask]
    prim_part = df_part[df_part['parent_particle_id'] == df_part['particle_id']]
    prim_on_tct = df_single_hit[df_single_hit['collimator'].str.contains('tct.v', case=False)]
    print(f'Total number of primary particles on tct.v s:{(len(prim_on_tct))}') #/len(prim_part)*100

    #df_impacts = pd.merge(df_single_hit, df_part, on=['particle_id'])
    df_impacts = df_impacts[(df_impacts['collimator'].str.contains(coll, case=False))]  
    
      # Set up the FacetGrid with separate plots for each combination of collimator and interaction_type
    '''g_x = sns.FacetGrid(df_impacts, col='collimator', row='interaction_type', sharex= False,sharey=False, hue='turn', palette='viridis', margin_titles=True)

    # Map histograms to each subplot in the grid for x_before
    g_x.map_dataframe(sns.histplot, x='x_before', log_scale=False , stat="count", alpha=0.5,bins='auto' )# binrange=(0,0.0018), bins = 1000)# bins='auto')

    if coll_plane == 'v':
        g_x.set_axis_labels("y[m]", "# particles")
        name = f'impacts_y_{coll}.png'
    elif coll_plane == 'h':
        g_x.set_axis_labels("x[m]", "# particles")
        name = f'impacts_x_{coll}.png'
    g_x.set_titles(row_template="{row_name}", col_template="{col_name}")
    g_x.add_legend(title="Turn")

    x_max = max(df_impacts['x_before'])
    g_x.set(xlim=(0,0.004))
    #g_x.set(ylim=(0, 300))
    g_x.set_xticklabels(size=8)  # X-axis tick label font size
    g_x.set_yticklabels(size=8)    
    

    plt.savefig(os.path.join(output_dir,name), dpi=300, bbox_inches='tight') 
    plt.show()
    plt.close()
    

    # Set up the FacetGrid with separate plots for each combination of collimator and interaction_type
    g_y = sns.FacetGrid(df_impacts, col='collimator', row='interaction_type', hue='turn', sharex= False,sharey=False,palette='viridis', margin_titles=True)

    # Map histograms to each subplot in the grid for x_before
    g_y.map_dataframe(sns.histplot, x='y_before', log_scale=False, stat="count", alpha=0.5,bins='auto')# binrange=(-0.032,0.05), bins = 1000)#bins='auto')

    # Add legends and adjust titles
    g_y.add_legend(title="Turn")
    y_max = max(df_impacts['y_before'])
    y_min = min(df_impacts['y_before'])
    #g_y.set(xlim=(y_min+0.001,y_max-0.001))
    g_y.set(xlim=(-0.005, 0.005))

    if coll_plane == 'v':
        g_y.set_axis_labels("x[m]", "# particles")
        name_y = f'impacts_x_{coll}.png'
    elif coll_plane == 'h':
        g_y.set_axis_labels("y[m]", "# particles")
        name_y = f'impacts_y_{coll}.png'

    g_y.set_xticklabels(size=8)  # X-axis tick label font size
    g_y.set_yticklabels(size=8) 

    g_y.set_titles(row_template="{row_name}", col_template="{col_name}")
    plt.savefig(os.path.join(output_dir,name_y), dpi=300, bbox_inches='tight') 
    plt.show()
    plt.close()'''


    #g_density = sns.FacetGrid(df_impacts, col='collimator', row='interaction_type', margin_titles=True)
    #df_impacts = df_impacts.dropna(subset=['x_before', 'y_before'])  # Remove NaNs
    #df_impacts = df_impacts[(df_impacts['x_before'] != float('inf')) & (df_impacts['y_before'] != float('inf'))]  # Remove inf values
    
    print(f"x range: {df_impacts['x_before'].min()} to {df_impacts['x_before'].max()}")
    print(f"y range: {df_impacts['y_before'].min()} to {df_impacts['y_before'].max()}")
   
    sigma_y = df_impacts['y_before'].std()
    sigma_x = df_impacts['x_before'].std()

    bins_x = np.linspace(-0.0025, 0.0025, 500)  # Default linear bins for x
    bins_y = np.linspace(0, 0.004, 500) 
    #bins_x = np.linspace(0, 0.004, 500) 
    #bins_y = np.linspace(-0.0030, 0.003, 500) 
    
    g = sns.FacetGrid(df_impacts, col='collimator', row='interaction_type', sharex=False, sharey=False, height=4, margin_titles=True)
    g.map_dataframe(plot_energy2d,coll_plane=coll_plane,bins=(bins_x, bins_y))
    # Add titles and adjust layout
    g.set_axis_labels('X[mm]', 'Y[mm]')
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    g.figure.subplots_adjust(top=0.9)
    g.set(ylim=(0, 4))
    #g.set(ylim=(-0.15, 0.15))
    #g.figure.suptitle('FacetGrid with Energy per cm²', fontsize=16)
    plt.savefig(os.path.join(output_dir,f'impacts_2D_{coll}.png'), dpi=300, bbox_inches='tight') 
    plt.show()
    plt.close()
    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Track particles with a kicker and plot the results.')
    parser.add_argument('--work_dir', type=str, required=True, help='Directory where to look for impacts file and where to save outputs. Example: dataset/new_vertical/3_turns/ver_phase_90_3turns/')
    parser.add_argument('--coll_type', type=str, required=True, help='Type of collimator to produce the plot, for ex: tcp, or tcs, or tct etc..')
    parser.add_argument('--coll_plane', type=str, required=True, help='Plane of collimator v (vertical) or h (horizontal).')
    args = parser.parse_args()
    # Call the main function with parsed arguments
    main(args.work_dir, args.coll_type, args.coll_plane)



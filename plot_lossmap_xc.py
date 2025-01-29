import os
import re
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

POW_TOT =  17.5e6 # J

FCC_EE_WARM_REGIONS = np.array([
    [0.0, 2.2002250210956813], [2.900225021095681, 2.9802250210956807], 
    [4.230225021095681, 4.310225021095681], [5.560225021095681, 5.860225021095681], 
    [7.110225021095681, 7.1902250210956815], [8.440225021095682, 19.021636933288438], 
    [21.921636933288436, 24.095566014890213], [26.995566014890212, 83.9467184323269], 
    [86.84671843232691, 94.10927511652588], [97.00927511652588, 122.25505466852684], 
    [125.15505466852684, 125.45505466852684], [125.68005466852684, 125.68005466852685], 
    [125.90505466852684, 280.1903764070126], [280.4153764070126, 280.4153764070126], 
    [280.64037640701264, 22011.935992171686], [22012.160992171684, 22012.160992171684], 
    [22012.385992171683, 22225.188674086643], [22225.41367408664, 22225.41367408664], 
    [22225.63867408664, 22225.93867408664], [22228.83867408664, 22243.586622081384], 
    [22246.486622081386, 22335.403176079093], [22338.303176079095, 22467.336337002067], 
    [22470.23633700207, 22547.19338558328], [22550.09338558328, 22656.24610479999], 
    [22657.49610479999, 22657.57610479999], [22658.82610479999, 22659.126104799994], 
    [22660.376104799994, 22660.456104799996], [22661.706104799996, 22661.786104799998], 
    [22662.4861048, 22666.8865548422], [22667.5865548422, 22667.66655484221], 
    [22668.91655484221, 22668.996554842208], [22670.246554842208, 22670.546554842207], 
    [22671.796554842207, 22671.87655484221], [22673.12655484221, 22683.707966754406], 
    [22686.607966754407, 22688.781895836008], [22691.68189583601, 22748.63304825344], 
    [22751.53304825344, 22758.795604937637], [22761.69560493764, 22786.941384489633], 
    [22789.841384489635, 22790.141384489634], [22790.366384489633, 22790.366384489633], 
    [22790.59138448963, 22944.876706228104], [22945.101706228103, 22945.101706228106], 
    [22945.326706228105, 44676.622330340004], [44676.84733034, 44676.847330339995], 
    [44677.072330339994, 44889.87501225503], [44890.100012255025, 44890.100012255025], 
    [44890.32501225502, 44890.625012255034], [44893.525012255035, 44908.27296024977], 
    [44911.17296024977, 45000.089514247484], [45002.989514247485, 45132.02267517048], 
    [45134.922675170485, 45211.8797237517], [45214.7797237517, 45320.932442968406], 
    [45322.182442968406, 45322.26244296841], [45323.51244296841, 45323.8124429684], 
    [45325.0624429684, 45325.1424429684], [45326.3924429684, 45326.47244296841], 
    [45327.1724429684, 45331.57289301058], [45332.27289301058, 45332.35289301058], 
    [45333.60289301058, 45333.68289301059], [45334.93289301059, 45335.232893010594], 
    [45336.482893010594, 45336.5628930106], [45337.8128930106, 45348.394304922804], 
    [45351.294304922805, 45353.46823400441], [45356.36823400441, 45413.31938642187], 
    [45416.21938642187, 45423.481943106075], [45426.38194310608, 45451.62772265809], 
    [45454.527722658095, 45454.8277226581], [45455.052722658096, 45455.052722658096], 
    [45455.277722658095, 45609.56304439665], [45609.78804439665, 45609.78804439665], 
    [45610.01304439665, 67341.30866016546], [67341.53366016547, 67341.53366016547], 
    [67341.75866016548, 67554.56134208044], [67554.78634208045, 67554.78634208045], 
    [67555.01134208046, 67555.31134208046], [67558.21134208045, 67572.9592900752], 
    [67575.8592900752, 67664.77584407291], [67667.6758440729, 67796.70900499586], 
    [67799.60900499586, 67876.5660535771], [67879.4660535771, 67985.6187727938], 
    [67986.8687727938, 67986.9487727938], [67988.1987727938, 67988.49877279381], 
    [67989.74877279381, 67989.82877279382], [67991.07877279382, 67991.15877279382], 
    [67991.85877279381, 67996.25922283597], [67996.95922283597, 67997.03922283597], 
    [67998.28922283597, 67998.36922283597], [67999.61922283597, 67999.91922283598], 
    [68001.16922283598, 68001.24922283598], [68002.49922283598, 68013.08063474817], 
    [68015.98063474816, 68018.15456382977], [68021.05456382976, 68078.00571624722], 
    [68080.90571624722, 68088.16827293142], [68091.06827293141, 68116.31405248342], 
    [68119.21405248341, 68119.51405248341], [68119.73905248342, 68119.73905248342], 
    [68119.96405248342, 68274.24937422191], [68274.47437422191, 68274.47437422191], 
    [68274.69937422192, 90005.9949899894], [90006.2199899894, 90006.21998998942], 
    [90006.44498998942, 90219.24767190438], [90219.47267190438, 90219.47267190438], 
    [90219.69767190439, 90219.99767190439], [90222.89767190439, 90237.64561989912], 
    [90240.54561989912, 90329.46217389684], [90332.36217389684, 90461.39533481981], 
    [90464.2953348198, 90541.25238340105], [90544.15238340104, 90650.30510261773], 
    [90651.55510261773, 90651.63510261773], [90652.88510261773, 90653.18510261775], 
    [90654.43510261775, 90654.51510261776], [90655.76510261776, 90655.84510261775]])
     
def check_warm_loss(s, warm_regions):
    # FOR LATTICE V24.4_GHC
    x_shift = 22664.626431  # 2032.000 + 1400.000 + 9616.175
    x_wrap = 90658.50572430472 # Length of the machine
    # Wrap the s value
    s_wrapped = (s + x_shift) % x_wrap

    return np.any((warm_regions.T[0] < s_wrapped) & (warm_regions.T[1] > s_wrapped))

def load_multiple_lossmaps(base_dir, output_dir, single_file=None):
    '''
    Function to produce datframe to handles better the datasets.
    '''
    merged_data = {
        'collimator': {'s': [], 'name': [], 'length': [], 'n': []},
        'aperture': {'s': [], 'name': [], 'length': [], 'n': []}
    }
    missing_files = 0
    from_part = False
        # Check if a single file is provided
    if single_file:
        file_path = os.path.join(base_dir,single_file)
        if os.path.exists(file_path):
            df = pd.read_json(file_path)
            if 'collimator' in df.columns and 'aperture' in df.columns:
                # Safely merge the lists
                
                for col in ['collimator', 'aperture']:
                    for row in ['s', 'name', 'length', 'n']:
                        data = df.at[row, col]
                        if isinstance(data, list):
                            merged_data[col][row].extend(data)
                        else:
                            merged_data[col][row].append(data)
            else:
                print(f"Columns 'collimator' or 'aperture' missing in {file_path}")
                from_part = True
                merged_data['collimator']['length'] = df['coll_end'] - df['coll_start']
                merged_data['collimator']['s'] = df['coll_start'] + (merged_data['collimator']['length'] / 2)
                merged_data['collimator']['name'] = df['coll_name'].tolist()
                merged_data['collimator']['n'] = df['coll_loss'].fillna(0).tolist()
                merged_data['aperture']['n'] = df['aper_loss'].fillna(0).tolist()

        else:
            print(f"File does not exist: {file_path}")
    else:
        # Loop through directories Job.0 to Job.99
        for i in range(100):  # Adjust as needed for your range of jobs
            job_dir = f"Job.{i}/plots"
            file_path = os.path.join(base_dir, job_dir, 'lossmap.json')

            if os.path.exists(file_path):
                df = pd.read_json(file_path)
                if 'collimator' in df.columns and 'aperture' in df.columns:
                    # Safely merge the lists
                    for col in ['collimator', 'aperture']:
                        for row in ['s', 'name', 'length', 'n']:
                            data = df.at[row, col]
                            if isinstance(data, list):
                                merged_data[col][row].extend(data)
                            else:
                                merged_data[col][row].append(data)
                else:
                    print(f"Columns 'collimator' or 'aperture' missing in {file_path}")
            else:
                print(f"Warning: {file_path} does not exist.")
                missing_files += 1
        
        save_lossmap_to_json(merged_data, os.path.join(output_dir,'merged_lossmap.json'))
        print(missing_files)

    collimator_df = pd.DataFrame({
    's': merged_data['collimator']['s'],
    'name': merged_data['collimator']['name'],
    'length': merged_data['collimator']['length'],
    'n': merged_data['collimator']['n'],
    'type': 'collimator'  # Label to distinguish between types
    })  

    if not from_part:

        aperture_df = pd.DataFrame({
            's': merged_data['aperture']['s'],
            'name': merged_data['aperture']['name'],
            #'length': merged_data['aperture']['length'],
            'n': merged_data['aperture']['n'],
            'type': 'aperture'  # Label to distinguish between types
        })

    else:
        aperture_df = pd.DataFrame({
            #'s': merged_data['aperture']['s'],
            #'name': merged_data['aperture']['name'],
            #'length': merged_data['aperture']['length'],
            'n': merged_data['aperture']['n'],
            'type': 'aperture'  # Label to distinguish between types
        })


    combined_df = pd.concat([collimator_df, aperture_df], ignore_index=True)

    # Convert the merged_data dictionary into a final DataFrame
    return collimator_df, aperture_df, from_part

def save_lossmap_to_json(merged_data, output_file):
    """
    Saves the merged lossmap data into a JSON file.
    """
    try:
        with open(output_file, 'w') as json_file:
            json.dump(merged_data, json_file, indent=4)
        print(f"Lossmap data successfully saved to {output_file}")
    except Exception as e:
        print(f"Error saving lossmap data to JSON file: {e}")

def prepare_lossmap_values(base_dir, output_dir, s_min, s_max, single_file, norm='none', tot_energy_full = 0):
    '''
    Function to process the losses data to produce datframe of collimator losses and aperture losses. It handles both json file from original script in htcondor and also the one produces from particles.hdf 
    '''
    lossmap_norms = ['none', 'max', 'coll_max','total', 'coll_total', 'coll_turn']
    if norm not in lossmap_norms:
        raise ValueError('norm must be in [{}]'.format(', '.join(lossmap_norms)))

    bin_w = 0.10
    nbins = int(np.ceil((s_max - s_min)/bin_w))
    
    warm_regions = FCC_EE_WARM_REGIONS


    if single_file:
        coll_group, ap_group, from_part = load_multiple_lossmaps(base_dir, output_dir, single_file=single_file)
    else: 
        coll_group, ap_group = load_multiple_lossmaps(base_dir, output_dir)

    if not from_part:

        ap_group = ap_group.groupby('name').agg(
            n=('n', 'sum'),           # Sum of 'n'
            s=('s', 'mean'),        # Average of 's'
        ).reset_index()
                
        ap_s = ap_group['s']
        aper_loss = ap_group['n']
        aper_edges = np.linspace(s_min, s_max, nbins + 1)

        coll_group = coll_group.groupby('name').agg(
            n=('n', 'sum'),           # Sum of 'n'
            s=('s', 'mean'),        # Average of 's'
            length=('length', 'mean')   # Sum of 'n' for each group
        ).reset_index()

    else:
        aper_edges = np.linspace(s_min, s_max, nbins)
        aper_loss = ap_group['n'].reindex(range(0, nbins-1), fill_value=0)
    
    coll_name = coll_group['name']
    coll_loss = coll_group['n']
    coll_s = coll_group['s']
    coll_length = coll_group['length']

    coll_end = coll_s + (coll_length/2)
    coll_start = coll_s - (coll_length/2)    
    
    if norm == 'total':
        norm_val = sum(coll_loss) + sum(aper_loss) # This is not exactly the energy of the initial beam due to the secondary particles/interaction where energy is lost
    elif norm == 'max':
        norm_val = max(max(coll_loss), max(aper_loss))
    elif norm == 'coll_total':
        norm_val = sum(coll_loss)
    elif norm == 'coll_max':
        norm_val = max(coll_loss)
    elif norm == 'none':
        norm_val = 1
    elif norm == 'coll_turn':
        norm_val = tot_energy_full
        #norm_val = NORM_VAL[0][0] #TO CHANGE FOR EVERY CONFIGURATION, NEEDS TO BE IMPROVED
    
    tot_energy = sum(coll_loss) + sum(aper_loss)
    # To have the lossmap express in power: mulitply the loss factor(100% in this case) 
    # for the real energy stored energy divided by the lifetime

    #pow_tot = 17.5e6 # J

    if aper_loss.sum() > 0:
        
        aper_loss /= (norm_val * bin_w)

        if from_part:
            mask_warm = np.array([check_warm_loss(s, warm_regions)
                         for s in aper_edges[:-1]])

            ap_warm = aper_loss * mask_warm
            ap_cold = aper_loss * ~mask_warm

            ap_warm_pow = ap_warm * POW_TOT * bin_w
            ap_cold_pow = ap_cold * POW_TOT * bin_w

        else:
            mask_warm = np.array([check_warm_loss(s, warm_regions)
                                for s in ap_s])

            warm_loss = aper_loss * mask_warm
            cold_loss = aper_loss * ~mask_warm

            ap_warm = np.zeros(len(aper_edges) - 1)
            ap_cold = np.zeros(len(aper_edges) - 1)

            ap_warm_pow = np.zeros(len(aper_edges) - 1)
            ap_cold_pow = np.zeros(len(aper_edges) - 1)

            ap_indices = np.digitize(ap_s, aper_edges)

            np.add.at(ap_warm, ap_indices[mask_warm] - 1, warm_loss[mask_warm])
            np.add.at(ap_cold, ap_indices[~mask_warm] - 1, cold_loss[~mask_warm])

            warm_loss_pow = warm_loss * POW_TOT * bin_w
            cold_loss_pow = cold_loss * POW_TOT * bin_w

            np.add.at(ap_warm_pow, ap_indices[mask_warm] - 1, warm_loss_pow[mask_warm])
            np.add.at(ap_cold_pow, ap_indices[~mask_warm] - 1, cold_loss_pow[~mask_warm])

        '''for i in range(len(cold_loss_pow)):
            if cold_loss_pow[i] != 0:
                print(cold_loss_pow[i])
                print(aper_edges[i])'''
    else:
        aper_edges = [0]
        ap_warm = [0]
        ap_cold = [0]
        ap_warm_pow = [0]
        ap_cold_pow = [0]

    if coll_loss.sum() > 0:
        coll_pow = coll_loss * POW_TOT/ norm_val #loss fraction as energy_lost/(tot_energy_lost)
        coll_loss /= (norm_val * coll_length)
        zeros = np.full_like(coll_group.index, 0)  # Zeros to pad the bars
        coll_edges = np.dstack([coll_start, coll_start, coll_end, coll_end]).flatten()
        coll_loss = np.dstack([zeros, coll_loss, coll_loss, zeros]).flatten()
        coll_pow = np.dstack([zeros, coll_pow, coll_pow, zeros]).flatten()

    else:
        coll_edges = [0]
        coll_loss = [0]
        coll_pow = [0]    

    return coll_edges, coll_loss, coll_pow, aper_edges, ap_warm, ap_cold, ap_warm_pow, ap_cold_pow, tot_energy

def plot_lossmaps(base_dir, output_dir, twiss, single_file, output_file, norm='none', tot_energy_full = 0):
    '''
    Function to plot lossmap in cleaning efficency, energy lost and zoom in IPG and IPF.
    '''
    # Load twiss parameters
    twiss = pd.read_json(twiss, orient='split')
    s_min, s_max = twiss.s.min(), twiss.s.max()
    
    coll_edges, coll_loss, coll_pow, aper_edges, ap_warm, ap_cold, ap_warm_pow, ap_cold_pow, tot_energy = prepare_lossmap_values(base_dir, output_dir, s_min, s_max, single_file, norm, tot_energy_full)
    
    fig, ax = plt.subplots(figsize=(18, 6))

    # UNCOMMENT FOR LLSS COMMON LATTICE

    x_shift = 22664.626431 #2032.000 + 1400.000 + 9616.175 
    x_wrap = s_max
    
    if coll_edges is None or len(coll_edges) == 0:  # Check if aper_edges is empty
        print("Warning: aper_edges is empty. Skipping operation.")
        return  # or handle the empty case appropriately
    else:
        coll_edges = [(edge + x_shift) % x_wrap for edge in coll_edges]
    
    if aper_edges is None or len(aper_edges) == 0:   # Check if aper_edges is empty
        print("Warning: aper_edges is empty. Skipping operation.")
        return  # or handle the empty case appropriately
    else:
        #aper_edges = (aper_edges + x_shift)  % x_wrap
        aper_edges = [(edge + x_shift) % x_wrap for edge in aper_edges]

    lw=1
    if np.sum(coll_loss) == 0:  # Check if any coll_loss value is greater than zero
        print("coll_loss is zero; skipping plot.")
    else:
        ax.fill_between(coll_edges, coll_loss, step='pre', color='k', zorder=9)
        ax.step(coll_edges, coll_loss, color='k', lw=lw, zorder=10, label='Collimator losses')
    
    if np.sum(ap_warm) == 0 and np.sum(ap_cold) == 0:
        print("No losses in the aperture; skipping plot.")
    else:
        # Plot warm losses
        ax.fill_between(aper_edges[:-1], ap_warm, step='post', color='r', zorder=9)
        ax.step(aper_edges[:-1], ap_warm, where='post', color='red', label='Warm', linewidth=1)

        # Plot cold losses
        ax.fill_between(aper_edges[:-1], ap_cold, step='post', color='b', zorder=9)
        ax.step(aper_edges[:-1], ap_cold, where='post', color='blue', label='Cold', linewidth=1)

    plot_margin = 500
    ax.set_xlim(s_min - plot_margin, s_max + plot_margin)

    ax.yaxis.grid(visible=True, which='major', zorder=0)
    ax.yaxis.grid(visible=True, which='minor', zorder=0)

    ax.set_xlabel('s [m]')
    ax.set_yscale('log', nonpositive='clip')
    ax.set_ylabel('Cleaning inefficiency[$m^{-1}$]' )
    #ax.set_title('Lossmap with Exciter')
    ax.legend(loc='upper right')  # Corrected 'upper rigth' to 'upper right' 
    ax.grid()

    plt.tick_params(axis='both', which='major', labelsize=14) 
    # Finalize and save the plot
    plt.savefig(os.path.join(output_dir,f'{output_file}_lossmap.png'),bbox_inches='tight') 
    plt.close()
    # POWER PLOT
    fig, ax_pow = plt.subplots(figsize=(18, 6))
    #fig, ax_pow = plt.subplots(figsize=(5, 7))

    lw=1
    if np.sum(coll_pow) == 0:  # Check if any coll_loss value is greater than zero
        print("coll_loss is zero; skipping plot.")
    else:
        ax_pow.fill_between(coll_edges, coll_pow, step='pre', color='k', zorder=9)
        ax_pow.step(coll_edges, coll_pow, color='k', lw=lw, zorder=10, label='Coll')

    if np.sum(ap_warm_pow) == 0 and np.sum(ap_cold_pow) == 0:
        print("No losses in the aperture; skipping plot.")
    else:
        # Plot warm losses
        ax_pow.fill_between(aper_edges[:-1], ap_warm_pow, step='post', color='r', zorder=9)
        ax_pow.step(aper_edges[:-1], ap_warm_pow, where='post', color='red', label='Warm', linewidth=1)

        # Plot cold losses
        ax_pow.fill_between(aper_edges[:-1], ap_cold_pow, step='post', color='b', zorder=9)
        ax_pow.step(aper_edges[:-1], ap_cold_pow, where='post', color='blue', label='Cold', linewidth=1)

    plot_margin = 500
    ax_pow.set_xlim(s_min - plot_margin, s_max + plot_margin)
    ax_pow.set_ylim(1,2e7)

    ax_pow.yaxis.grid(visible=True, which='major', zorder=0)
    ax_pow.yaxis.grid(visible=True, which='minor', zorder=0)

    ax_pow.set_xlabel('s [m]', fontsize=16)
    ax_pow.set_yscale('log', nonpositive='clip')
    ax_pow.set_ylabel('Energy Lost [$J$]', fontsize=16)
    #ax_pow.set_title('Lossmap in power with Exciter')
    ax_pow.legend(loc='upper right', fontsize=12)  # Corrected 'upper rigth' to 'upper right'
    ax_pow.grid()
    plt.tick_params(axis='both', which='major', labelsize=16) 
    # Finalize and save the plot
    plt.savefig(os.path.join(output_dir,f'{output_file}_pow.png'),bbox_inches='tight') 
    plt.close()
    # Create a zoomed-in plot
    fig_zoom_IPG, ax_zoom_IPG = plt.subplots(figsize=(10, 5))

    # Define the zoom region for the x and y axes
    x_min, x_max = 44500, 46500 #  around IPG  
    #x_min, x_max = 67300, 68700 #  around IPJ  
    #y_min, y_max = 0, 500

    if np.sum(coll_pow) == 0:  # Check if any coll_loss value is greater than zero
        print("coll_loss is zero; skipping plot.")
    else:
        ax_zoom_IPG.fill_between(coll_edges, coll_pow, step='pre', color='k', zorder=9)
        ax_zoom_IPG.step(coll_edges, coll_pow, color='k', lw=lw, zorder=10, label='Collimator losses')
    
    if np.sum(ap_warm_pow) == 0 and np.sum(ap_cold_pow) == 0:
        print("No losses in the aperture; skipping plot.")
    else:
        # Plot warm losses
        ax_zoom_IPG.fill_between(aper_edges[:-1], ap_warm_pow, step='post', color='r', zorder=9)
        ax_zoom_IPG.step(aper_edges[:-1], ap_warm_pow, where='post', color='red', label='Warm losses', linewidth=1)

        # Plot cold losses
        ax_zoom_IPG.fill_between(aper_edges[:-1], ap_cold_pow, step='post', color='b', zorder=9)
        ax_zoom_IPG.step(aper_edges[:-1], ap_cold_pow, where='post', color='blue', label='Cold losses', linewidth=1)
    
    plot_margin = 500
    ax_zoom_IPG.set_xlim(s_min - plot_margin, s_max + plot_margin)

    ax_zoom_IPG.yaxis.grid(visible=True, which='major', zorder=0)
    ax_zoom_IPG.yaxis.grid(visible=True, which='minor', zorder=0)

    # Set the limits for zooming in
    ax_zoom_IPG.set_xlim(x_min, x_max)
    ax_zoom_IPG.set_ylim(0.01, 1e10)

    # Add labels, title, and legend for the zoomed plot
    ax_zoom_IPG.set_xlabel('s[m]')
    ax_zoom_IPG.set_yscale('log', nonpositive='clip')
    #ax_zoom_IPG.set_ylabel('Cleaning inefficiency[$m^{-1}$]')
    ax_zoom_IPG.set_ylabel('Energy Lost [$J$]')
    ax_zoom_IPG.set_title('Zoom at IPG: Lossmap with exciter')
    ax_zoom_IPG.legend(loc='upper right')
    ax_zoom_IPG.grid()

    # Finalize and save the zoomed plot
    plt.savefig(os.path.join(output_dir,f'{output_file}_zoom_IPG.png'))
    plt.close()
    # Zoom in other range
    fig_zoom_IPF, ax_zoom_IPF = plt.subplots(figsize=(10, 5))

    # Define the zoom region for the x and y axes
    x_min, x_max = 33000, 35500  # around collimation insertion IPF
    #y_min, y_max = 0, 500

    if np.sum(coll_pow) == 0:  # Check if any coll_loss value is greater than zero
        print("coll_loss is zero; skipping plot.")
    else:
        ax_zoom_IPF.fill_between(coll_edges, coll_pow, step='pre', color='k', zorder=9)
        ax_zoom_IPF.step(coll_edges, coll_pow, color='k', lw=lw, zorder=10, label='Collimator losses')
    
    if np.sum(ap_warm_pow) == 0 and np.sum(ap_cold_pow) == 0:
        print("No losses in the aperture; skipping plot.")
    else:
        # Plot warm losses
        ax_zoom_IPF.fill_between(aper_edges[:-1], ap_warm_pow, step='post', color='r', zorder=9)
        ax_zoom_IPF.step(aper_edges[:-1], ap_warm_pow, where='post', color='red', label='Warm losses', linewidth=1)

        # Plot cold losses
        ax_zoom_IPF.fill_between(aper_edges[:-1], ap_cold_pow, step='post', color='b', zorder=9)
        ax_zoom_IPF.step(aper_edges[:-1], ap_cold_pow, where='post', color='blue', label='Cold losses', linewidth=1)
    
    plot_margin = 500
    ax_zoom_IPF.set_xlim(s_min - plot_margin, s_max + plot_margin)

    ax_zoom_IPF.yaxis.grid(visible=True, which='major', zorder=0)
    ax_zoom_IPF.yaxis.grid(visible=True, which='minor', zorder=0)

    # Set the limits for zooming in
    ax_zoom_IPF.set_xlim(x_min, x_max)
    ax_zoom_IPF.set_ylim(0.01, 1e10)

    # Add labels, title, and legend for the zoomed plot
    ax_zoom_IPF.set_xlabel('s[m]')
    ax_zoom_IPF.set_yscale('log', nonpositive='clip')
    #ax_zoom_IPF.set_ylabel('Cleaning inefficiency[$m^{-1}$]')
    ax_zoom_IPF.set_ylabel('Energy Lost [$J$]')
    ax_zoom_IPF.set_title('Zoom at IPF: Lossmap with exciter')
    ax_zoom_IPF.legend(loc='upper right')
    ax_zoom_IPF.grid()

    # Finalize and save the zoomed plot
    plt.savefig(os.path.join(output_dir,f'{output_file}_zoom_IPF.png'))
    plt.close()

    return tot_energy

def plot_lossmap_with_slider(base_dir, output_dir, twiss, turns=None, norm='none'):
    '''
    Function to plot lossmap wrt turns and change the turn interactively with a slider.
    '''
    # Load twiss parameters
    twiss = pd.read_json(twiss, orient='split')
    s_min, s_max = twiss.s.min(), twiss.s.max()

    # Initial data load for setting up the plot
    init_turn = turns[0] if turns else 0
    lossmap_at_0 = f'merged_lossmap_turn_{init_turn}.json'
    coll_edges, coll_loss, coll_pow, aper_edges, ap_warm, ap_cold, ap_warm_pow, ap_cold_pow = prepare_lossmap_values(base_dir, output_dir, s_min, s_max, lossmap_at_0, norm)

    # Create the initial plot
    fig, ax = plt.subplots(figsize=(14, 6))
    plt.subplots_adjust(bottom=0.25)  # Adjust for slider space

    # Set up slider
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor="lightgoldenrodyellow")
    slider = Slider(ax_slider, 'Turn', turns[0], turns[-1], valinit=init_turn, valstep=1)

    lw = 1
    # Define a base alpha value

    if np.sum(coll_pow) > 0:
        fill_collimator = ax.fill_between(coll_edges, coll_pow, step='pre', color='k', alpha=0.9, zorder=9)
        line_collimator = ax.step(coll_edges, coll_pow, color='k', lw=lw, zorder=10, label='Collimator losses')

    if np.sum(ap_warm_pow) > 0:
        fill_ap_warm = ax.fill_between(aper_edges[:-1], ap_warm_pow, step='post', color='r', alpha=0.9, zorder=9)
        line_ap_warm = ax.step(aper_edges[:-1], ap_warm_pow, where='post', color='red', label='Warm losses', linewidth=1)

    if np.sum(ap_cold_pow) > 0:
        fill_ap_cold = ax.fill_between(aper_edges[:-1], ap_cold_pow, step='post', color='b', alpha=0.9, zorder=9)
        line_ap_cold = ax.step(aper_edges[:-1], ap_cold_pow, where='post', color='blue', label='Cold losses', linewidth=1)

    plot_margin = 500
    ax.set_xlim(s_min - plot_margin, s_max + plot_margin)
    ax.set_ylim(0.01, 1e10)
    ax.set_xlabel('s [m]')
    ax.set_yscale('log', nonpositive='clip')
    ax.set_ylabel('Energy Lost [$J$]')
    ax.set_title('Interactive lossmap')
    ax.legend(loc='upper right')
    ax.grid(visible=True)

    # Slider update function
    def update(val):
        turn = int(slider.val)
        lossmap_at_turn = f'merged_lossmap_turn_{turn}.json'
        coll_edges, coll_loss, coll_pow, aper_edges, ap_warm, ap_cold, ap_warm_pow, ap_cold_pow = prepare_lossmap_values(base_dir, output_dir, s_min, s_max, lossmap_at_turn, norm)

        # Clear previous plots
        #ax.clear()
        base_alpha = 0.1  # Starting alpha for plots
        alpha_increment = 0.1  # Amount to increase alpha for each subsequent plot
        # Adjust alpha for increasing transparency
        base_alpha += alpha_increment
        base_alpha = min(base_alpha, 1.0)  # Ensure alpha does not exceed 1

        # Redraw with increasing transparency
        if np.sum(coll_pow) > 0:
            fill_collimator = ax.fill_between(coll_edges, coll_pow, step='pre', color='k', alpha=base_alpha, zorder=9)
            line_collimator = ax.step(coll_edges, coll_pow, color='k', lw=lw, zorder=10, label='Collimator losses')

        if np.sum(ap_warm_pow) > 0:
            fill_ap_warm = ax.fill_between(aper_edges[:-1], ap_warm_pow, step='post', color='r', alpha=base_alpha, zorder=9)
            line_ap_warm = ax.step(aper_edges[:-1], ap_warm_pow, where='post', color='red', label='Warm losses', linewidth=1)

        if np.sum(ap_cold_pow) > 0:
            fill_ap_cold = ax.fill_between(aper_edges[:-1], ap_cold_pow, step='post', color='b', alpha=base_alpha, zorder=9)
            line_ap_cold = ax.step(aper_edges[:-1], ap_cold_pow, where='post', color='blue', label='Cold losses', linewidth=1)


        fig.canvas.draw_idle()

    # Connect the slider to update function
    slider.on_changed(update)
    plt.show()

def collimators_names(coll_dat):
        
    data = {
        "name": [],
        "opening": [],
        "material": [],  # New column for "mat."
        "length": [],
        "angle": [],
        "offset": []
    }

    # Parse the file
    with open(coll_dat, 'r') as file:
        start_parsing = False
        for line in file:
            # Strip whitespace from line
            line = line.strip()
            
            # Start parsing when we reach the relevant section
            if "name" in line:
                start_parsing = True
                continue
            
            # Stop parsing at the end of the collimators section
            if "SETTINGS" in line:
                break
            
            # Parse only relevant lines that contain collimator data
            if start_parsing and line:
                parts = line.split()
                
                # Ensure there are enough parts to avoid index errors
                if len(parts) >= 6:
                    # Append the relevant data
                    data["name"].append(parts[0])
                    data["opening"].append(parts[1])
                    data["material"].append(parts[2])       # New column for material
                    data["length"].append(float(parts[3]))  # Convert to float
                    data["angle"].append(float(parts[4]))   # Convert to float
                    data["offset"].append(float(parts[5]))  # Convert to float

    # Create DataFrame
    df = pd.DataFrame(data)

    return df['name'].values

def collimators_names_json(coll_dat_json):
    # Load the JSON data
    with open(coll_dat_json, 'r') as file:
        data = json.load(file)
    
    # Extract the families and collimators sections
    families = data["families"]
    collimators = data["Collimators"]
    
    # Initialize data structure for DataFrame
    df_data = {
        "name": [],
        "gap": [],
        "stage": [],
        "material": [],
        "length": [],
        "angle": []
    }
    
    # Iterate over collimators and map their attributes from the families section
    for coll_name, coll_info in collimators.items():
        family_name = coll_info["family"]
        
        # Ensure the family exists in the families section
        if family_name in families:
            family_attrs = families[family_name]
            
            # Append the collimator data to the DataFrame structure
            df_data["name"].append(coll_name)
            df_data["gap"].append(family_attrs["gap"])
            df_data["stage"].append(family_attrs["stage"])
            df_data["material"].append(family_attrs["material"])
            df_data["length"].append(float(family_attrs["length"]))  # Convert to float
            df_data["angle"].append(float(family_attrs["angle"]))    # Convert to float
    
    # Create the DataFrame
    df = pd.DataFrame(df_data)
    
    # Return the 'name' column values as a NumPy array
    return df["name"].values

def main(base_dir):

    output_dir = base_dir
    twiss = os.path.join(output_dir,f'twiss_params.json')

    # Uncomment to produce lossmap with slider 
    #plot_lossmap_with_slider(base_dir, output_dir, twiss, turns, norm='coll_turn')

    single_file = 'merged_lossmap_full.json'
    output_file = 'merged_lossmap_full'
    tot_energy_full = plot_lossmaps(output_dir, output_dir, twiss, single_file, output_file, norm='total')

    with open(os.path.join(output_dir,'loss.txt'), "w") as file:
        file.write(f"Loss total: {tot_energy_full}\n")
        
        '''for i in turns:
            single_file = f'merged_lossmap_turn_{i}.json'
            output_file = f'merged_lossmap_turn_{i}'
            tot_energy_turn = plot_lossmaps(output_dir, output_dir, twiss, single_file, output_file, norm='coll_turn',tot_energy_full=tot_energy_full)
            file.write(f"Loss at turn {i}: {tot_energy_turn}\n")'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Track particles with a kicker and plot the results.')
    parser.add_argument('--base_dir', type=str, required=True, help='Path to lossmap files, for ex.: dataset/new_vertical/3_turns/ver_phase_90_3turns.')
    args = parser.parse_args()
    # Call the main function with parsed arguments
    main(args.base_dir)
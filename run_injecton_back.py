import os
import sys
import h5py
import yaml
import copy
import json
import time
import glob
import scipy
import random
import shutil
import argparse
import numpy as np
import xcoll as xc
import xpart as xp
import xtrack as xt
import xfields as xf
import pandas as pd
import xobjects as xo
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from warnings import warn
from copy import deepcopy
from multiprocessing import Pool
from collections import namedtuple
from contextlib import redirect_stdout, redirect_stderr, contextmanager
#from pylhc_submitter.job_submitter import main as htcondor_submit

FCC_ARCS = np.array([
    [12348.6, 21965.2], [23365.2, 32981.8], 
    [35013.8, 44630.4], [46030.4, 55647], 
    [57679,   67295.6], [68695.6, 78312.2], 
    [80344.2, 89960.8], [700,     10316.6]])

# IMPLEMENT PARAMETERD FROM REFERNCE JSON FILE ON GITLAB

#EMITT_X = 6.335825151891213e-05 # Normalized emittance gemitt*gamma*beta 
#EMITT_Y = 1.6955025054356766e-07 

#GEMIT_X = 0.71e-9
#GEMIT_Y = 1.9e-12

ParticleInfo = namedtuple('ParticleInfo', ['name', 'pdgid', 'mass', 'A', 'Z','charge'])

def load_yaml_config(config_file):
    with open(config_file, 'r') as stream:
        config_dict = yaml.safe_load(stream)
    return config_dict

def get_particle_info(particle_name):
    pdg_id = xp.pdg.get_pdg_id_from_name(particle_name)
    charge, A, Z, _ = xp.pdg.get_properties_from_pdg_id(pdg_id)
    mass = xp.pdg.get_mass_from_pdg_id(pdg_id)
    return ParticleInfo(particle_name, pdg_id, mass, A, Z, charge)

def prepare_injection_insertion(line, twiss_df):

    # Insert septa 1 at injection marker in correct lls (PB) hence finj.4 if starting from IPD
    tab = line.get_table()
    septa_location_s = tab['s','finj.4']
    line.insert_element(name="septa.1.b1", element=xt.Marker(), at_s=septa_location_s, s_tol=1e-3)
    print(f"Septa 1 placed at s = {septa_location_s} m")
    # Insert septa 2 just before injection point as a drift to transform into collimator
    line.insert_element(name="septa.2.b1", element=xt.Drift(), at_s=septa_location_s - 40, s_tol=1e-3)
    line.insert_element('septa.b2_aper', element=xt.LimitEllipse(a=0.03, b=0.03), at='septa.2.b1')

    start_llss_2 = tab['s', "start_lss_2"]
    end_llss_2 = tab['s', "end_lss_2"]

    # Find the drift element with the minimum of betax between the center of lss and the septa location 
    possible_locations = twiss_df[
        (twiss_df['s'] > (start_llss_2 + (2032 / 2))) &
        (twiss_df['s'] < (septa_location_s))&
        (twiss_df['name'].str.contains('drift'))
    ]

    betx_min_location = possible_locations[possible_locations['betx'] == possible_locations['betx'].min()]

    if not betx_min_location.empty:
        # Check that dispersion is small
        dispersion_at_kick1 = betx_min_location['dx'].min()

        if np.abs(dispersion_at_kick1) > 0.5:
            raise ValueError("Dispersion is too large for the first kicker. Please choose a different location.")
        print(f'Dispersion at kicker 1 location is: {dispersion_at_kick1}')

        # Select the row(s) with the minimum dispersion
        min_dispersion_location = betx_min_location[
            betx_min_location['dx'] == dispersion_at_kick1
        ]

        # If multiple locations have the same minimum dispersion, choose the one that is closest to start_llss_2
        min_dispersion_location_s = min_dispersion_location.iloc[0]['s']  # Taking the first entry if there are duplicates

        # Insert the first kicker at the location with the minimum dispersion
        # Hardcoding where to place the kickers to have a nice 180 phase adavcance 
        # TO DO: make it lattice indipendendt/ automatized
        line.insert_element(name="injection_kick_1", element=xt.Multipole(knl=[0]), at_s=min_dispersion_location_s-70, s_tol=1e-3)
        line.insert_element('injection_kick_1_aper', element=xt.LimitEllipse(a=0.03, b=0.03), at='injection_kick_1')
        print(f"First kicker placed at s = {min_dispersion_location_s-70} m (min dispersion)")
    else:
        print("No suitable location found for the first kicker with minimal dispersion.")
    
    # Hardcoding where to place the kickers to have a nice 180 phase adavcance 
    # TO DO: make it lattice indipendendt/ automatized
    line.insert_element(name="injection_kick_2", element=xt.Multipole(knl=[0]), at_s=end_llss_2-110, s_tol=1e-3)
    line.insert_element('injection_kick_2_aper', element=xt.LimitEllipse(a=0.03, b=0.03), at='injection_kick_2')
    print(f"Second kicker placed at s = {end_llss_2-110} m")

    line.build_tracker()

    # Calculate the Twiss parameters again to check the phase advace
    twiss = line.twiss()
    phase_diff = (twiss['mux', 'injection_kick_1'] - twiss['mux', 'injection_kick_2'])*2*180
    print(f'Phase advance between kick_1 and kick_2:{phase_diff}')

    # CHECK IF IT HAS TO BE TURNED ON FOR CIRC BEAM
    #line.insert_element(name="injection_septa_kick", element=xt.Multipole(knl=[0]), at='septa.1.b1')
    #line.insert_element(name="injection_septa2_kick", element=xt.Multipole(knl=[0]), at='septa.2.b1')

    # check if really needed
    #line.insert_element(name="injection_dirft", element=xt.Marker(), at='septa.1.b1')
 

def prepare_injected_beam(twiss, line, ref_particle, injection_config, num_particles, capacity):
    
    start_element = injection_config['start_element']#+'..0'
    print(f'Preparing beam coming from booster to be injected at {start_element}')

    line.build_tracker()

    inj_seed_x = injection_config['seed_x']
    inj_seed_y = injection_config['seed_y']
    inj_seed_z = injection_config['seed_z']
    ## define energy offset of injected 
    energy_offset_injection = injection_config['energy_offset']
    ## define horizontal offset of injected 
    x_offset_injection = injection_config['x_offset'] # #m
    ## define energy spread of beam extracted from booster
    energy_spread_injection = injection_config['energy_spread']
    ## define bunch leangth of beam extracted from booster
    bunch_length_injection = injection_config['bunch_length'] #m
    ## define injected emittances (keep the ration of the collider reference paramiters)
    emittance_x_injection = injection_config['emittance_x_injection']#m
    emittance_y_injection = injection_config['emittance_y_injection'] #m #emittance_x_injection*emittance_y/emittance_x #m
    
    beta_relativistic = ref_particle.beta0
    gamma_relativistic = ref_particle.gamma0
    
    normalized_emittance_x_injection = emittance_x_injection*gamma_relativistic*beta_relativistic #m
    normalized_emittance_y_injection = emittance_y_injection*gamma_relativistic*beta_relativistic #m

    # Horizontal and vertical plane: generate gaussian distribution in normalized coordinates
    np.random.seed(inj_seed_x)
    x_in_sigmas, px_in_sigmas = xp.generate_2D_gaussian(num_particles)
    np.random.seed(inj_seed_y)
    y_in_sigmas, py_in_sigmas = xp.generate_2D_gaussian(num_particles)
    # Longitudinal plane: generate gaussian distribution 
    np.random.seed(inj_seed_z)
    zeta_in_sigmas, delta_in_sigmas = xp.generate_2D_gaussian(num_particles)
    zeta = zeta_in_sigmas*bunch_length_injection
    delta = delta_in_sigmas*energy_spread_injection+energy_offset_injection

    ## Cycle the line so to start from the injection point qi6.1..16 for zero alfx and dpx

    line.cycle(name_first_element=start_element, inplace=True)

    # The longitudinal closed orbit needs to be manually supplied for now

    element_index = line.element_names.index(start_element)
    if isinstance(twiss, pd.DataFrame):
        zeta_co = twiss['zeta'].iloc[element_index]
        delta_co = twiss['delta'].iloc[element_index]
    else:    
        zeta_co = twiss.zeta[element_index] 
        delta_co = twiss.delta[element_index] 

    # Build particles:
    part = line.build_particles(
        _capacity=capacity,
        zeta=zeta+zeta_co, delta=delta+delta_co,
        x_norm=x_in_sigmas, px_norm=px_in_sigmas,
        y_norm=y_in_sigmas, py_norm=py_in_sigmas,
        nemitt_x=normalized_emittance_x_injection, nemitt_y=normalized_emittance_y_injection,  
        at_element=start_element
        )

    part.x = part.x + x_offset_injection
    #part.start_tracking_at_element = -1
    #line.cycle(name_first_element='qi3.4..5', inplace=True)
    
    return part

def prepare_matched_beam(twiss, line, ref_particle, dist_config, nemitt, num_particles, capacity):
    
    start_element = dist_config['start_element'] #+'..0'
    print(f'Preparing a matched Gaussian beam at {start_element}')
    sigma_z = dist_config['sigma_z']

    line.build_tracker()
    
    x_norm, px_norm = xp.generate_2D_gaussian(num_particles)
    y_norm, py_norm = xp.generate_2D_gaussian(num_particles)
    
    line.cycle(name_first_element=start_element, inplace=True)
    
    # The longitudinal closed orbit needs to be manually supplied for now
    element_index = line.element_names.index(start_element)

    if isinstance(twiss, pd.DataFrame):
        zeta_co = twiss['zeta'].iloc[element_index]
        delta_co = twiss['delta'].iloc[element_index]
    else:    
        zeta_co = twiss.zeta[element_index] 
        delta_co = twiss.delta[element_index] 

    assert sigma_z >= 0
    zeta = delta = 0
    if sigma_z > 0:
        print(f'Paramter sigma_z > 0, preparing a longitudinal distribution matched to the RF bucket')
        zeta, delta = xp.generate_longitudinal_coordinates(
                        line=line,
                        num_particles=num_particles, distribution='gaussian',
                        sigma_z=sigma_z, particle_ref=ref_particle)

    

    part = line.build_particles(
        _capacity=capacity,
        particle_ref=ref_particle,
        x_norm=x_norm, px_norm=px_norm,
        y_norm=y_norm, py_norm=py_norm,
        zeta=zeta + zeta_co,
        delta=delta + delta_co,
        nemitt_x=nemitt[0],
        nemitt_y=nemitt[1],
        at_element=start_element,
        )
    
    #part.start_tracking_at_element = -1

    return part

def install_collimators(line, input_config, nemitt):

    line.discard_tracker()
    
    colldb = xc.CollimatorDatabase.from_json(input_config['collimator_file'],
                                                nemitt_x=nemitt[0],
                                                nemitt_y=nemitt[1])
    
    colldb.install_geant4_collimators(verbose=True,
                                      line=line)
                                      #bdsim_config_file=input_config['bdsim_config'],
                                      #relative_energy_cut= 0.165,
                                      #random_seed=lossmap_config['seed'])
    twiss = line.twiss(method='6d')

    line.build_tracker()
    line.collimators.assign_optics(nemitt_x=nemitt[0], nemitt_y=nemitt[1], twiss=twiss)
    #xc.assign_optics_to_collimators(line=line, twiss=twiss)
    line.discard_tracker()

    context = xo.ContextCpu()  # Only support CPU for Geant4 coupling TODO: maybe not needed anymore?
    # Transfer lattice on context and compile tracking code
    global_aper_limit = 1e3  # Make this large to ensure particles lost on aperture markers

    # compile the track kernel once and set it as the default kernel. TODO: a bit clunky, find a more elegant approach

    line.build_tracker(_context=context)

    tracker_opts=dict(track_kernel=line.tracker.track_kernel,
                        _buffer=line.tracker._buffer,
                        _context=line.tracker._context,
                        io_buffer=line.tracker.io_buffer)

    line.discard_tracker() 
    line.build_tracker(**tracker_opts)
    line.config.global_xy_limit=global_aper_limit

    return twiss

def generate_lossmap(line, num_turns, particles, ref_part, input_config, monitor_names, lossmap_config, start_element, theta_1=0, theta_2=0, output_dir="plots", impact=False):

    if impact:
        impacts = xc.InteractionRecord.start(line=line)

    xc.Geant4Engine.start(line=line,
                            particle_ref=ref_part,  
                            seed=lossmap_config['seed'],
                            bdsim_config_file=input_config['bdsim_config'],
                            relative_energy_cut= 0.165,
                            batch_mode=True)
    #batch_mode = False
    line.scattering.enable()

    for turn in range(num_turns):
        if turn == 10:
            line['injection_kick_1'].knl = theta_1
            line['injection_kick_2'].knl = theta_2 #3.457728093870607e-5
            print(f'Start turn {turn}, Survivng particles: {particles._num_active_particles}')
            line.track(particles, num_turns=1)
        else:
            line['injection_kick_1'].knl = 0
            line['injection_kick_2'].knl = 0
            line.track(particles, num_turns=1)

        if particles._num_active_particles == 0:
            print(f'All particles lost by turn {turn}, teminating.')
            break

    # Track (saving turn-by-turn data)
    '''for turn in range(num_turns):
        print(f'Start turn {turn}, Survivng particles: {particles._num_active_particles}')
        line.track(particles, num_turns=1)
        #start_element = 'qi5.4..7'
        #line.track(particles, num_turns=1, ele_start=start_element, ele_stop=start_element)          
            
        if particles._num_active_particles == 0:
            print(f'All particles lost by turn {turn}, teminating.')
            break'''
    
    for name in monitor_names:
        last_track = line.get(name)
        save_track_to_h5(last_track, name , output_dir)
    
    line.scattering.disable()

    xc.Geant4Engine.stop()
    #print(f'Tracking {num_particles} turns done in: {time.time()-t0} s')

    if impact:
        impacts.stop()
        # Saving impacts table
        df = impacts.to_pandas()
        df.to_csv(os.path.join(output_dir,'impacts_line.csv'), index=False)
    
    aper_interp = lossmap_config['aperture_interp']
    # Make loss map
    weights = lossmap_config.get('weights', 'none')

    line.cycle(name_first_element='l000017$start', inplace=True)

    df_line = line.to_pandas()
    beam_start_index= df_line[df_line['name'] == start_element].index[0]
    max_index = df_line.index.max()
    beam_start_s = df_line[df_line['name'] == start_element]['s'].values[0]
    s_max = df_line['s'].max()

    particles = particles.remove_unused_space()  

    particles.s = (particles.s + beam_start_s) % s_max 
    particles.at_element = ((particles.at_element + beam_start_index) % max_index).astype(int)

    file_path = os.path.join(output_dir, f'merged_lossmap_full.json')

    if weights == 'energy':
        part_mass_ratio = particles.charge_ratio / particles.chi
        part_mass = part_mass_ratio * particles.mass0
        p0c = particles.p0c
        f_x = lambda x: 1  # No modification for x
        f_px = lambda px: 1  # No modification for px
        f_y = lambda y: 1  # No modification for y
        f_py = lambda py: 1  # No modification for py
        f_zeta = lambda zeta: 1 
        f_delta = lambda delta: np.sqrt(((delta + 1) * p0c * part_mass_ratio)**2 + part_mass**2)
        weight_function = [f_x, f_px, f_y, f_py, f_zeta, f_delta]


    LossMap_full = xc.LossMap(line,
                            part=particles,
                            line_is_reversed=False,
                            interpolation=aper_interp,
                            weights=None,
                            weight_function=weight_function
                            )
    LossMap_full.to_json(file_path)
    print(f'Lossmap for all turns saved to {file_path}')

    output_file = os.path.join(output_dir, f"part.hdf")
    _save_particles_hdf(
        LossMap_full.part, lossmap_data=None, filename=output_file)

def insert_monitors(num_turns, rad_line, num_particles, config_dict):
    """
    Inserts monitors at specific locations in the line:
    - At the syncrhtron mask position tcr.h.c3.2.b1
    - At the primary collimator tcp.h.b1
    - At the experimental insertion IPG.
    - At the 'injection' loation. 
    """
    # Define monitor, for now just one particle
    monitor = xt.ParticlesMonitor(start_at_turn=0, stop_at_turn =num_turns, num_particles = num_particles, auto_to_numpy=True)
    tab = rad_line.get_table()
    #df_line = rad_line.to_pandas()

    monitors = [monitor.copy() for _ in range(4)]
    monitor_names = ['monitor_prim_coll','monitor_inject'] #, 'monitor_start_beam'] # 'monitor_at_ipg', 'monitor_mask', ,'monitor_s_0']

    rad_line.discard_tracker()

    s_prim_coll = tab['s','tcp.h.b1']#df_line.loc[df_line['name'] == 'tcp.h.b1', 's'].values[0]
    s_inj = tab['s', config_dict['injection']['start_element']] #df_line.loc[df_line['name'] == inj_element, 's'].values[0]
    
   
    # Insert monitor at the primary collimator position, also with a small offset
    rad_line.insert_element(monitor_names[0], monitors[0], at_s=s_prim_coll + 0.5)

    # Insert monitor at the beam starting point, adjusting for a small offset
    rad_line.insert_element(monitor_names[1], monitors[1], at_s=s_inj+ 0.01)

    if config_dict['beam']['type'] == 'circulating':
        s_kick1 = tab['s', 'injection_kick_1']
        s_kick2 = tab['s', 'injection_kick_2']
        monitor_names.extend(['monitor_kick1', 'monitor_kick2'])
        # Insert monitor at the beam starting point, adjusting for a small offset
        rad_line.insert_element(monitor_names[2], monitors[2], at_s=s_kick1+ 0.01)
        
        # Insert monitor at the beam starting point, adjusting for a small offset
        rad_line.insert_element(monitor_names[3], monitors[3], at_s=s_kick2+ 0.01)

    # Ideal for mismatch
    #s_beam = df_line.loc[df_line['name'] == dist_config['start_element'], 's'].values[0]
    #rad_line.insert_element(monitor_names[2], monitors[2], at_s=s_beam+0.01)

    return monitor_names 

def configure_tracker_radiation(line, radiation_model, beamstrahlung_model=None, bhabha_model=None, for_optics=False):
    mode_print = 'optics' if for_optics else 'tracking'

    print_message = f"Tracker synchrotron radiation mode for '{mode_print}' is '{radiation_model}'"

    _beamstrahlung_model = None if beamstrahlung_model == 'off' else beamstrahlung_model
    _bhabha_model = None if bhabha_model == 'off' else bhabha_model

    if radiation_model == 'mean':
        if for_optics:
            # Ignore beamstrahlung and bhabha for optics
            line.configure_radiation(model=radiation_model)
        else:
            line.configure_radiation(model=radiation_model, 
                                     model_beamstrahlung=_beamstrahlung_model,
                                     model_bhabha=_bhabha_model)
        # The matrix stability tolerance needs to be relaxed for radiation and tapering
        # TODO: check if this is still needed
        line.matrix_stability_tol = 0.5

    elif radiation_model == 'quantum':
        if for_optics:
            print_message = ("Cannot perform optics calculations with radiation='quantum',"
            " reverting to radiation='mean' for optics.")
            line.configure_radiation(model='mean')
        else:
            line.configure_radiation(model='quantum',
                                     model_beamstrahlung=_beamstrahlung_model,
                                     model_bhabha=_bhabha_model)
        line.matrix_stability_tol = 0.5

    elif radiation_model == 'off':
        pass
    else:
        raise ValueError('Unsupported radiation model: {}'.format(radiation_model))
    print(print_message)

def find_apertures(line):
    i_apertures = []
    apertures = []
    for ii, ee in enumerate(line.elements):
        if ee.__class__.__name__.startswith('Limit'):
            i_apertures.append(ii)
            apertures.append(ee)
    return np.array(i_apertures), np.array(apertures)

def find_bb_lenses(line):
    i_apertures = []
    apertures = []
    for ii, ee in enumerate(line.elements):
        if ee.__class__.__name__.startswith('BeamBeamBiGaussian3D'):
            i_apertures.append(ii)
            apertures.append(ee)
    return np.array(i_apertures), np.array(apertures)

def insert_bb_lens_bounding_apertures(line):
    # Place aperture defintions around all beam-beam elements in order to ensure
    # the correct functioning of the aperture loss interpolation
    # the aperture definitions are taken from the nearest neighbour aperture in the line
    s_pos = line.get_s_elements(mode='upstream')
    apert_idx, apertures = find_apertures(line)
    apert_s = np.take(s_pos, apert_idx)

    bblens_idx, bblenses = find_bb_lenses(line)
    bblens_names = np.take(line.element_names, bblens_idx)
    bblens_s_start = np.take(s_pos, bblens_idx)
    bblens_s_end = np.take(s_pos, bblens_idx + 1)

    # Find the nearest neighbour aperture in the line
    bblens_apert_idx_start = np.searchsorted(apert_s, bblens_s_start, side='left')
    bblens_apert_idx_end = bblens_apert_idx_start + 1

    aper_start = apertures[bblens_apert_idx_start]
    aper_end = apertures[bblens_apert_idx_end]

    idx_offset = 0
    for ii in range(len(bblenses)):
        line.insert_element(name=bblens_names[ii] + '_aper_start',
                            element=aper_start[ii].copy(),
                            at=bblens_idx[ii] + idx_offset)
        idx_offset += 1

        line.insert_element(name=bblens_names[ii] + '_aper_end',
                            element=aper_end[ii].copy(),
                            at=bblens_idx[ii] + 1 + idx_offset)
        idx_offset += 1

def _make_bb_lens(nb, phi, sigma_z, alpha, n_slices, other_beam_q0,
                  sigma_x, sigma_px, sigma_y, sigma_py, beamstrahlung_on=False):
       
    slicer = xf.TempSlicer(n_slices=n_slices, sigma_z=sigma_z, mode="shatilov")

    el_beambeam = xf.BeamBeamBiGaussian3D(
            #_context=context,
            config_for_update = None,
            other_beam_q0=other_beam_q0,
            phi=phi, # half-crossing angle in radians
            alpha=alpha, # crossing plane
            # decide between round or elliptical kick formula
            min_sigma_diff = 1e-28,
            # slice intensity [num. real particles] n_slices inferred from length of this
            slices_other_beam_num_particles = slicer.bin_weights * nb,
            # unboosted strong beam moments
            slices_other_beam_zeta_center = slicer.bin_centers,
            slices_other_beam_Sigma_11    = n_slices*[sigma_x**2], # Beam sizes for the other beam, assuming the same is approximation
            slices_other_beam_Sigma_22    = n_slices*[sigma_px**2],
            slices_other_beam_Sigma_33    = n_slices*[sigma_y**2],
            slices_other_beam_Sigma_44    = n_slices*[sigma_py**2],
            # only if BS on
            slices_other_beam_zeta_bin_width_star_beamstrahlung = None if not beamstrahlung_on else slicer.bin_widths_beamstrahlung / np.cos(phi),  # boosted dz
            # has to be set
            slices_other_beam_Sigma_12    = n_slices*[0],
            slices_other_beam_Sigma_34    = n_slices*[0],
            compt_x_min                   = 1e-4,
        )
    el_beambeam.iscollective = True # Disable in twiss

    return el_beambeam

def _insert_beambeam_elements(line, config_dict, twiss_table, nemitt):

    beamstrahlung_mode = config_dict['run'].get('beamstrahlung', 'off')
    #bhabha_mode = config_dict['run'].get('bhabha', 'off')
    # This is needed to set parameters of the beam-beam lenses
    beamstrahlung_on = beamstrahlung_mode != 'off'

    beambeam_block = config_dict.get('beambeam', None)
    if beambeam_block is not None:

        beambeam_list = beambeam_block
        if not isinstance(beambeam_list, list):
            beambeam_list = [beambeam_list, ]

        print('Beam-beam definitions found, installing beam-beam elements at: {}'
              .format(', '.join([dd['at_element'] for dd in beambeam_list])))
            
        for bb_def in beambeam_list:
            element_name = bb_def['at_element']
            # the beam-beam lenses are thin and have no effects on optics so no need to re-compute twiss
            element_twiss_index = list(twiss_table.name).index(element_name)
            # get the line index every time as it changes when elements are installed
            element_line_index = line.element_names.index(element_name)
            #element_spos = twiss_table.s[element_twiss_index]
            
            sigmas = twiss_table.get_betatron_sigmas(*nemitt if hasattr(nemitt, '__iter__') else (nemitt, nemitt))

            bb_elem = _make_bb_lens(nb=float(bb_def['bunch_intensity']), 
                                    phi=float(bb_def['crossing_angle']), 
                                    sigma_z=float(bb_def['sigma_z']),
                                    n_slices=int(bb_def['n_slices']),
                                    other_beam_q0=int(bb_def['other_beam_q0']),
                                    alpha=0, # Put it to zero, it is okay for this use case
                                    sigma_x=np.sqrt(sigmas['Sigma11'][element_twiss_index]), 
                                    sigma_px=np.sqrt(sigmas['Sigma22'][element_twiss_index]), 
                                    sigma_y=np.sqrt(sigmas['Sigma33'][element_twiss_index]), 
                                    sigma_py=np.sqrt(sigmas['Sigma44'][element_twiss_index]), 
                                    beamstrahlung_on=beamstrahlung_on)
            
            line.insert_element(index=element_line_index, 
                                element=bb_elem,
                                name=f'beambeam_{element_name}')
        
        insert_bb_lens_bounding_apertures(line)

def save_track_to_h5(monitor, monitor_name='0', output_dir="plots"):
    """Saves multi-particle tracking data to an HDF5 file with gzip compression."""

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f'merged_data_{monitor_name}.h5')

    # Define attributes and expected NumPy types
    attributes = {
        'x': np.float32, 'px': np.float32, 'y': np.float32, 'py': np.float32,
        'zeta': np.float32, 'delta': np.float32, 's': np.float32, 'at_turn': np.int32
    }

    try:
        with h5py.File(file_path, 'w') as h5f:
            for attr, dtype in attributes.items():
                data = getattr(monitor, attr, None)
                
                if data is not None:
                    data = np.asarray(data, dtype=dtype)  # Ensure NumPy array
                    
                    # Handle empty arrays gracefully
                    if data.size > 0:
                        h5f.create_dataset(attr, data=data, compression='gzip')

        print(f"Data successfully saved to {file_path}")

    except Exception as e:
        raise RuntimeError(f"Failed to save data to {file_path}: {e}")

def save_track_along_ring_to_h5(monitor, output_dir="plots"):
    """
    Saves monitor data (assumed shape: (num_part, 201358)) to an HDF5 file.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, f'merged_data_along_ring.h5')

    try:
        with h5py.File(file_path, 'w') as h5f:
            # Extract and store relevant properties
            for attr in ['x', 'px', 'y', 'py', 'zeta', 'delta', 's', 'at_turn']:
                data = getattr(monitor, attr, None)
                if data is not None:
                    dtype = np.float32 if data.dtype.kind in 'fc' else np.int32
                    h5f.create_dataset(attr, data=data.astype(dtype), compression="gzip", chunks=True)

        print(f"Data successfully saved to {file_path}")
    except Exception as e:
        print(f"Failed to save data to {file_path}: {e}")

def save_twiss_to_json(twiss, file_path):
    """Save Twiss parameters to JSON after converting to a serializable format."""

    df_twiss= twiss.to_pandas()
    # Save DataFrame to JSON format with orientation 'split' for better structure
    df_twiss.to_json(file_path, orient='split', indent=4)

    print(f"Twiss parameters saved to {file_path}.")

def load_twiss_from_json(file_path):
    """
    Load Twiss parameters from a JSON file and return them as a Pandas DataFrame.
    """
    # Load DataFrame from JSON format with orientation 'split'
    df_twiss = pd.read_json(file_path, orient='split')

    print(f"Twiss parameters loaded from {file_path}.")

    return df_twiss
 
def initialize_optics_and_calculate_twiss(xtrack_line, config_dict, nemitt, twiss_file_path):
    """
    Initialize the optics and calculate Twiss parameters.
    """
    rad_line = xt.Line.from_json(xtrack_line)
    tab = rad_line.get_table()
    # Place markers at the beggining and end of straight section fo injection PB 
    rad_line.insert_element(name="start_lss_2", element=xt.Marker(), at_s=tab['s', "ip.7"] + 10286.855722259006, s_tol=1e-3)
    tab = rad_line.get_table()
    start_llss_2 = tab['s', "start_lss_2"]
    end_llss_2 = start_llss_2+2032
    rad_line.insert_element(name="end_lss_2", element=xt.Marker(), at_s=end_llss_2 ,s_tol=1e-3)
    rad_line.build_tracker()
    configure_tracker_radiation(rad_line, config_dict['run']['radiation'], beamstrahlung_model=None, bhabha_model=None, for_optics=True)

    #LINE IS ALREADY TAPERED
    '''rad_line.configure_radiation(model='mean')
    rad_line.compensate_radiation_energy_loss() 
    tw = rad_line.twiss(eneloss_and_damping=True)

    # Compensate for SR + tapering
    delta0 = tw.delta[0] - np.mean(tw.delta)
    rad_line.compensate_radiation_energy_loss(delta0=delta0)'''

    # Re-calculate Twiss parameters 
    twiss_rad = rad_line.twiss(method='6d', eneloss_and_damping=True)
    #df_twiss_rad = twiss_rad.to_pandas()  
    #save_twiss_to_json(twiss_rad, twiss_file_path)  
    print(f"Twiss parameters after SR compensation saved to {twiss_file_path}.")

    rad_line.discard_tracker()

    _insert_beambeam_elements(rad_line, config_dict, twiss_rad, nemitt)

    return rad_line, twiss_rad

def load_output(directory, output_file, match_pattern='*part.hdf*',
                imax=None, load_lossmap=False, load_particles=False):

    t0 = time.time()

    job_dirs = glob.glob(os.path.join(directory, 'Job.*')
                         )  # find directories to loop over

    job_dirs_sorted = []
    for i in range(len(job_dirs)):
        # Very inefficient, but it sorts the directories by their numerical index
        job_dir_idx = job_dirs.index(
            os.path.join(directory, 'Job.{}'.format(i)))
        job_dirs_sorted.append(job_dirs[job_dir_idx])

    part_hdf_files = []
    part_dataframes = []
    lossmap_dicts = []
    
    tqdm_ncols=100
    tqdm_miniters=10
    print(f'Parsing directories...')
    dirs_visited = 0
    files_loaded = 0
    for i, d in tqdm(enumerate(job_dirs_sorted), total=len(job_dirs_sorted), 
                     ncols=tqdm_ncols, miniters=tqdm_miniters):
        if imax is not None and i > imax:
            break

        #print(f'Processing {d}')
        dirs_visited += 1
        output_dir = os.path.join(d, 'plots')
        output_files = glob.glob(os.path.join(output_dir, match_pattern))
        if output_files:
            of = output_files[0]
            part_hdf_files.append(of)
            files_loaded += 1
        else:
            print(f'No output found in {d}')

    part_merged = None
    if load_particles:
        print(f'Loading particles...')
        with Pool() as p:
            part_dataframes = list(tqdm(p.imap(_read_particles_hdf, part_hdf_files), total=len(part_hdf_files), 
                                        ncols=tqdm_ncols, miniters=tqdm_miniters))
        part_objects = [xp.Particles.from_pandas(pdf) for pdf in tqdm(part_dataframes, total=len(part_dataframes),
                                                                      ncols=tqdm_ncols, miniters=tqdm_miniters)]

        print('Particles load finished, merging...')
        part_merged = xp.Particles.merge(list(tqdm(part_objects, total=len(part_objects),
                                              ncols=tqdm_ncols, miniters=tqdm_miniters)))

    # Load the loss maps
    lmd_merged = None
    if load_lossmap:
        print(f'Loading loss map data...')
        with Pool() as p:
            lossmap_dicts = list(tqdm(p.imap(_load_lossmap_hdf, part_hdf_files), total=len(part_hdf_files), 
                                      ncols=tqdm_ncols, miniters=tqdm_miniters))

        print('Loss map load finished, merging..')

        num_tol = 1e-9
        lmd_merged = lossmap_dicts[0]
        for lmd in tqdm(lossmap_dicts[1:], ncols=tqdm_ncols, miniters=tqdm_miniters):
            # Scalar parameters
            # Ensure consistency
            identical_params = ('s_min', 's_max', 'binwidth', 'nbins')
            identical_strings = ('weights',)
            for vv in identical_params:
                assert np.isclose(lmd_merged['lossmap_scalar'][vv],
                                  lmd['lossmap_scalar'][vv],
                                  num_tol)
            for vv in identical_strings:
                assert np.all(lmd_merged['lossmap_scalar'][vv] == lmd['lossmap_scalar'][vv])

            lmd_merged['lossmap_scalar']['n_primaries'] += lmd['lossmap_scalar']['n_primaries']

            # Collimator losses
            # These cannot be empty dataframes even if there is no losses
            assert np.allclose(lmd_merged['lossmap_coll']['coll_start'],
                               lmd['lossmap_coll']['coll_start'],
                               atol=num_tol)

            assert np.allclose(lmd_merged['lossmap_coll']['coll_end'],
                               lmd['lossmap_coll']['coll_end'],
                               atol=num_tol)
            
            assert np.array_equal(lmd_merged['lossmap_coll']['coll_element_index'],
                                  lmd['lossmap_coll']['coll_element_index'])
            
            assert np.array_equal(lmd_merged['lossmap_coll']['coll_name'],
                                  lmd['lossmap_coll']['coll_name'])

            lmd_merged['lossmap_coll']['coll_loss'] += lmd['lossmap_coll']['coll_loss']

            # Aperture losses
            alm = lmd_merged['lossmap_aper']
            al = lmd['lossmap_aper']

            # If the aperture loss dataframe is empty, it is not stored on HDF
            if al is not None:
                if alm is None:
                    lmd_merged['lossmap_aper'] = al
                else:
                    lm = alm.aper_loss.add(al.aper_loss, fill_value=0)
                    lmd_merged['lossmap_aper'] = pd.DataFrame(
                        {'aper_loss': lm})

    _save_particles_hdf(particles=part_merged,
                        lossmap_data=lmd_merged, filename=output_file)

    print('Directories visited: {}, files loaded: {}'.format(
        dirs_visited, files_loaded))
    print(f'Processing done in {time.time() -t0} s')

def _read_particles_hdf(filename):
    return pd.read_hdf(filename, key='particles')

def _save_particles_hdf(particles=None, lossmap_data=None, filename='part', reduce_particles_size=False):
    if not filename.endswith('.hdf'):
        filename += '.hdf'

    fpath = Path(filename)
    # Remove a potential old file as the file is open in append mode
    if fpath.exists():
        fpath.unlink()

    if particles is not None:
        df = particles.to_pandas(compact=True)
        if reduce_particles_size:
            for dtype in ('float64', 'int64'):
                thistype_columns = df.select_dtypes(include=[dtype]).columns
                df[thistype_columns] = df[thistype_columns].astype(dtype.replace('64', '32'))

        df.to_hdf(fpath, key='particles', format='table', mode='a',
                  complevel=9, complib='blosc')

    if lossmap_data is not None:
        for key, lm_df in lossmap_data.items():
            lm_df.to_hdf(fpath, key=key, mode='a', format='table',
                         complevel=9, complib='blosc')
            
@contextmanager
def set_directory(path: Path):
    """
    Taken from: https://dev.to/teckert/changing-directory-with-a-python-context-manager-2bj8
    """
    origin = Path().absolute()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)

def resolve_and_cache_paths(iterable_obj, resolved_iterable_obj, cache_destination):
    if isinstance(iterable_obj, (dict, list)):
        for k, v in (iterable_obj.items() if isinstance(iterable_obj, dict) else enumerate(iterable_obj)):
            possible_path = Path(str(v))
            if not isinstance(v, (dict, list)) and possible_path.exists() and possible_path.is_file():
                shutil.copy(possible_path, cache_destination)
                resolved_iterable_obj[k] = possible_path.name
            resolve_and_cache_paths(v, resolved_iterable_obj[k], cache_destination)


def dump_dict_to_yaml(dict_obj, file_path):
        with open(file_path, 'w') as yaml_file:
            yaml.dump(dict_obj, yaml_file, 
                      default_flow_style=False, sort_keys=False)


def submit_jobs(config_dict, config_file):
    # Relative path from the config file should be relative to
    # the file itself, not to where the script is executed from
    if config_file:
        conf_path = Path(config_file).resolve()
        conf_dir = conf_path.parent
        conf_fname = conf_path.name
    else:
        conf_dir = Path().resolve()
        conf_fname = 'config_collimation.yaml'
        conf_path = Path(conf_dir, conf_fname)
        
    with set_directory(conf_dir):
        #config_dict = CONF_SCHEMA.validate(config_dict)

        sub_dict = config_dict['jobsubmission']
        workdir = Path(sub_dict['working_directory']).resolve()
        num_jobs = sub_dict['num_jobs']
        replace_dict_in = sub_dict.get('replace_dict', {})
        executable = sub_dict.get('executable', 'bash')
        mask_abspath = Path(sub_dict['mask']).resolve()
        
        max_local_jobs = 10
        if sub_dict.get('run_local', False) and num_jobs > max_local_jobs:
            raise Exception(f'Cannot run more than {max_local_jobs} jobs locally,'
                            f' {num_jobs} requested.')
            
        # Make a directory to copy the files for the submission
        input_cache = Path(workdir, 'input_cache')
        os.makedirs(workdir)
        os.makedirs(input_cache)

        # Copy the files to the cache and replace the path in the config
        # Copy the configuration file
        if conf_path.exists():
            shutil.copy(conf_path, input_cache)
        else:
            # If the setup came from a dict a dictionary still dump it to archive
            dump_dict_to_yaml(config_dict, Path(input_cache, conf_path.name))
            
        exclude_keys = {'jobsubmission',} # The submission block is not needed for running
        # Preserve the key order
        reduced_config_dict = {k: config_dict[k] for k in 
                               config_dict.keys() if k not in exclude_keys}
        resolved_config_dict = copy.deepcopy(reduced_config_dict)
        resolve_and_cache_paths(reduced_config_dict, resolved_config_dict, input_cache)

        resolved_conf_file = f'for_jobs_{conf_fname}' # config file used to run each job
        dump_dict_to_yaml(resolved_config_dict, Path(input_cache, resolved_conf_file))

        # compress the input cache to reduce network traffic
        shutil.make_archive(input_cache, 'gztar', input_cache)
        # for fpath in input_cache.iterdir():
        #     fpath.unlink()
        # input_cache.rmdir()

        # Set up the jobs
        seeds = np.arange(num_jobs) + 1 # Start the seeds at 1
        replace_dict_base = {'seed': seeds.tolist(),
                             'config_file': resolved_conf_file,
                             'input_cache_archive': str(input_cache) + '.tar.gz'}

        # Pass through additional replace dict option and other job_submitter flags
        if replace_dict_in:
            replace_dict = {**replace_dict_base, **replace_dict_in}
        else:
            replace_dict = replace_dict_base
        
        processed_opts = {'working_directory', 'num_jobs','executable', 'mask'}
        submitter_opts = list(set(sub_dict.keys()) - processed_opts)
        submitter_options_dict = { op: sub_dict[op] for op in submitter_opts }
        
        # Send/run the jobs via the job_submitter interface
        htcondor_submit(
            mask=mask_abspath,
            working_directory=workdir,
            executable=executable,
            replace_dict=replace_dict,
            **submitter_options_dict)

        print('Done!')


def main(config_file, submit, merge):
    """
    Main function to run the tracking and plotting with the kicker added to the optics line.
    """
    start_time = time.time()
    config_dict = load_yaml_config(config_file)

    input_config = config_dict['input']
    beam_config = config_dict['beam']
    run_config = config_dict['run']
    lossmap_config = config_dict['lossmap']
    injection_config = config_dict['injection']
    dist_config =config_dict['dist']
    #job_config = config_dict['jobsubmission']

    output_dir = run_config['output_dir']

    if submit:

        submit_jobs(config_dict, config_file)

    elif merge:
        working_directory = config_dict['jobsubmission']['working_directory']
        match_pattern = '*part.hdf*'
        output_file = os.path.join(working_directory, 'part_merged.hdf')
        load_output(working_directory, output_file, match_pattern=match_pattern, load_particles=True)
    
    else:
        os.makedirs(output_dir, exist_ok=True)

        # DEFINE REFERENCE PARAMETERES FROM FILE 

        with open(input_config['reference_parameters'], 'r') as file:
            REF_PAR = json.load(file)

        GEMIT_X = 0.71e-9 # REF_PAR["z"]["EMITTANCE_X"]
        GEMIT_Y = 2.1e-12 # REF_PAR["z"]["EMITTANCE_Y"]

        # Define paths for to JSON files for LINE, TWISS and LOSSMAP
        SR_coll_line = os.path.join(output_dir, 'SR_coll_line.json')
        twiss_file_path = os.path.join(output_dir, f'twiss_params.json')
        lossmap_json = os.path.join(output_dir, 'merged_lossmap_full.json')

        # Parameters
        bin_w = lossmap_config['aperture_binwidth']
        num_particles = run_config['nparticles']
        capacity = run_config['max_particles'] 
        particle_name = beam_config['particle']
        num_turns = run_config['turns']

        # Get particle information
        particle_info = get_particle_info(particle_name)
        ref_part = xt.Particles(p0c=45.6e9, mass0=particle_info.mass, q0=particle_info.charge, pdg_id=particle_info.pdgid)
        
        beta0 = ref_part.beta0
        gamma0 = ref_part.gamma0
        nemitt = np.array([GEMIT_X * (beta0 * gamma0),GEMIT_Y * (beta0 * gamma0)])

        xtrack_line = input_config['xtrack_line']
        rad_line, twiss_rad = initialize_optics_and_calculate_twiss(xtrack_line, config_dict, nemitt, twiss_file_path)
        df_twiss = twiss_rad.to_pandas()
        # Uncomment to save the line after processing, does not work properly if you want to use it later to install collimators
        # LINE IS ALREADY TAPERED NO NEED TO SAVE IT AGAIN
        
        #if beam_config['type'] == 'circulating':
        prepare_injection_insertion(rad_line, df_twiss)

        # Insert monitors
        monitor_names = insert_monitors(num_turns, rad_line, num_particles, config_dict)

        # Install collimators
        twiss_rad = install_collimators(rad_line, input_config, nemitt)
        save_twiss_to_json(twiss_rad, twiss_file_path)  

        if beam_config['type'] == 'injected':            
            # Prepare beam injected from booster
            particles = prepare_injected_beam(twiss_rad, rad_line, ref_part, injection_config, num_particles, capacity)
            start_element = injection_config['start_element']
            
        if beam_config['type'] == 'circulating':   
            # Prepare bumped circulating beam 
            particles = prepare_matched_beam(twiss_rad, rad_line, ref_part, dist_config, nemitt, num_particles, capacity)
            start_element = dist_config['start_element']
            #particles_test =particles.copy()

        radiation_mode = config_dict['run']['radiation']
        beamstrahlung_mode = config_dict['run']['beamstrahlung']
        bhabha_mode = config_dict['run']['bhabha']
        
        configure_tracker_radiation(rad_line, radiation_mode, beamstrahlung_mode, bhabha_mode, for_optics=False)
        if 'quantum' in (radiation_mode, beamstrahlung_mode, bhabha_mode):
            # Explicitly initialise the random number generator for the quantum mode
            seed = config_dict['run']['seed']
            if seed > 1e5:
                raise ValueError('The random seed is too large. Please use a smaller seed (<1e5).')
            seeds = np.full(particles._capacity, seed) + np.arange(particles._capacity)
            particles._init_random_number_generator(seeds=seeds)

        '''test= False
        if test is True:
            rad_line['injection_kick_1'].knl = theta_1
            rad_line['injection_kick_2'].knl = theta_2
            rad_line.track(particles_test ,turn_by_turn_monitor='ONE_TURN_EBE')
            trajectory =  rad_line.record_last_track
            save_track_along_ring_to_h5(trajectory,output_dir)
            
        # Track (saving turn-by-turn data)
        for turn in range(num_turns):
            if turn == 10:
                rad_line['injection_kick_1'].knl = theta_1
                rad_line['injection_kick_2'].knl = theta_2 #3.457728093870607e-5
                print(f'Start turn {turn}, Survivng particles: {particles._num_active_particles}')
                rad_line.track(particles, num_turns=1)
            else:
                rad_line['injection_kick_1'].knl = 0
                rad_line['injection_kick_2'].knl = 0
                rad_line.track(particles, num_turns=1)

            if particles._num_active_particles == 0:
                print(f'All particles lost by turn {turn}, teminating.')
                break
        
        for name in monitor_names:
            last_track = rad_line.get(name)
            save_track_to_h5(last_track, name , output_dir)'''
        
        if beam_config['type'] == 'circulating':
            # To have a correctly closed orbt bump the angle k2 must satify: k2 = sqrt(beta1/beta2) * k1
            beta_1 = twiss_rad['betx', 'injection_kick_1'] # whys this angle? to recah the 10 sigma max
            beta_2 = twiss_rad['betx', 'injection_kick_2']
            theta_1 = 2.4e-5
            theta_2 = theta_1 * (beta_2 / beta_1)
            generate_lossmap(rad_line, num_turns, particles, ref_part, input_config, monitor_names, lossmap_config, start_element, theta_1, theta_2, output_dir, impact=True)
        else:
            generate_lossmap(rad_line, num_turns, particles, ref_part, input_config, monitor_names, lossmap_config, start_element, output_dir=output_dir, impact=True)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Simulation time: {elapsed_time:.4f} seconds")
        print('Done!')
    
if __name__ == "__main__":
    # Setup command-line argument parser
    parser = argparse.ArgumentParser(description='Track particles with a kicker and plot the results.')
    parser.add_argument('--config_file', type=str, required=True, help='Path to the YAML configuration file.')
    parser.add_argument('--submit', action='store_true', help='Select True to submit to htcondor')
    parser.add_argument('--merge', action='store_true', help='Select True to merge output files')
    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.config_file, args.submit, args.merge)


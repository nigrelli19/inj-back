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
import pandas as pd
import xobjects as xo
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from warnings import warn
from copy import deepcopy
from multiprocessing import Pool
from collections import namedtuple
from scipy.optimize import curve_fit
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
        at_element=start_element)
    
    part.start_tracking_at_element = -1
    part.x = part.x + x_offset_injection

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

def generate_lossmap(line, num_turns, particles, ref_part, input_config, monitor_names, lossmap_config, start_element, output_dir="plots", impact=False):

    if impact:
        impacts = xc.InteractionRecord.start(line=line)

    xc.Geant4Engine.start(line=line,
                            particle_ref=ref_part,  
                            seed=lossmap_config['seed'],
                            bdsim_config_file=input_config['bdsim_config'],
                            relative_energy_cut= 0.165)
    line.scattering.enable()

    # Track (saving turn-by-turn data)
    for turn in range(num_turns):
        #print(f'Start turn {turn}, Survivng particles: {particles._num_active_particles}')
        #if turn == 0 and particles.start_tracking_at_element < 0:
        #    line.track(particles, num_turns=1)
        #else:
        line.track(particles, num_turns=1, ele_start=start_element, ele_stop=start_element)          
            
        if particles._num_active_particles == 0:
            print(f'All particles lost by turn {turn}, teminating.')
            break
    for name in monitor_names:
        last_track = line.get(name)
        save_track_to_h5(last_track, num_turns, name , output_dir, turn)
    
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

    particles = particles.remove_unused_space()  

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

    return file_path

def insert_monitors(num_turns, rad_line, twiss, num_particles, start_element):
    """
    Inserts monitors at specific locations in the line:
    - At the syncrhtron mask position tcr.h.c3.2.b1
    - At the primary collimator tcp.h.b1
    - At the experimental insertion IPG.
    - At the 'injection' loation. 
    """
    # Define monitor, for now just one particle
    monitor = xt.ParticlesMonitor(start_at_turn=0, stop_at_turn =num_turns, num_particles = num_particles, auto_to_numpy=True)

    monitors = [monitor.copy() for _ in range(4)]
    monitor_names = ['monitor_prim_coll', 'monitor_at_ipg', 'monitor_mask', 'monitor_inject']

    if isinstance(twiss, pd.DataFrame):
        df_twiss = twiss
        s_mask = df_twiss.loc[df_twiss['name'] == 'tcr.h.c0.2.b1', 's'].values[0]
        s_prim_coll = df_twiss.loc[df_twiss['name'] == 'tcp.h.b1', 's'].values[0]
        s_exp_ipg = df_twiss.loc[df_twiss['name'] == 'ip.4', 's'].values[0]
        s_inj = df_twiss.loc[df_twiss['name'] == start_element, 's'].values[0]
    else:
        s_mask = twiss['s','tcr.h.c0.2.b1'] # SR coll 
        s_prim_coll = twiss['s','tcp.h.b1']
        s_exp_ipg = twiss['s','ip.4']
        df_twiss = twiss.to_pandas()
        s_inj = twiss['s', start_element]

    rad_line.discard_tracker()

    # Insert monitor at the primary collimator position, also with a small offset
    rad_line.insert_element(monitor_names[0], monitors[0], at_s=s_prim_coll + 0.5)

    # Insert monitor at the experietal insertion IPG, also with a small offset
    rad_line.insert_element(monitor_names[1], monitors[1], at_s=s_exp_ipg - 0.1)

    # Insert monitor at the SR mask, adjusting for a small offset
    rad_line.insert_element(monitor_names[2], monitors[2], at_s=s_mask - 0.5)

    # Insert monitor at the beam starting point, adjusting for a small offset
    rad_line.insert_element(monitor_names[3], monitors[3], at_s=s_inj+ 0.01)
    
    return monitor_names #, monitors

def save_track_to_h5(monitor, num_turns, monitor_name='0', output_dir="plots", turn =0):
    # Extract the relevant properties directly from the monitor
    filtered_data = {
        'x': monitor.x,
        'px': monitor.px,
        'y': monitor.y,
        'py': monitor.py,
        'zeta' : monitor.zeta,
        'delta' : monitor.delta,
        'at_turn': monitor.at_turn
        #'at_element': monitor.at_element
    }
    
    # Ensure the output directory exists; if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if turn != 0:
        file_path = os.path.join(output_dir, f'merged_data_{monitor_name}.h5')
    else:
        # Construct the full file path for the .h5 file
        file_path = os.path.join(output_dir, f'merged_data_{monitor_name}.h5')

    try:
        # Open or create an HDF5 file
        with h5py.File(file_path, 'w') as h5f:
            # Directly create each dataset with predefined data types and gzip compression
            h5f.create_dataset('x', data=filtered_data['x'].astype(np.float32), compression='gzip')
            h5f.create_dataset('px', data=filtered_data['px'].astype(np.float32), compression='gzip')
            h5f.create_dataset('y', data=filtered_data['y'].astype(np.float32), compression='gzip')
            h5f.create_dataset('py', data=filtered_data['py'].astype(np.float32), compression='gzip')
            h5f.create_dataset('zeta', data=filtered_data['zeta'].astype(np.float32), compression='gzip')
            h5f.create_dataset('delta', data=filtered_data['delta'].astype(np.float32), compression='gzip')
            h5f.create_dataset('at_turn', data=filtered_data['at_turn'].astype(np.int32), compression='gzip')
            #h5f.create_dataset('at_element', data=filtered_data['at_element'].astype(h5py.string_dtype(encoding='utf-8')), compression='gzip')

        print(f"Data successfully saved to {file_path}")

    except Exception as e:
        print(f"Failed to save data to {file_path}: {e}")

def save_twiss_to_json(df_twiss, file_path):
    """Save Twiss parameters to JSON after converting to a serializable format."""

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
 
def initialize_optics_and_calculate_twiss(xtrack_line, num_turns, twiss_file_path):
    """
    Initialize the optics and calculate Twiss parameters.
    """
    rad_line = xt.Line.from_json(xtrack_line)
    rad_line.build_tracker()
    
    rad_line.configure_radiation(model='mean')
    rad_line.compensate_radiation_energy_loss() 
    tw = rad_line.twiss(eneloss_and_damping=True)

    # Compensate for SR + tapering
    delta0 = tw.delta[0] - np.mean(tw.delta)
    rad_line.compensate_radiation_energy_loss(delta0=delta0)

    # Re-calculate Twiss parameters 
    twiss_rad = rad_line.twiss(method='6d', eneloss_and_damping=True)
    df_twiss_rad = twiss_rad.to_pandas()  
    save_twiss_to_json(df_twiss_rad, twiss_file_path)  
    print(f"Twiss parameters after SR compensation saved to {twiss_file_path}.")

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
    config_dict = load_yaml_config(config_file)

    input_config = config_dict['input']
    beam_config = config_dict['beam']
    run_config = config_dict['run']
    lossmap_config = config_dict['lossmap']
    injection_config = config_dict['injection']
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

        GEMIT_X = REF_PAR["z"]["EMITTANCE_X"]
        GEMIT_Y = REF_PAR["z"]["EMITTANCE_Y"]

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
        
        if os.path.exists(SR_coll_line):
            rad_line = xt.Line.from_json(SR_coll_line)
            if os.path.exists(twiss_file_path):
                print(f"Loading Twiss after SR compensation from {twiss_file_path}...")
                twiss_rad = load_twiss_from_json(twiss_file_path)
            else:
                twiss_rad = rad_line.twiss(method='6d', eneloss_and_damping=True)
                df_twiss_rad = twiss_rad.to_pandas()  
                save_twiss_to_json(df_twiss_rad, twiss_file_path)  
                print(f"Twiss parameters after SR compensation saved to {twiss_file_path}.")
        else:
            xtrack_line = input_config['xtrack_line']
            rad_line, twiss_rad = initialize_optics_and_calculate_twiss(xtrack_line, num_turns, twiss_file_path)
            # Uncomment to save the line after processing, does not work properly if you want to use it later to install collimators
            rad_line.to_json(SR_coll_line)

        # Insert monitors
        start_element = injection_config['start_element']
        monitor_names = insert_monitors(num_turns, rad_line, twiss_rad, num_particles, start_element)
        
        if os.path.exists(lossmap_json):
            #plot_lossmap(lossmap_json, bin_w, output_dir)
            print('Done!')
        else:
            # Install collimators
            twiss_rad = install_collimators(rad_line, input_config, nemitt)

            # Prepare beam injected from booster
            particles = prepare_injected_beam(twiss_rad, rad_line, ref_part,injection_config, num_particles, capacity)
            
            lossmap_json = generate_lossmap(rad_line, num_turns, particles, ref_part, input_config, monitor_names, lossmap_config, start_element, output_dir, impact=False)
            
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


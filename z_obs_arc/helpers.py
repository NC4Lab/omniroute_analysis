import os
from dateutil.parser import parse as parsedate
import logging
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import re
import shutil
import math
import pickle
import glob
from signal_processing import *
from binary_utils import TrodesDIOBinaryLoader

logger = logging.getLogger(__name__)
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)

## Data path level functions
def get_session_folder(data_folder, anim, date):
    anim_folder = get_anim_folder(data_folder, anim)

    if '-' in date: # e.g. date = '14-Apr-23'
        date = parsedate(date).strftime('%y%m%d')

    subfolders = [ f.path for f in os.scandir(anim_folder) if f.is_dir() ]
    session_folders = [s for s in subfolders if date in s]
    if not session_folders:
        logger.error("Session Directory does not exist")
        return None
    elif len(session_folders)>1:
        logger.warning("Multiple session folders found: %s" % ', '.join(['%s']*len(session_folders)))
    else:
        logger.info('Session folder located at %s'% session_folders[0])
    return session_folders[0]

def get_anim_folder(data_folder, anim):
    anim_folder = os.path.join(data_folder,'NC4%04d' % anim)

    if not os.path.isdir(anim_folder):
        logger.error("Animal directory does not exist: %s" % anim_folder)
        return None
    else:
        logger.info('Animal folder located at %s' % anim_folder)
        return anim_folder

def get_trf(anim):
    '''Return the task-relevant frame for the given animal in str'''
    if isinstance(anim, str):
        anim = int(anim)
    if anim == 10 or anim == 14:
        return 'sound'
    elif anim == 15:
        return 'landmark'
    else:
        return None
    
def get_n_linearLaps(anim, date):
    if anim == 15:
        if date >= '240726': # date started cue-off epochs
            if date >= '240806':
                return 10
            else:
                return 20
        

def remove_output_folder(folder_name):
    """Creates an output directory in the current directory and then makes a new folder within it with the specified name.
        If it exists, removes it
    """
    output_folder = os.path.join('.', 'output')
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    
    folder_name = os.path.join(output_folder, folder_name)
    if os.path.isdir(folder_name):
        shutil.rmtree(folder_name)
    
    return folder_name

def clean_output_folder(folder_name):
    """Creates an output directory in the current directory and then makes a new folder within it with the specified name.
        If it exists, cleans it
    """
    output_folder = os.path.join('.', 'output')
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    
    folder_name = os.path.join(output_folder, folder_name)
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    else:
        clean_folder(folder_name)
    
    return folder_name

def clean_folder(folder):
    """Cleans everything from a folder
        USE WITH CAUTION!!
    """
    for files in os.listdir(folder):
        path = os.path.join(folder, files)
        try:
            shutil.rmtree(path)
        except OSError:
            os.remove(path)

## Session class (Experimental) - to wrap all session-specific code
class Session:
    def __init__(self, folder='', anim='', date=''):
        if anim=='' or date=='':
            if folder=='':      ## Empty session
                logger.info('Initializing empty session')
            else:               ## Session folder is provided
                logger.info('Initializing session using folder only')
                
                if not os.path.isdir(folder):
                    logger.error("Session Folder does not exist")
                self.sess_folder = Path(folder).absolute().name
                self.date = parsedate(folder, dayfirst=False, yearfirst=True, fuzzy=True).strftime('%y%m%d')
                self.anim = re.findall(r'NC4\d+', folder)[0]
                
        else:   ## Both anim and date are provided, folder is assumed to be data folder
            if folder=='':      ## Data path not provided
                folder = os.environ['NC4_DATA_DIR']
            
            if not os.path.isdir(folder):
                logger.error("Session Folder does not exist")
            self.date = date
            self.anim = anim
            self.sess_folder = get_session_folder(folder, self.anim, self.date)


def get_subfolder(session_folder, subfolders, create_if_missing=False):
    current_folder = session_folder
    raw_flag = False

    if isinstance(subfolders, str):
        subfolders = [subfolders]

    for folder in subfolders:
        raw_flag = raw_flag or (folder == 'raw')
        current_folder = os.path.join(current_folder, folder)

        if not os.path.isdir(current_folder):
            if raw_flag: 
                logger.error('Cannot write inside raw folder')
                return None
                
            if create_if_missing:
                logger.info('Folder not found: %s' % current_folder)
                os.mkdir(current_folder)
                logger.info('Creating folder: %s' % current_folder)
            else:
                logger.error('Folder not found: %s' % current_folder)
                return None
    
    return current_folder

def get_pyks_folder(session_folder, rec_file_name, create_if_missing=True, clean=False):
    rec_base_name = get_base_name(rec_file_name)
    pyks_folder_name = rec_base_name+'.pyks'
    pyks_folder_name = get_subfolder(session_folder, ['analyzed', pyks_folder_name], create_if_missing)
    if clean:
        clean_folder(pyks_folder_name)
    return pyks_folder_name

def get_si_folder(session_folder, rec_file_name, create_if_missing=True, clean=False):
    rec_base_name = get_base_name(rec_file_name)
    si_folder_name = rec_base_name+'.si'
    si_folder_name = get_subfolder(session_folder, ['analyzed', si_folder_name], create_if_missing)
    if clean:
        clean_folder(si_folder_name)
    return si_folder_name


def get_analyzed_folder(session_folder, create_if_missing=True):
    return get_subfolder(session_folder, 'analyzed', create_if_missing)

def get_mts_analyzed_folder(session_folder, mts_vers, create_if_missing=True):
    '''Return the path to the Mountiansort subfolder in thhe analyzed folder\n
    mts_vers: 'mts4' or 'mts5' '''
    if mts_vers == 'mts4':
        return get_subfolder(session_folder, ['analyzed', 'mts4'], create_if_missing)
    if mts_vers == 'mts5':
        return get_subfolder(session_folder, ['analyzed', 'mts_scheme2'], create_if_missing)

def get_raster_plot_folder(session_folder, create_if_missing=True):
    return get_subfolder(session_folder, ['analyzed', 'mts_scheme2', 'raster_plot'], create_if_missing)

def get_unit_heatmap_folder(session_folder, create_if_missing=True):
    return get_subfolder(session_folder, ['analyzed', 'unit_heatmap'], create_if_missing)

def get_extracted_folder(session_folder, create_if_missing=False):
    return get_subfolder(session_folder, 'extracted', create_if_missing)

def get_theta_folder(session_folder, create_if_missing=True):
    return get_subfolder(session_folder, ['extracted', 'theta'], create_if_missing)

def get_raw_folder(session_folder):
    return get_subfolder(session_folder, 'raw', create_if_missing=False)

def get_trodes_folder(session_folder):
    return get_subfolder(session_folder, ['raw', 'Trodes'], create_if_missing=False)

def get_ros_folder(session_folder):
    return get_subfolder(session_folder, ['raw', 'ROS'], create_if_missing=False)

def get_sorting_folder(session_folder, create_if_missing=True):
    return get_subfolder(session_folder, 'sorting', create_if_missing)

def get_mts_folder(session_folder, mts_vers, create_if_missing=False):
    return get_subfolder(session_folder, ['sorting', mts_vers], create_if_missing)

def get_mts_object_folder(session_folder, mts_vers, scheme, tetrode, autocurated = False):
    """
    mts_vers: "mts4" or "mts5" \n
    Scheme: scheme of MountainSort5. Scheme is None for MountainSort4.
    autocurated: if True, return the autocurated mountainsort object folder
    """
    if isinstance(tetrode, int):
        tetrode = str(tetrode)

    if autocurated:
        mts_object_folder_name = 'mts_object_autocurated'
    else:
        mts_object_folder_name = 'mts_object'

    if mts_vers == 'mts5':
        subfolder = get_subfolder(session_folder, ['sorting', mts_vers, mts_object_folder_name, 'scheme%d' % scheme], create_if_missing=True)

    if mts_vers == 'mts4':
        subfolder = get_subfolder(session_folder, ['sorting', mts_vers, mts_object_folder_name], create_if_missing=True)

    # Need the folder name for saving sorting object when the folder doesn't exist
    # Thus use os.path.join instead of get_subfolder
    return os.path.join(subfolder, 'mts_TT%s' %tetrode)

def get_spike_vector_folder(session_folder, mts_vers, scheme, autocurated = False, create_if_missing=True):
    """Scheme: scheme of MountainSort5. Scheme is None for MountainSort4.
    autocurated: if True, return the autocurated spike vector folder
    """
    sorting_folder = get_sorting_folder(session_folder)

    if autocurated:
        spike_vector_folder_name = 'spike_vector_autocurated'
    else:
        spike_vector_folder_name = 'spike_vector'
    
    if mts_vers == 'mts5':
        return get_subfolder(sorting_folder, [mts_vers, spike_vector_folder_name, 'scheme%d' % scheme], create_if_missing)
    
    if mts_vers == 'mts4':
        return get_subfolder(sorting_folder, [mts_vers, spike_vector_folder_name], create_if_missing)
    

def get_report_folder(session_folder, mts_vers, scheme, tetrode, autocurated = False):
    '''autocurated: if True, return the autocurated report folder'''
    if isinstance(tetrode, int):
        tetrode = str(tetrode)

    if autocurated:
        report_folder_name = 'report_autocurated'
    else:
        report_folder_name = 'report'

    if mts_vers == 'mts5':
        subfolder = get_subfolder(session_folder, ['sorting', mts_vers, report_folder_name, 'scheme%d' % scheme], create_if_missing=True)
    
    if mts_vers == 'mts4':
        subfolder = get_subfolder(session_folder, ['sorting', mts_vers, report_folder_name], create_if_missing=True)

    # Need the folder name for saving report when the folder doesn't exist
    # Thus use os.path.join instead of get_subfolder
    return os.path.join(subfolder, 'mts_TT%s' %tetrode)

def get_wf_folder(session_folder, mts_vers, scheme, tetrode, autocurated = False):
    '''Return the path to the waveforms folder with given session, mountainsort version, scheme and tetrode \n
    autocurated: if True, return the autocurated waveform folder'''
    if isinstance(tetrode, int):
        tetrode = str(tetrode)

    if autocurated:
        waveforms_folder_name = 'waveforms_autocurated'
    else:
        waveforms_folder_name = 'waveforms'

    if mts_vers == 'mts5':
        subfolder = get_subfolder(session_folder, ['sorting', mts_vers, waveforms_folder_name, 'scheme%d' % scheme], create_if_missing=True)
    
    if mts_vers == 'mts4':
        subfolder = get_subfolder(session_folder, ['sorting', mts_vers, waveforms_folder_name], create_if_missing=True)
   
    # Need the folder name for saving waveforms when the folder doesn't exist
    # Thus use os.path.join instead of get_subfolder
    return os.path.join(subfolder, 'mts_TT%s' %tetrode)

def get_qm_folder(session_folder, mts_vers, scheme, autocurated = False):
    '''Return the path to the quality metrics folder with given session, mountainsort version, scheme \n
    Quality metrics pkl files of all tetrodes are saved in the same folder \n
    autocurated: if True, return the autocurated quality metrics folder'''

    if autocurated:
        qm_folder_name = 'quality_metrics_autocurated'
    else:
        qm_folder_name = 'quality_metrics'

    if mts_vers == 'mts5':
        subfolder = get_subfolder(session_folder, ['sorting', mts_vers, qm_folder_name, 'scheme%d' % scheme], create_if_missing=True)
    
    if mts_vers == 'mts4':
        subfolder = get_subfolder(session_folder, ['sorting', mts_vers, qm_folder_name], create_if_missing=True)
   
    # Need the folder name for saving waveforms when the folder doesn't exist
    # Thus use os.path.join instead of get_subfolder
    return subfolder

def save_quality_metrics(session_folder, mts_vers, scheme, tetrode, qm_df, autocurated = False):
    '''Save the quality metrics dataframe to the pkl file in the mts_vers/quality_metrics folder \n
    Note that a csv file is autosaved in the waveforms folder by spikeinterface, \n
    but the dataframe read from that csv file is not the same as the one output by si.compute_quality_metrics \n
    The former has first column named as "Unnamed: 0" \n
    '''
    qm_folder = get_qm_folder(session_folder, mts_vers, scheme, autocurated)
    pkl_file_path = os.path.join(qm_folder, 'quality_metrics_TT%s.pkl' %tetrode)
    qm_df.to_pickle(pkl_file_path)

def get_quality_metrics_df(session_folder, mts_vers, scheme, tetrode, autocurated = False):
    '''Read the qulity metrics pkl file in the mts_vers/quality_metrics folder and return the dataframe \n
    Notice that this is not reading the csv file in the waveforms folder autosaved by spikeinterface \n
    Wasn't able to find a function in spikeinterface to read the csv file \n
    '''
    qm_folder = get_qm_folder(session_folder, mts_vers, scheme, autocurated)
    pkl_file = os.path.join(qm_folder, 'quality_metrics_TT%s.pkl' %tetrode)
    if os.path.isfile(pkl_file):
        return pd.read_pickle(pkl_file)
    else:
        logger.error('pkl file does not exist: %s' % pkl_file)
        return None    

def save_autocurated_unit_ids(session_folder, mts_vers, scheme, kept_unit_ids_df):
    '''Save the autocurated unit ids dataframe to the pkl file in the mts_vers folder \n'''
    if mts_vers == 'mts5':
        subfolder = get_subfolder(session_folder, ['sorting', mts_vers], create_if_missing=False)
        pkl_file_path = os.path.join(subfolder, 'autocurated_unit_ids_scheme%d.pkl' %scheme)
    
    if mts_vers == 'mts4':
        subfolder = get_subfolder(session_folder, ['sorting', mts_vers], create_if_missing=False)
        pkl_file_path = os.path.join(subfolder, 'autocurated_unit_ids.pkl')
    kept_unit_ids_df.to_pickle(pkl_file_path)

def get_autocurated_unit_ids_df(session_folder, mts_vers, scheme):
    '''Read the autocurated unit ids dataframe from the pkl file in the mts_vers folder \n'''
    if mts_vers == 'mts5':
        subfolder = get_subfolder(session_folder, ['sorting', mts_vers], create_if_missing=False)
        pkl_file = os.path.join(subfolder, 'autocurated_unit_ids_scheme%d.pkl' %scheme)
    
    if mts_vers == 'mts4':
        subfolder = get_subfolder(session_folder, ['sorting', mts_vers], create_if_missing=False)
        pkl_file = os.path.join(subfolder, 'autocurated_unit_ids.pkl')
    
    if os.path.isfile(pkl_file):
        return pd.read_pickle(pkl_file)
    else:
        logger.error('pkl file does not exist: %s' % pkl_file)
        return None

def get_num_autocurated_units(session_folder, mts_vers, scheme):
    '''Return the number of autocurated units of a session'''
    autocurated_units_df = get_autocurated_unit_ids_df(session_folder, mts_vers, scheme)
    num_units = 0
    for i in range(len(autocurated_units_df['Tetrode'])):
        num_units = num_units + len(autocurated_units_df['Autocurated_unit_ids'][i])
    
    return num_units

def get_visually_excluded_unit_ids_xlsx(session_folder, mts_vers, scheme):
    '''Return the visually_excluded_unit_ids.xlsx file in the mts_vers folder \n'''
    if mts_vers == 'mts5':
        subfolder = get_subfolder(session_folder, ['sorting', mts_vers], create_if_missing=False)
        xlsx_file = os.path.join(subfolder, 'visually_excluded_unit_ids_scheme%d.xlsx' %scheme)
    
    if mts_vers == 'mts4':
        subfolder = get_subfolder(session_folder, ['sorting', mts_vers], create_if_missing=False)
        xlsx_file = os.path.join(subfolder, 'visually_excluded_unit_ids.xlsx')
    
    if os.path.isfile(xlsx_file):
        return xlsx_file
    else:
        logger.error('xlsx file does not exist: %s' % xlsx_file)
        return None
 

def get_visually_excluded_unit_ids_df(session_folder, mts_vers, scheme):
    '''Read the the visually_excluded_unit_ids.xlsx file and convert the dataframe from each unit_id having
    one entry to units on the same tetrode having one entry'''
    if get_visually_excluded_unit_ids_xlsx(session_folder, mts_vers, scheme) is not None:
        original_df = pd.read_excel(get_visually_excluded_unit_ids_xlsx(session_folder, mts_vers, scheme))
    else: 
        return None
    
    tetrodes = np.unique(original_df['tetrode'])
    excluded_unit_ids_array = []
    for tt in tetrodes:
        tt_df = original_df[original_df['tetrode']==tt]
        excluded_unit_ids_array.append(tt_df['excluded_unit_id'].to_numpy())

    d = {'tetrode': tetrodes, 'excluded_unit_ids': excluded_unit_ids_array}
    excluded_units_df = pd.DataFrame(data=d)
    return excluded_units_df
    

def get_phy_folder(session_folder, mts_vers, scheme, tetrode):
    if isinstance(tetrode, int):
        tetrode = str(tetrode)

    if mts_vers == 'mts5':
        subfolder = get_subfolder(session_folder, ['sorting', mts_vers, 'phy', 'scheme%d' % scheme], create_if_missing=True)
    
    if mts_vers == 'mts4':
        subfolder = get_subfolder(session_folder, ['sorting', mts_vers, 'phy'], create_if_missing=True)
    # Need the folder name for saving phy filess when the folder doesn't exist
    # Thus use os.path.join instead of get_subfolder
    return os.path.join(subfolder, 'mts_TT%s' %tetrode)


def get_rec_files(session_folder):
    trodes_folder = get_trodes_folder(session_folder)

    rec_file_list = []
    for file in os.listdir(trodes_folder):
        if file.endswith('.rec'):
            rec_file_list.append(os.path.join(trodes_folder, file))
    
    return rec_file_list

def get_merged_rec(session_folder):
    trodes_folder = get_trodes_folder(session_folder)

    merged_rec_file_list = []
    for file in os.listdir(trodes_folder):
        if file.endswith('_merged.rec'):
            merged_rec_file_list.append(os.path.join(trodes_folder, file))
    
    if len(merged_rec_file_list) == 1:
        return merged_rec_file_list[0]
    
    elif len(merged_rec_file_list) > 1:
        logger.error("More than one merged rec file found at %s" % session_folder)
        return None
    
    else:
        logger.error("No merged rec file found at %s" % session_folder)
        return None
    

def get_ros_data_bag_files(session_folder):
    ros_folder = get_ros_folder(session_folder)

    ros_data_bag_file_list = []
    for file in os.listdir(ros_folder):
        if 'ExperimentData' in file and file.endswith('.bag'):
            ros_data_bag_file_list.append(os.path.join(ros_folder, file))
    
    return ros_data_bag_file_list

def get_ros_camera_bag_files(session_folder):
    ros_folder = get_ros_folder(session_folder)

    ros_camera_bag_file_list = []
    for file in os.listdir(ros_folder):
        if 'ExperimentCamera' in file and file.endswith('.bag'):
            ros_camera_bag_file_list.append(os.path.join(ros_folder, file))
    
    return ros_camera_bag_file_list

def get_dio(session_folder, rec_file_name, channel):
    extracted_folder = get_extracted_folder(session_folder)
    dio_folder = os.path.join(extracted_folder, get_base_name(rec_file_name)) + '.DIO'
    if not os.path.isdir(dio_folder):
        logger.error('DIO folder not found : %s' % dio_folder)
        return None

    for file in os.listdir(dio_folder):
        if file.endswith('Din%d.dat' % channel):
            dio_bin = TrodesDIOBinaryLoader(os.path.join(dio_folder, file))
            return dio_bin
    
    logger.error('DIO file not found for channel %d' % channel)

def get_folder_date(session_folder):
    '''Get the date from the provided session folder'''
    date = os.path.basename(session_folder).split('_')[0]
    return date

def get_base_name(rec_file_name):
    date = parsedate(os.path.basename(rec_file_name), dayfirst=False, yearfirst=True, fuzzy=True).strftime('%y%m%d')
    label = Path(rec_file_name).stem.split('_')[-1]
    return date+'_'+label


# TODO: Use code from rec_to_binaries to speed up
def export_dio(session_folder, rec_files=[]):
    #analyzed_folder = get_analyzed_folder(session_folder, create_if_missing=True)
    extracted_folder = get_extracted_folder(session_folder, create_if_missing=True)
    session_rec_files = get_rec_files(session_folder)

    for file in session_rec_files:
        if (not rec_files) or (file in rec_files):
            export_cmd = 'exportdio -rec %s -outputdirectory %s -output %s' % (file, extracted_folder, Path(file).stem)
            logger.info('Extracting DIO for %s', file)
            result = subprocess.run(export_cmd, capture_output=True, text=True)
            logger.info("exportdio stdout: %s" % result.stdout)
            if result.stderr:
                logger.error("exportdio stderr: %s" % result.stderr)

def get_experiment_vars_df(session_folder):
    extracted_folder = get_extracted_folder(session_folder)
    file_name = os.path.join(extracted_folder, 'experiment_vars.pkl')
    if os.path.isfile(file_name):
        logger.info('experimental_vars pkl file found at %s', file_name)
        return pd.read_pickle(file_name)
    else:
        logger.error('experimental_vars pkl file not found: %s', file_name)
        return None
    
def get_nose_pokes_df(session_folder):
    extracted_folder = get_extracted_folder(session_folder)
    file_name = os.path.join(extracted_folder, 'nose_pokes.pkl')
    if os.path.isfile(file_name):
        logger.info('nose_pokes pkl file found at %s', file_name)
        return pd.read_pickle(file_name)
    else:
        logger.error('nose_pokes pkl file not found: %s', file_name)
        return None

def get_rewards_df(session_folder):
    extracted_folder = get_extracted_folder(session_folder)
    file_name = os.path.join(extracted_folder, 'rewards.pkl')
    if os.path.isfile(file_name):
        logger.info('rewards pkl file found at %s', file_name)
        return pd.read_pickle(file_name)
    else:
        logger.error('rewards pkl file not found: %s', file_name)
        return None
                
def get_cluster_file(session_folder, tetrode, cluster_index):
    """Return t64 file (manually sorted via MClust) name"""
    extracted_folder = get_extracted_folder(session_folder, create_if_missing=False)

    date = parsedate(os.path.basename(session_folder), dayfirst=False, yearfirst=True, fuzzy=True).strftime('%y%m%d')
    if not date:
        logger.error('date not found in session folder name: %s', session_folder)
        return None
    
    merged_spikes_folder = os.path.join(extracted_folder,'%s_merged.spikes' % date)
    if not os.path.isdir(merged_spikes_folder):
        logger.error('merged_spikes folder not found in %s' % extracted_folder)
    else:
        logger.info('merged_spikes folder found at %s', merged_spikes_folder)
    
    file_name = os.path.join(merged_spikes_folder, '%s_merged.spikes_nt%d_%02d.t64' % (date, tetrode, cluster_index))
    if os.path.isfile(file_name):
        return file_name
    else:
        logger.error('Cluster file not found: %s', file_name)
        return None
    
    
def get_timestamps_dat(session_folder):
    """Return [date]_merged.timestamps.dat file name"""
    extracted_folder = get_extracted_folder(session_folder, create_if_missing=False)
    date = parsedate(os.path.basename(session_folder), dayfirst=False, yearfirst=True, fuzzy=True).strftime('%y%m%d')
    if not date:
        logger.error('date not found in session folder name: %s', session_folder)
        return None
    
    merged_time_folder = os.path.join(extracted_folder,'%s_merged.time' % date)
    if not os.path.isdir(merged_time_folder):
        logger.error('merged_time folder not found in %s' % extracted_folder)
    else:
        logger.info('merged_time folder found at %s', merged_time_folder)
        
    file_name = os.path.join(merged_time_folder, '%s_merged.timestamps.dat' % date)
    if os.path.isfile(file_name):
        return file_name
    else:
        logger.error('Timestamps dat file not found: %s', file_name)
        return None
    

def read_timestamps_dat(session_folder, skip_timestamps = False, save_timestamps_array = False):
    """Read the timestamps_dat ([date]_merged.timestamps.dat) file and 
    return a dictionary of system time at creation, SpikeGadgets timestamp at creation,
    SpikeGadgets first timestamp and SpikeGadgets timestamps array

    If skip_timestamps == True, skip reading timestamps to save time
    in case only creation time is needed
    If save_timestamps_array == True, save the SpikeGadgets timestamps array to 
    the extracted folder as a npy file
    """
    
    n1 = 20 # number of lines before System_time_at_creation in the dat file header
    n2 = 3 # number of lines after First_timestamp in the dat file head
    # data type of the SpikeGadgets timestamps and system time in the dat file
    dt = np.dtype([('time', '<u4'), ('systime', '<i8')])

    timestamps_dat = get_timestamps_dat(session_folder)
    
    with open(timestamps_dat, 'rb') as file:
        for i in range(n1):
            line = next(file).strip()
            print(line)
            
        line_21 = next(file).strip()
        system_time_at_creation = int(line_21.decode("utf-8").split(': ')[1])
        print(line_21)
        print('System time at creation:' + str(system_time_at_creation))
        line_22 = next(file).strip()
        spikegadgets_timestamp_at_creation = int(line_22.decode("utf-8").split(': ')[1])
        print(line_22)
        print('SpikeGadgets timestamp at creation:' + str(spikegadgets_timestamp_at_creation))
        line_23 = next(file).strip()
        spikegadgets_first_timestamp = int(line_23.decode("utf-8").split(': ')[1])
        print(line_23)
        print('SpikeGadgets first timestamp:' + str(spikegadgets_first_timestamp))
        
        for i in range(n2):
            line = next(file).strip()
            print(line)
        if skip_timestamps == False:
            timestamps_systime = np.fromfile(file, dtype = dt)
            spikegadgets_timestamps = pd.DataFrame(timestamps_systime, columns=['time']).to_numpy()
        else:
            spikegadgets_timestamps = None
    file.close()
    
    if save_timestamps_array:
        extracted_folder = get_extracted_folder(session_folder, create_if_missing=False)
        file_name = os.path.join(extracted_folder, 'spikegadgets_timestamps.npy')
        np.save(file_name, spikegadgets_timestamps)
    
    datdict = {'systime_creation': system_time_at_creation,
           'spikegadgets_timestamp_creation': spikegadgets_timestamp_at_creation,
           'spikegadgets_first_timestamp': spikegadgets_first_timestamp,
           'spikegadgets_timestamps': spikegadgets_timestamps
           }
    return datdict

def save_time_dict(session_folder):
    time_dict = read_timestamps_dat(session_folder, skip_timestamps = False)
    extracted_folder = get_extracted_folder(session_folder)
    dict_filename = os.path.join(extracted_folder, 'spikegadgets_time_dict.pkl')
    with open(dict_filename, 'wb') as f:
        pickle.dump(time_dict, f)

def get_time_dict(session_folder):
    '''Get the Spikegadgets time dictonary that has systime_creation, spikegadgets_timestamp_creation \n
    spikegadgets_first_timestamp, spikegadgets_timestamps from spikegadgets_time_dict.pkl in the extracted folder \n
    if the pkl file exists
    '''
    extracted_folder = get_extracted_folder(session_folder, create_if_missing = False)
    dict_filename = os.path.join(extracted_folder, 'spikegadgets_time_dict.pkl')
    if os.path.isfile(dict_filename):
        logger.info("Loading pkl file: %s" % dict_filename)
        with open(dict_filename, 'rb') as f:
            time_dict = pickle.load(f)
        return time_dict
    else:
        logger.warning("pkl file doesn't exsit: %s" % dict_filename)
        return None

## Functions for processing sorted data from MountainSort
def save_spike_vector(session_folder, spike_vector, mts_vers, scheme, tetrode, autocurated = False):
    """Save the spike vector from MountainSort"""
    spike_vector_folder = get_spike_vector_folder(session_folder, mts_vers, scheme, autocurated, create_if_missing=True)
    file_name = os.path.join(spike_vector_folder, 'spike_vector_TT%d' % tetrode)
    np.save(file_name, spike_vector)


def get_mts_df(session_folder, mts_vers, scheme, tetrode, autocurated):
    """ Return a dataframe of units sorted by mountainsort on a tetrode of a session.
    Spike vector loaded from a npy file and converted into a pandas dataframe
    """

    if isinstance(tetrode, int):
        tetrode = str(tetrode)
    
    spike_vector_folder = get_spike_vector_folder(session_folder, mts_vers, scheme, autocurated)
    file_name = os.path.join(spike_vector_folder, 'spike_vector_TT%s.npy' % tetrode)
    if os.path.isfile(file_name):
        spike_vector = np.load(file_name)
        logger.info("Loaded %s"  %file_name)
        df = pd.DataFrame(spike_vector, columns=['sample_ind', 'unit_ind'])
        return df
    else:
        logger.error('Failed to retrieve a dataframe. Spike vector npy file not found: %s', file_name)
        return None


def get_mts_unit_num(session_folder, mts_vers, scheme, tetrode, autocurated):
    """Return the number of units sorted by MountainSort"""
    mts_df = get_mts_df(session_folder, mts_vers, scheme, tetrode, autocurated)
    if autocurated:
        autocurated_tag = ' autocurated'
    else:
        autocurated_tag = ''
    if mts_df is None:
        logging.info("No%s units on tetrode %d in %s" % (autocurated_tag, tetrode, session_folder))
        return 0
    else:
        return len(mts_df['unit_ind'].unique())
    
def get_mts_unit_sample_index(session_folder, mts_vers, scheme, tetrode, unit_index, autocurated):
    """Return the sample indices of a MountainSort sorted unit"""
    mts_df = get_mts_df(session_folder, mts_vers, scheme, tetrode, autocurated)
    if mts_df is None:
        return None
    else:
        # input index starts with 1 while MountainSort spike vector index starts with 0
        unit_index = unit_index - 1
        unit_sample_index = mts_df[mts_df['unit_ind']==unit_index]['sample_ind'].to_numpy()
        return unit_sample_index
    
def get_mts_unit_timestamps(session_folder, mts_vers, scheme, tetrode, unit_index, autocurated, spikegadgets_timestamps = None):
    """Return the SpikeGadgets timestamps of a MountainSort sorted unit \n
    Read SpikeGadgets timestamps from dat file if not given
    """
    unit_sample_index = get_mts_unit_sample_index(session_folder, mts_vers, scheme, tetrode, unit_index, autocurated)
    if spikegadgets_timestamps is None:
        spikegadgets_timestamps = read_timestamps_dat(session_folder, skip_timestamps = False)['spikegadgets_timestamps']
    
    # some units have outbound sample indices, only use the inbound indices
    inbound_unit_sample_index = unit_sample_index[unit_sample_index < len(spikegadgets_timestamps)]

    unit_inbound_n = inbound_unit_sample_index.shape[0] # number of inbound sample indices of the unit
    if unit_inbound_n < unit_sample_index.shape[0]:
        logging.warn("""Tetrode %d unit index %d has sample index out of bound of the
                     spikeGadgets timestamps array. Session folder: %s
                     """ % (tetrode, unit_index, session_folder))

    unit_sg_timestamps = np.zeros(unit_inbound_n, dtype='<i8')
    for i in range(unit_inbound_n):
        unit_sg_timestamps[i] = spikegadgets_timestamps[inbound_unit_sample_index[i]]

    return unit_sg_timestamps


def get_unit_to_plot_xlsx(session_folder):
    """Return the path to a xlsx file with information of which units (visually picked by \n
    checking report plots) from a session to generate information score histogram
    """
    sorting_folder = get_sorting_folder(session_folder)
    return os.path.join(sorting_folder, 'unit_to_plot.xlsx')

def get_unit_to_plot_df(session_folder):
    """Return a pandas dataframe with information of which units
    from a session to generate information score histogram
    """
    xlsx_file = get_unit_to_plot_xlsx(session_folder)
    return pd.read_excel(xlsx_file)

def save_info_fr_array(session_folder, info_fr_array):
    """Save a 2D array with information scores and firing rates of units in a session"""
    mts_analyzed_folder = get_mts_analyzed_folder(session_folder)
    file_name = os.path.join(mts_analyzed_folder, 'info_fr_array')
    np.save(file_name, info_fr_array)

def info_fr_array_to_df(info_fr_array):
    info_fr_df =  pd.DataFrame(info_fr_array,
                columns=['tetrode', 'unit_index', 'n_spikes', 'velocity_filtered_n_spikes', 'sound_info', 'lab_info', 'landmark_info',
                    'sound_lab_info', 'sound_landmark_info', 'landmark_lab_info',
                    'sound_max_fr', 'lab_max_fr', 'landmark_max_fr'])
    
    convert_dict = {'tetrode': int,
                'unit_index': int,
                'n_spikes': int,
                'velocity_filtered_n_spikes': int
                } 
    info_fr_df = info_fr_df.astype(convert_dict)
    return info_fr_df

def save_info_fr_df(session_folder, info_fr_df, mts_vers, df_name = 'info_fr_df_n_spikes'):
    '''Save the provided info_fr_df to the mts subfolder in analyzed folder'''
    mts_analyzed_folder = get_mts_analyzed_folder(session_folder, mts_vers)
    file_name = os.path.join(mts_analyzed_folder, '%s.pkl' % df_name)
    info_fr_df.to_pickle(file_name)

def get_info_fr_df(session_folder, mts_vers, df_with_n_spikes=True):
    '''
    df_with_n_spikes: bool, get df with n_spikes and velocity_filtered_n_spikes columns or not
    '''
    mts_analyzed_folder = get_mts_analyzed_folder(session_folder, mts_vers)
    if df_with_n_spikes:
        df_file = 'info_fr_df_n_spikes.pkl'
    else:
        df_file = 'info_fr_df.pkl'
    pkl_file = os.path.join(mts_analyzed_folder, df_file)
    if os.path.isfile(pkl_file):
        logger.info("pkl file found at %s", pkl_file)
        return pd.read_pickle(pkl_file)
    else:
        logger.error("pkl file not found at %s", pkl_file)

def save_trodes_sync_ts_array(session_folder, trodes_sync_ts):
    '''Save the Trodes sync timestamps array to extracted folder'''
    extracted_folder = get_extracted_folder(session_folder)
    np.save(os.path.join(extracted_folder, 'Trodes_sync_timestamps.npy'), trodes_sync_ts)
    logger.info("trodes_sync_ts saved as %s" % os.path.join(extracted_folder, 'Trodes_sync_timestamps.npy'))

def load_trodes_sync_ts_array(session_folder):
    '''Load the Trodes sync timestamps array to extracted folder'''
    extracted_folder = get_extracted_folder(session_folder)
    trodes_sync_ts_npy = os.path.join(extracted_folder, 'Trodes_sync_timestamps.npy')
    if os.path.isfile(trodes_sync_ts_npy):
        logging.info("Loading %s" % trodes_sync_ts_npy)
        return np.load(trodes_sync_ts_npy)
    else:
        logger.error("npy file not found: %s" % trodes_sync_ts_npy)
        return None
    
def save_ros_sync_ts_array(session_folder, ros_sync_ts):
    '''Save the ROS sync timestamps array to extracted folder'''
    extracted_folder = get_extracted_folder(session_folder)
    np.save(os.path.join(extracted_folder, 'ROS_sync_timestamps.npy'), ros_sync_ts)
    logger.info("ros_sync_ts saved as %s" % os.path.join(extracted_folder, 'ROS_sync_timestamps.npy'))

def load_ros_sync_ts_array(session_folder):
    '''Load the ROS sync timestamps array to extracted folder'''
    extracted_folder = get_extracted_folder(session_folder)
    ros_sync_ts_npy = os.path.join(extracted_folder, 'ROS_sync_timestamps.npy')
    if os.path.isfile(ros_sync_ts_npy):
        logging.info("Loading %s" % ros_sync_ts_npy)
        return np.load(ros_sync_ts_npy)
    else:
        logger.error("npy file not found: %s" % ros_sync_ts_npy)
        return None

def save_sync_p(session_folder):
    ''' Save the parameters for synchronizing Trodes and ROS timestamps \n
    trodes_sync_ts and ros_sync_ts are needed
    '''
    trodes_sync_timestamps = load_trodes_sync_ts_array(session_folder)
    ros_sync_timestamps = load_ros_sync_ts_array(session_folder)
    ## Sync timestamps
    #p = np.polyfit(x_match, y_match, 1)
    sync_p, trodes_sync_ts_match, ros_sync_ts_match = align_timestamps_nw(trodes_sync_timestamps,
                                                                    ros_sync_timestamps, new = True)
    extracted_folder = get_extracted_folder(session_folder)
    np.save(os.path.join(extracted_folder, 'sync_p.npy'), sync_p)

def load_sync_p(session_folder):
    '''Load the parameters for synchronizing Trodes and ROS timestamps from the extracted folder'''
    extracted_folder = get_extracted_folder(session_folder)
    sync_p_file = os.path.join(extracted_folder, 'sync_p.npy')
    if os.path.isfile(sync_p_file):
        return np.load(sync_p_file)
    else:
        logger.error("%s is not a valid file" % sync_p_file)

def get_num_tt_manual_cluster(session_folder, tetrode):
    '''Return the number of manual clusters on a given tetrode in a given session'''
    extracted_folder = get_extracted_folder(session_folder)
    spikes_folder = glob.glob(os.path.join(extracted_folder, '*.spikes'))
    if len(spikes_folder) == 1:
        spikes_folder = spikes_folder[0]
    else:
        logger.warning("More than one .spikes folder exist in the extracted folder")
        return None
    
    if os.path.isdir(spikes_folder):
        t64file_list = glob.glob(os.path.join(spikes_folder, '*nt%d_*.t64' % tetrode))
        return len(t64file_list)
    else:
        logger.warning("%s is not a valid directory")
        return None
    
def get_velocity_filtered_n_spike(session_folder, mts_vers, scheme, tetrode, unit_index, autocurated, t64file = None,
                                 time_dict = None, vel_thre = 5, abs_vel = True):
    """Get the number of units after velocity filtering \n
    Unit can be manully clustered by MClust (read from .t64 file) \n
    or sorted by MountainSort 5 (use get_mts_unit_timestamps to get unit timestamps) \n
    Used code from plot_firing_fields in plot_helpers.py \n
    time_dict: has systime_creation, spikegadgets_timestamp_creation, and spikegadgets_timestamps \n
    abs_vel: Use absolute value of rat's velocity for filtering
    """
    if time_dict is None:
        # get system_time_at_creation, spikeGadgets_timestamp_at_creation
        # and spikegadgets_timestamps from the dat file
        time_dict = read_timestamps_dat(session_folder, skip_timestamps = False)

    system_time_at_creation = time_dict['systime_creation']
    spikegadgets_timestamp_at_creation = time_dict['spikegadgets_timestamp_creation']
    spikegadgets_timestamps = time_dict['spikegadgets_timestamps']

    experiment_vars_system_time = get_experiment_vars_df(session_folder)
    experiment_vars = experiment_vars_system_time.copy() #modifications to the data or indices of the copy will not be reflected in the original object

    #ros_timestamps = np.polyval(p, experiment_vars['Time'])
    ros_timestamps = experiment_vars['Time']
    # Set start time of trodes data to be 0
    ros_timestamps = ros_timestamps - system_time_at_creation/1000

        # get timestamps of spikes of a cell from the t64 file
    if not t64file is None: # spikes manually sorted via MClust
        N = 10
        with open(t64file, 'rb') as file:
            for i in range(N):
                line = next(file).strip()
                print(line)
            spike_timestamps = np.fromfile(file, dtype = '>u8') # unit: tenths of msecs
        file.close()
        logging.info("t64 file read: %s", t64file)

        spike_timestamps = spike_timestamps / 10000 # unit: sec
        spike_timestamps = spike_timestamps - spikegadgets_timestamp_at_creation / 30000 # Set start time of trodes data to be 0
    
    else: # spikes from MountainSort 5
        unit_timestamps = get_mts_unit_timestamps(session_folder, mts_vers, scheme, tetrode, unit_index,
                                                  autocurated, spikegadgets_timestamps)
        logging.info("""Processing a MountainSort 5 sorted unit: tetrode %d unit_index %d
                     Session folder: %s""" % (tetrode, unit_index, session_folder))
        spike_timestamps = (unit_timestamps - spikegadgets_timestamp_at_creation) / 30000

    # Calculate velocity and extract ros data
    ros_ratAngle = np.unwrap(experiment_vars['ratAngle'] + 180, period=360)
    #time = experiment_vars['Time']
    ros_ratAngle_smoothed = savitzky_golay(ros_ratAngle, 999, 5) #smoothed angle data
    ros_velocity = get_velocity(ros_ratAngle_smoothed, ros_timestamps)
    ros_velocity_smoothed = savitzky_golay(ros_velocity, 99, 3) #smoothed velocity

        # interpolate behavioural data on spikes time stamps
    spike_velocity = np.interp(spike_timestamps, ros_timestamps, ros_velocity_smoothed)

    if abs_vel == True:
        spike_bool_filter = abs(spike_velocity) > vel_thre
    else:
        spike_bool_filter = spike_velocity > vel_thre
    
    velocity_filtered_n_unit = len(spike_timestamps[spike_bool_filter])
    return velocity_filtered_n_unit


def get_new_ratAngle_landmark(ros_ratAngle, ros_landmarkAngle):
    '''Both ros_ratAngle and ros_landmarkAngle are unwrapped \n
    (can be beyond the range of 0-360)'''
    landmarkAngle_rounded = np.zeros(len(ros_landmarkAngle))
    for i in range(len(ros_landmarkAngle)):
        if  ros_landmarkAngle[i] >= 0:
            landmarkAngle_rounded[i] = ros_landmarkAngle[i] % 360
        else:
            landmarkAngle_rounded[i] = ros_landmarkAngle[i] % (-360)

    ratAngle_rounded = ros_ratAngle % 360
    new_ros_ratAngle_landmark = (ratAngle_rounded + landmarkAngle_rounded) % 360
    return new_ros_ratAngle_landmark



def merge_heatmaps(anim, startDate, endDate):
    """ anim example: '10'
    start_date and end_date format: 'YYMMDD'"""
    
    if anim == '10':
        NC4anim = 'NC40010'
    
    merger = PdfMerger()
    path = 'plots/'
    plots = os.listdir(path)
    plots = sorted(plots)
    for plot in plots:
        split = plot.split('_')
        anim = split[0]
        date = split[1]
        plotType = split[2]
    
        if anim == NC4anim and date >= startDate and date <= endDate and plotType == 'heatmap':
            plot = path + plot
            print(plot)
            merger.append(plot)
    
    merger.write(path + NC4anim + '_heatmaps_' + startDate + '-' + endDate + '.pdf')
    merger.close()


def get_velocity(angle, time):
    """Angle is unwrapped"""
    angular_velocity = []
    angular_velocity.append(0)
    for i in range(len(angle)-1):
        angular_velocity.append( (angle[i+1] - angle[i]) / (time[i+1]-time[i]) )
    return np.array(angular_velocity)

# savitzky_golay() moved to signal_processing.py

# # Source: https://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGaolay
# # Edited
# def savitzky_golay(y, window_size, order, deriv=0, rate=1):
#     r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
#     The Savitzky-Golay filter removes high frequency noise from data.
#     It has the advantage of preserving the original shape and
#     features of the signal better than other types of filtering
#     approaches, such as moving averages techniques.
#     Parameters
#     ----------
#     y : array_like, shape (N,)
#         the values of the time history of the signal.
#     window_size : int
#         the length of the window. Must be an odd integer number.
#     order : int
#         the order of the polynomial used in the filtering.
#         Must be less then `window_size` - 1.
#     deriv: int
#         the order of the derivative to compute (default = 0 means only smoothing)
#     Returns
#     -------
#     ys : ndarray, shape (N)
#         the smoothed signal (or it's n-th derivative).
#     Notes
#     -----
#     The Savitzky-Golay is a type of low-pass filter, particularly
#     suited for smoothing noisy data. The main idea behind this
#     approach is to make for each point a least-square fit with a
#     polynomial of high order over a odd-sized window centered at
#     the point.
#     Examples
#     --------
#     t = np.linspace(-4, 4, 500)
#     y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
#     ysg = savitzky_golay(y, window_size=31, order=4)
#     import matplotlib.pyplot as plt
#     plt.plot(t, y, label='Noisy signal')
#     plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
#     plt.plot(t, ysg, 'r', label='Filtered signal')
#     plt.legend()
#     plt.show()
#     References
#     ----------
#     .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
#        Data by Simplified Least Squares Procedures. Analytical
#        Chemistry, 1964, 36 (8), pp 1627-1639.
#     .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
#        W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
#        Cambridge University Press ISBN-13: 9780521880688
#     """

#     try:
#         window_size = np.abs(int(window_size))
#         order = np.abs(int(order))
#     #except ValueError, msg:
#     except ValueError:
#         raise ValueError("window_size and order have to be of type int")
#     if window_size % 2 != 1 or window_size < 1:
#         raise TypeError("window_size size must be a positive odd number")
#     if window_size < order + 2:
#         raise TypeError("window_size is too small for the polynomials order")
#     order_range = range(order+1)
#     half_window = (window_size -1) // 2
#     # precompute coefficients
#     b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
#     m = np.linalg.pinv(b).A[deriv] * rate**deriv * math.factorial(deriv)
#     # pad the signal at the extremes with
#     # values taken from the signal itself
#     firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
#     lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
#     y = np.concatenate((firstvals, y, lastvals))
#     return np.convolve( m[::-1], y, mode='valid')
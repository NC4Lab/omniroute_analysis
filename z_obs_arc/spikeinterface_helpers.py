import numpy as np
import os
import logging
from probeinterface import Probe, ProbeGroup
from probeinterface.plotting import plot_probe, plot_probe_group
from probeinterface import generate_tetrode
import spikeinterface.full as si
import spikeinterface.preprocessing as spre
from spikeinterface.exporters import export_to_phy, export_report
from spikeinterface.postprocessing import compute_spike_amplitudes, compute_principal_components, compute_correlograms
from spikeinterface.qualitymetrics import compute_quality_metrics
import spikeinterface.sorters as ss
#import mountainsort4 as ms4
import mountainsort5 as ms5
import math
from rosbags.highlevel import AnyReader as RosBagReader
from datetime import datetime
from helpers import *

logger = logging.getLogger(__name__)
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)


# parameters from https://github.com/magland/mountainsort4/blob/main/examples/mountainsort4_example1.py
# sorting = ms4.mountainsort4(
#         recording=recording,
#         detect_sign=-1,
#         clip_size=50,
#         adjacency_radius=20,
#         detect_threshold=3,
#         detect_interval=10,
#         num_workers=None,
#         verbose=True,
#         use_recording_directly=False
#     )


# preprocessing parameters set to be same as Knierim Lab
preprocess_params = {
            # 'bandpass_filter': False,
            # 'bandpass_min': 300,
            # 'bandpass_max':3000,
            'bandpass_filter': True, # different from Knierim Lab
            'bandpass_min': 600, # different from Knierim Lab
            'bandpass_max':6000, # different from Knierim Lab
            'whiten': True,
            # mask parameters are not used now (Jan 2024)
            'mask_artifacts': True,
            'mask_threshold': 6,
            'mask_chunk_size': 2000,
            'mask_num_write_chunks': 150
            }

# parameters set to be same as Knierim Lab except for num_workers
mts_params = {
    'adjacency_radius': -1,
    'detect_sign': 1,
    'detect_threshold': 4,
    'detect_interval': 10,
    'clip_size': 40,
    #'num_workers': 8,
    'num_workers': None,
    'num_features': 10, # not used due to error while running with this param
    'max_num_clips_for_pca': 1000 # not used due to error while running with this param
}

# Automatic curation parameters from Knierim Lab msort_function except for the isi_violations_ratio thresh
curation_params = {
            'firing_rate_thresh': 0.5,
            'isolation_thresh': 0.90,
            'noise_overlap_thresh': 0.3,
            'peak_snr_thresh': 1.5,
            'isi_violations_ratio_thresh' : 5.6
}

'''
From Knierim Lab code
firing_rate_thresh : float64
    (Optional) firing rate must be above this
isolation_thresh : float64
    (Optional) isolation must be above this
noise_overlap_thresh : float64
    (Optional) noise_overlap_thresh must be below this
peak_snr_thresh : float64
    (Optional) peak snr must be above this
'''

selected_metric_names = ['num_spikes',
 'firing_rate',
 'presence_ratio',
 'snr',
 'isi_violation',
 'rp_violation',
 'sliding_rp_violation',
 'amplitude_cutoff',
 'amplitude_median',
 'isolation_distance',
 'l_ratio',
 'd_prime',
 'nearest_neighbor',
 'nn_isolation',
 'nn_noise_overlap']
# removed 'drift' as the drift metrics require the `spike_locations` waveform extension. 

# autocuration query
autocuration_query = f"firing_rate > {curation_params['firing_rate_thresh']} \
            & snr > {curation_params['peak_snr_thresh']} \
            & nn_isolation > {curation_params['isolation_thresh']} \
            & nn_noise_overlap < {curation_params['noise_overlap_thresh']} \
            & isi_violations_ratio < {curation_params['isi_violations_ratio_thresh']}"


def generate_tetrode_groups(n_tt=16, tt_spacing=50, plot=False):
    """Generate probe groups for tetrodes

    Uses example code from probeinterface:
    https://probeinterface.readthedocs.io/en/main/examples/ex_07_probe_generator.html#sphx-glr-examples-ex-07-probe-generator-py
    """
    probegroup = ProbeGroup()
    for i in range(n_tt):
        tetrode = generate_tetrode()
        tetrode.move([i * tt_spacing, 0])
        probegroup.add_probe(tetrode)
    probegroup.set_global_device_channel_indices(np.arange(n_tt*4))

    # df = probegroup.to_dataframe()
    if plot:
        plot_probe_group(probegroup, with_channel_index=True, same_axes=True)
    
    return probegroup

def slice_recordings_by_tt(rec, tt_list):
    ch = []
    for tt in tt_list:
        print(tt)
        ch = np.append(ch, np.char.mod('%d', (4*tt-1) + np.array([0,1,2,3])))
    
    return rec.channel_slice(channel_ids = ch)


def preprocess_1TT(rec, tetrode):
    """Cut one tetrode and do lazy preprocessing"""
    tt_to_sort = [tetrode] ## These are 1-indexed, i.e. starting from ntrode 1 in Trodes

    np.array(tt_to_sort)-1

    # Cut to these tetrodes
    channel_ids_to_sort = rec.get_channel_ids()[np.isin(rec.get_channel_groups(), np.array(tt_to_sort)-1)]
    rec_to_sort = rec.channel_slice(channel_ids=channel_ids_to_sort)

    # Source: https://github.com/flatironinstitute/mountainsort5
    # https://github.com/flatironinstitute/mountainsort5/blob/main/examples/scheme1/toy_example.py
    # Should bandpass filter and whiten the recording for MountainSort5 to work
    # lazy preprocessing?
    #rec_filtered = spre.bandpass_filter(rec_to_sort, freq_min=300, freq_max=6000)
    if preprocess_params['bandpass_filter']:
        rec_filtered = spre.bandpass_filter(rec_to_sort, freq_min=preprocess_params['bandpass_min'], 
                                            freq_max=preprocess_params['bandpass_max'])
        if preprocess_params['whiten']:
            rec_preprocessed: si.BaseRecording = spre.whiten(rec_filtered, dtype='float32')
            return rec_preprocessed
        else:
            return rec_filtered
    elif preprocess_params['whiten']:
        rec_preprocessed: si.BaseRecording = spre.whiten(rec_to_sort, dtype='float32')
        return rec_preprocessed
    else:
        return rec_to_sort



def mts_save(session_folder, rec_preprocessed, mts_vers, scheme, tetrode):
    """Run MountainSort 5 on a preprocessed rec file with scheme 1 or 2,
    , save the sorting object to lil_swallow and return the object
    See following pages for scheme documentation
    https://github.com/flatironinstitute/mountainsort5/blob/main/docs/scheme1.md
    https://github.com/flatironinstitute/mountainsort5/blob/main/docs/scheme2.md
    """
    if mts_vers == 'mts5':
        if scheme == 1:
            logging.info("""Starting MountainSort5 Scheme 1 on session folder:
                            %s
                            tetrode %d""" % (session_folder, tetrode))
            mts_sorting = ms5.sorting_scheme1(
                rec_preprocessed,
                sorting_parameters=ms5.Scheme1SortingParameters(detect_channel_radius=20)
            )
        if scheme == 2:
            logging.info("""Starting MountainSort5 Scheme 2 on session folder:
                            %s
                            tetrode %d""" % (session_folder, tetrode))
            mts_sorting = ms5.sorting_scheme2(
                rec_preprocessed,
                sorting_parameters=ms5.Scheme2SortingParameters(
                phase1_detect_channel_radius=20, detect_channel_radius=20)
            )
    
    elif mts_vers == 'mts4':
        logging.info("""Starting MountainSort4 on session folder:
                            %s
                            tetrode %d""" % (session_folder, tetrode))
        # ms4.mountainsort4: https://github.com/magland/mountainsort4/blob/main/examples/mountainsort4_example1.py
        # mts_sorting = ms4.mountainsort4( # not working
        mts_sorting = ss.run_mountainsort4(
            recording = rec_preprocessed,
            #timeseries=next_step_input,
            #geom=geom_fname,
            #firings_out=firings_out,
            adjacency_radius=mts_params['adjacency_radius'],
            detect_sign=mts_params['detect_sign'],
            detect_threshold=mts_params['detect_threshold'],
            detect_interval=mts_params['detect_interval'],
            clip_size=mts_params['clip_size'],
            num_workers=mts_params['num_workers'],
            #num_features=mts_params['num_features'], # this gives an error: mountainsort4() got an unexpected keyword argument 'num_features'
            #max_num_clips_for_pca=mts_params['max_num_clips_for_pca'] # this gives an error: mountainsort4() got an unexpected keyword argument 'max_num_clips_for_pca'
        )
    
    # Save the sorting object to lil_swallow
    mts_sorting.save(folder = get_mts_object_folder(session_folder, mts_vers, scheme, tetrode))
    return mts_sorting

def mts_export_report(session_folder, rec_preprocessed, mts_sorting, mts_vers, scheme, tetrode, autocurated=False): 
    """Export a spike sorting report and save it to lil_swallow"""
    # Source: https://spikeinterface.readthedocs.io/en/latest/modules/exporters.html
    # some computations are done before to control all options
    wf_folder = remove_output_folder('mts_wf')
    # the waveforms are sparse so it is faster to export to phy
    we = si.extract_waveforms(rec_preprocessed, mts_sorting, wf_folder, sparse=True)
    compute_spike_amplitudes(we)
    compute_correlograms(we)
    compute_quality_metrics(we, metric_names=['snr', 'isi_violation', 'presence_ratio'])

    # the export process
    report_folder = get_report_folder(session_folder, mts_vers, scheme, tetrode, autocurated)
    export_report(we, output_folder=report_folder, format='png', show_figures=True, peak_sign='pos')


def load_mts(session_folder, mts_vers, scheme, tetrode):
    """Load a saved MountainSort object with given scheme and tetrode"""
    mts_object_folder = get_mts_object_folder(session_folder, mts_vers, scheme, tetrode)
    if not os.path.isdir(mts_object_folder):
        logging.error('MountainSort object folder does not exist %s', mts_object_folder)

    else:
        mts_sorting = si.load_extractor(mts_object_folder)
        logging.info('MountainSort object loaded from %s', mts_object_folder)
        return mts_sorting


def autocurate_1TT(session_folder, rec_preprocessed, mts_sorting, wf, mts_vers, scheme, tetrode):
    '''Autocurate one tetrode based on quality metrics \n
    wf: waveform extractor (obtained by si.extract_waveforms or si.load_waveforms) \n
    Return an array of  the ids of the units that are kept after autocuration'''
    # compute PCA scores
    pc = si.compute_principal_components(wf, n_components=3, load_if_exists=True)

    # compute quality metrics
    qm = si.compute_quality_metrics(wf, metric_names=selected_metric_names, verbose=True,  qm_params=si.get_default_qm_params())
    
    # save quality metrics as a pkl file
    save_quality_metrics(session_folder, mts_vers, scheme, tetrode, qm, autocurated=False)

    kept_units = qm.query(autocuration_query)
    # save the quality metrics of the kept units
    save_quality_metrics(session_folder, mts_vers, scheme, tetrode, kept_units, autocurated=True)
    kept_unit_ids = kept_units.index.values

    mts_sorting_autocurated = mts_sorting.select_units(kept_unit_ids)
    logger.info(f"Tetrode: {tetrode}")
    logger.info(f"Number of units before curation: {len(mts_sorting.get_unit_ids())}")
    logger.info(f"Number of units after curation: {len(mts_sorting_autocurated.get_unit_ids())}")

    if len(kept_unit_ids) > 0:
        #Save the autocurated mountainsort sorting object, spike vector and report if the number of kept units is greater than zero
        mts_sorting_autocurated_folder = get_mts_object_folder(session_folder, mts_vers, scheme, tetrode, autocurated=True)
        mts_sorting_autocurated.save(folder = mts_sorting_autocurated_folder)
        spike_vector_autocurated = mts_sorting_autocurated.to_spike_vector()
        save_spike_vector(session_folder, spike_vector_autocurated, mts_vers, scheme=scheme, tetrode=tetrode, autocurated=True)
        mts_export_report(session_folder, rec_preprocessed, mts_sorting_autocurated, mts_vers, scheme=scheme, tetrode=tetrode, autocurated=True)
    
    logger.info('Returning an array of ids of kept units')
    return kept_unit_ids


def postprocess_1TT(session_folder, rec, mts_vers, scheme, tetrode):
    '''Postprocess one tetrode, including waveform extraction, quality metrics computation \n
    and autocuration \n
    Return an array of the ids of the units that are kept after autocuration
    '''
    logger.info('Processing tetrode %d' % tetrode)
    mts_object_folder = get_mts_object_folder(session_folder, mts_vers, scheme, tetrode)
    # Load saved mts object
    mts_sorting = si.load_extractor(mts_object_folder)

    # get preprocessed rec (similar to rec_filt in spikeinterface_testing.ipynb)
    rec_preprocessed =  preprocess_1TT(rec, tetrode)

    wf_folder = get_wf_folder(session_folder, mts_vers, scheme, tetrode)


    if not os.path.isdir(wf_folder):
    # Only extract waveforms if not already done
    # Need to clear folder to re-extract waveforms
    # Use si.load_waveforms to load existed waveform extractor (check next section)
        
        # Remove excess spikes from the spike trains. 
        # Excess spikes are the ones exceeding a recording number of samples, for each segment.
        # Reference: https://spikeinterface.readthedocs.io/en/latest/api.html#module-spikeinterface.curation
        mts_sorting_excess_removed = si.remove_excess_spikes(mts_sorting, rec_preprocessed)
        wf = si.extract_waveforms(rec_preprocessed, mts_sorting_excess_removed, folder=wf_folder, 
                                  max_spikes_per_unit=None, progress_bar=True, 
                                  n_jobs=4, overwrite=True)
        
        kept_unit_ids = autocurate_1TT(session_folder, rec_preprocessed, mts_sorting, wf, mts_vers, scheme, tetrode)
        return kept_unit_ids
        
        
    else:
        logger.info('Waveform folder already exists. Waveforms not re-extracted.')
        # Todo: rewrite following code

        # attempt to get the quality metrics of the kept units that were saved
        #kept_units = get_quality_metrics_df(session_folder, mts_vers, scheme, tetrode, autocurated=True)

        #if kept_units == None: # a pkl file for kept units quality metrics doesn't exist
            # load waveform extractor from the folder
        #    wf = si.load_waveforms(wf_folder)
            # autocuration
        #    kept_unit_ids = autocurate_1TT(session_folder, rec_preprocessed, mts_sorting, wf, mts_vers, scheme, tetrode)
        
        #else:
        #    kept_unit_ids = kept_units.index.values
        #return kept_unit_ids

def reautocurate_1TT(session_folder, rec_preprocessed, mts_sorting, mts_vers, scheme, tetrode):
    '''Autocurate one tetrode based on quality metrics with updated autocuration query \n
    (added isi_violations_rato_thresh)\n
    Quality metrics of units from each tetrode already saved \n
    wf: waveform extractor (obtained by si.extract_waveforms or si.load_waveforms) \n
    Return an array of  the ids of the units that are kept after autocuration'''

    qm = get_quality_metrics_df(session_folder, mts_vers, scheme, tetrode, autocurated=False)
    
    kept_units = qm.query(autocuration_query)
    # save the quality metrics of the kept units
    save_quality_metrics(session_folder, mts_vers, scheme, tetrode, kept_units, autocurated=True)
    kept_unit_ids = kept_units.index.values

    mts_sorting_autocurated = mts_sorting.select_units(kept_unit_ids)
    logger.info(f"Tetrode: {tetrode}")
    logger.info(f"Number of units before curation: {len(mts_sorting.get_unit_ids())}")
    logger.info(f"Number of units after curation: {len(mts_sorting_autocurated.get_unit_ids())}")

    if len(kept_unit_ids) > 0:
        #Save the autocurated mountainsort sorting object, spike vector and report if the number of kept units is greater than zero
        mts_sorting_autocurated_folder = get_mts_object_folder(session_folder, mts_vers, scheme, tetrode, autocurated=True)
        mts_sorting_autocurated.save(folder = mts_sorting_autocurated_folder)
        spike_vector_autocurated = mts_sorting_autocurated.to_spike_vector()
        save_spike_vector(session_folder, spike_vector_autocurated, mts_vers, scheme=scheme, tetrode=tetrode, autocurated=True)
        mts_export_report(session_folder, rec_preprocessed, mts_sorting_autocurated, mts_vers, scheme=scheme, tetrode=tetrode, autocurated=True)
    
    logger.info('Returning an array of ids of kept units')
    return kept_unit_ids

def save_sync_ts(session_folder):
    ''' Save synchronizing timestamps trodes_sync_ts and ros_sycn_ts'''
    rec_file_to_process = get_merged_rec(session_folder)
    rec = si.read_spikegadgets(rec_file_to_process)
    date = get_folder_date(session_folder)

    dio_folder = os.path.join(session_folder, 'extracted', date+'_merged.DIO')
    if os.path.exists(dio_folder) == False:
        export_dio(session_folder, rec_files=rec_file_to_process)
    else:
        logging.info("%s exists, not re-exported" % dio_folder)
    
    dio = get_dio(session_folder, rec_file_to_process, 2).dio # sync signal is on digital input channel 2
    trodes_sync_ts = np.array(dio[dio.state==True].index, 'f')

    ## Scale timestamps to seconds
    trodes_sync_ts = np.divide(trodes_sync_ts, rec.get_sampling_frequency())

    ## Load timestamps from /sync channel in ros data bag
    databags = get_ros_data_bag_files(session_folder)

    ros_sync_ts = np.array([], 'f')
    ros_sync_msg_ts = np.array([], 'f')
    with RosBagReader([Path(databags[0])]) as reader:
        connections = [x for x in reader.connections if x.topic=='/sync']
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            ros_sync_ts = np.append(ros_sync_ts, msg.data.sec + msg.data.nanosec*1.0e-9)
            ros_sync_msg_ts = np.append(ros_sync_msg_ts, timestamp)

    ## Scale timestamps to seconds
    # ros_sync_ts = np.divide(ros_sync_ts, 1.0)
    ros_sync_msg_ts = np.divide(ros_sync_msg_ts, 1.0e9)

    save_trodes_sync_ts_array(session_folder, trodes_sync_ts)
    save_ros_sync_ts_array(session_folder, ros_sync_ts)


import numpy as np
import os
import logging
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from helpers import *

logger = logging.getLogger(__name__)
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)


freq_min = 5
freq_max = 11
resample_rate = 500

def load_theta_ts(session_folder):
        theta_folder = get_theta_folder(session_folder)
        theta_ts_file = os.path.join(theta_folder, 'theta_ts.npy')

        if os.path.isfile(theta_ts_file):
            theta_freq_filtered = np.load(theta_ts_file)
            return theta_freq_filtered
        else:
            logger.error("%s is not a valid file" % theta_ts_file)

def save_theta_freq_filtered(session_folder, show_cc_plot = False, remove_bad_tetrodes = False,
                    bad_tetrodes = None):
    ''' Save theta_freq_filtered ndarray to the theta folder \n
        Uses theta_hilbert in the folder so theta_hilbert need to be saved first \n
        show_cc_plot: whether to show the cross-correlation plot of theta amplitudes \n
        remove_bad_tetrodes: whether to remove user-defined bad tetrodes; For example, for rat 15 \n
        tetrode 14 should be removed as it is known to be not connected \n
        bad_tetrodes: user_defined bad tetrodes. e.g. 14, [2, 10] (should be 1-indexed tetrode numbers)
    '''

    theta_folder = get_theta_folder(session_folder)
    theta_hilbert_file = os.path.join(theta_folder, 'theta_hilbert.npy')
    if os.path.isfile(theta_hilbert_file):
        theta_hilbert = np.load(theta_hilbert_file)
    else:
        logger.error("%s is not a valid file" % theta_hilbert_file)

    # Theta amp using Hilbert transform
    theta_amp = np.abs(theta_hilbert)
    # Theta phase using Hilbert transform
    theta_phase = np.unwrap(np.angle(theta_hilbert), axis=0)

    zscore_theta_amp = stats.zscore(theta_amp, axis=0)

    # Evaluate theta frequency
    theta_freq = np.gradient(theta_phase, axis=0) * resample_rate / (2 * np.pi)
   
   # Eliminate theta freq when amp is low
    theta_freq_filtered = theta_freq.copy()
    theta_freq_filtered[zscore_theta_amp < 0.5] = np.nan

    # Compute cross-correlation of theta amplitudes to find bad channels
    theta_amp_corr = np.corrcoef(theta_amp.T)
    theta_amp_corr[np.isnan(theta_amp_corr)] = 0
    theta_amp_corr[np.isinf(theta_amp_corr)] = 0

    if show_cc_plot:
        logger.info("Plotting cross-correlation of theta amplitudes")
        fig = px.imshow(theta_amp_corr)
        fig.show()

    # Find bad channels
    avg_corr = (np.sum(theta_amp_corr, axis=0)-1)/(theta_amp_corr.shape[0]-1) # Exclude self-correlation
    logger.info(f"Average correlation: {avg_corr}")

    bad_channels = np.where(avg_corr < 0.2)[0]
    logger.info(f"Bad channels: {bad_channels}")

    # Set bad channels to nan
    theta_freq_filtered[:, bad_channels] = np.nan

    # Eliminate theta freq outside filter limits
    theta_freq_filtered[theta_freq_filtered < freq_min] = np.nan
    theta_freq_filtered[theta_freq_filtered > freq_max] = np.nan

    # Eliminate time points where there are less than 3 good channels
    bad_times = np.sum(np.isnan(theta_freq_filtered), axis=1) > (theta_freq_filtered.shape[1] - 3)
    theta_freq_filtered[bad_times, :] = np.nan

    # Remove the bad tetrodes that are defined by the user
    if remove_bad_tetrodes:
        if not isinstance(bad_tetrodes, int) and len(bad_tetrodes) > 1:
            bad_tetrodes = np.array(bad_channels)
        theta_freq_filtered_selected = np.delete(theta_freq_filtered, bad_tetrodes-1, axis=1) # -1 turns 1-indexed to 0-indexed
        np.save(os.path.join(theta_folder, 'theta_freq_filtered_selected.npy'), theta_freq_filtered_selected)
    else:
        np.save(os.path.join(theta_folder, 'theta_freq_filtered.npy'), theta_freq_filtered)

def load_theta_freq_filtered(session_folder, bad_tetrodes_removed = False):
    theta_folder = get_theta_folder(session_folder)
    if bad_tetrodes_removed:
        theta_freq_filtered_file = os.path.join(theta_folder, 'theta_freq_filtered_selected.npy')
    else:
        theta_freq_filtered_file = os.path.join(theta_folder, 'theta_freq_filtered.npy')

    if os.path.isfile(theta_freq_filtered_file):
        theta_freq_filtered = np.load(theta_freq_filtered_file)
        return theta_freq_filtered
    else:
        logger.error("%s is not a valid file" % theta_freq_filtered_file)

def get_theta_freq_median(session_folder, bad_tetrodes_removed = False):
    theta_freq_filtered = load_theta_freq_filtered(session_folder, bad_tetrodes_removed)
    theta_freq_median = np.nanmedian(theta_freq_filtered, axis = 1)
    return theta_freq_median

def plot_theta_median(theta_ts, theta_freq_filtered):
    theta_freq_median = np.nanmedian(theta_freq_filtered, axis = 1)

    fig = go.Figure()
    for i in range(theta_freq_filtered.shape[1]):
        fig.add_trace(go.Scatter(x=theta_ts, y=theta_freq_filtered[:, i], mode='lines', name=f'Channel {i}', line=dict(color='gray')))

    fig.add_trace(go.Scatter(x=theta_ts, y=theta_freq_median, mode='lines', name='Median', line=dict(color='red', width=3)))

    fig.show()

def get_theta_freq_velocity_df(session_folder, bad_tetrode_removed, sync_p, remove_nan, save_df):
    # aligned theta timestamps in ROS system time
    theta_ts = load_theta_ts(session_folder)
    theta_ts_aligned = np.polyval(sync_p, theta_ts)
    
    experiment_vars = get_experiment_vars_df(session_folder)
    ros_timestamps = experiment_vars['Time'] # in system time

    # Calculate velocity and extract ros data
    ros_ratAngle = np.unwrap(experiment_vars['ratAngle'] + 180, period=360)
    #time = experiment_vars['Time']
    ros_ratAngle_smoothed = savitzky_golay(ros_ratAngle, 999, 5) #smoothed angle data
    ros_lab_velocity = get_velocity(ros_ratAngle_smoothed, ros_timestamps)
    ros_lab_velocity_smoothed = savitzky_golay(ros_lab_velocity, 99, 3) #smoothed velocity

    ros_soundDegree = experiment_vars['soundPhase']/np.pi*180
    ros_soundDegree = ros_soundDegree.to_numpy()
    ros_soundDegree_smoothed = savitzky_golay(ros_soundDegree, 999, 5)
    ros_sound_velocity = get_velocity(ros_soundDegree_smoothed, ros_timestamps)
    ros_sound_velocity_smoothed = savitzky_golay(ros_sound_velocity, 99, 3)

    # use rat angle with regard to landmark to compute velocity in the landmark frame
    ros_landmarkAngle = experiment_vars['landmarkAngle']
    ros_ratAngle_landmark = get_new_ratAngle_landmark(ros_ratAngle, ros_landmarkAngle) # in range 0-360
    ros_ratAngle_landmark_unwrapped = np.unwrap(ros_ratAngle_landmark, period = 360)
    ros_ratAngle_landmark_unwrapped_smoothed = savitzky_golay(ros_ratAngle_landmark_unwrapped, 999, 5)
    ros_landmark_velocity = get_velocity(ros_ratAngle_landmark_unwrapped_smoothed, ros_timestamps)
    ros_landmark_velocity_smoothed = savitzky_golay(ros_landmark_velocity, 99, 3)

    ros_soundFrequency = experiment_vars['soundFrequency']
    ros_sound_gain = experiment_vars['soundGain']
    ros_landmark_gain = experiment_vars['landmarksGain']
    ros_boomAngle_landmark = experiment_vars['boomAngle_wrt_landmark']
    ros_laps = (ros_ratAngle - ros_ratAngle[0])/360.0


    # interpolate behavioural data on theta timestamps
    theta_time = theta_ts_aligned - ros_timestamps[0]
    theta_ratAngle = np.interp(theta_ts_aligned, ros_timestamps, ros_ratAngle)
    theta_soundDegree = np.interp(theta_ts_aligned, ros_timestamps, ros_soundDegree)
    theta_soundFrequency = np.interp(theta_ts_aligned, ros_timestamps, ros_soundFrequency)
    theta_lab_velocity_smoothed = np.interp(theta_ts_aligned, ros_timestamps, ros_lab_velocity_smoothed)
    theta_sound_velocity_smoothed = np.interp(theta_ts_aligned, ros_timestamps, ros_sound_velocity_smoothed)
    theta_landmark_velocity_smoothed = np.interp(theta_ts_aligned, ros_timestamps, ros_landmark_velocity_smoothed)
    theta_boomAngle_landmark = np.interp(theta_ts_aligned, ros_timestamps, ros_boomAngle_landmark)
    theta_ratAngle_landmark = np.interp(theta_ts_aligned, ros_timestamps, ros_ratAngle_landmark)
    theta_sound_gain = np.interp(theta_ts_aligned, ros_timestamps, ros_sound_gain)
    theta_laps = np.interp(theta_ts_aligned, ros_timestamps, ros_laps)

    theta_freq_median = get_theta_freq_median(session_folder, bad_tetrode_removed)
    d = {'time': theta_time,
     'laps': theta_laps,
     'theta_freq': theta_freq_median,
     'lab_velocity': theta_lab_velocity_smoothed,
     'sound_velocity': theta_sound_velocity_smoothed,
     'landmark_velocity': theta_landmark_velocity_smoothed 
    }

    theta_velocity_df = pd.DataFrame(data=d)
    
    if remove_nan:
        theta_velocity_nanremoved_df = theta_velocity_df.dropna(axis=0, how = 'any')
        if save_df:
            theta_folder = get_theta_folder(session_folder)
            theta_velocity_nanremoved_df.to_pickle(os.path.join(theta_folder, 'theta_velocity_nanremoved_df.pkl'))
        return theta_velocity_nanremoved_df
    
    else:
        if save_df:
            theta_velocity_df.to_pickle(os.path.join(theta_folder, 'theta_velocity_df.pkl'))
        return theta_velocity_df
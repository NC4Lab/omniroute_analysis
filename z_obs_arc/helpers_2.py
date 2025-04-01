import numpy as np
import matplotlib.pyplot as plt

def get_spike_files(data_path, rat, date, tetrode, cluster):
    # Not used since Aug 18, 2023
    # Kept for older jupyter notebooks

    # rat example: '10'
    # date format: 'YYMMDD'
    # tetrode example: '1', '12'
    # cluster example: '1'
    
    if rat == '10':
        NC4rat = 'NC40010'
        
    if len(cluster) == 1:
        cluster = '0' + cluster
    
    files = []
    files.append(data_path + NC4rat + '/' + date + '_training/' + 'extracted/' + date + '_merged.spikes/' + date + '_merged.spikes_nt' + tetrode + '_' + cluster + '.t64')
    files.append(data_path + NC4rat + '/' + date + '_training/' + 'extracted/' + date + '_merged.spikes/' + date + '_merged.spikes_nt' + tetrode + '.dat')
    
    return files
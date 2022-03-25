import os
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import pickle
from itertools import islice
import sys
from multiprocessing import Pool
import time
# path to outpatients: "/alto/shared/H2017A_Ddorf_27102020"
# path to inpatients: "/alto/shared/wearable/H2017A_Ddorf"
def get_interval_data(df_in, interval_length_in_seconds, patient_id):
    all_intervals = list()
    num_skipped_intervals = 0
    
    interval   = list()
    timestamps = list()  
    
    for k, row in df_in.iterrows():
        interval.append(list(row[1:]))
        timestamps.append(datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")) # use ORIGINAL timestamps
    
        if k>=1:
            if timestamps[-1].hour != timestamps[-2].hour or timestamps[-1].day>=(timestamps[-2].day)+1: 
                deltas = np.array([(a-b).total_seconds() for a, b in zip(timestamps[1:],timestamps[:-1])])
                start_timestamp = timestamps[0]
                end_timestamp   = timestamps[-2]

                is_scc, smallest_before_time, smallest_after_time, diag, cat, vitc = None, None, None, None, None, None

                all_intervals.append((np.array(interval[:-1]), start_timestamp, end_timestamp, 
                                      patient_id, is_scc, smallest_before_time, smallest_after_time, diag, cat, vitc,
                                      np.max(deltas)))


                interval   = list()
                timestamps = list()
                interval.append(list(row[1:]))
                timestamps.append(datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")) # use ORIGINAL timestamps

    return all_intervals, num_skipped_intervals


def fix_filename(fn):
    """ e.g.
    turn H2017A_01052_191121_m59_combined.csv
    into H2017A_1052_191121_m59_combined.csv
    """
    
    splits = fn.split('_')
    splits[1] = splits[1][-4:]
    return '_'.join(splits)

def get_patient_id(fn):
    return fn.split('_')[1]

if __name__=='__main__':
    wearable_dir = 'path to data directory'
    combined_files = [fn for fn in os.listdir(wearable_dir) if '_combined_no_ibi.csv' in fn]
    
    dataset_dir = 'name of the output file'
    output_dir = os.path.join(wearable_dir, dataset_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
        
    interval_length_in_seconds = 60*60 # SECONDS (5 min * 60 s/min)
    for filename in combined_files:

        print('working on ...', filename)

        ### input file
        filepath = os.path.join(wearable_dir, filename)


        ### output file
        filename_fixed = fix_filename(filename)
        output_path = os.path.join(wearable_dir, dataset_dir, filename_fixed.replace('_combined_no_ibi.csv', '.pkl'))

        if not os.path.exists(output_path):

            ### extract file
            df_in = pd.read_csv(filepath)
            patient_id = get_patient_id(filename)
            patient_id = '%4d' % int(patient_id) # fix ...
            all_intervals, num_skipped_intervals = get_interval_data(df_in, 
                                                                     interval_length_in_seconds,
                                                                     patient_id)

            print('number of accepted intervals: %d, number of skipped intervals: %d' % (len(all_intervals), 
                                                                                         num_skipped_intervals))

            with open(output_path, 'wb') as fileobj:
                pickle.dump([all_intervals, num_skipped_intervals], fileobj)        

import signal_processing as sp
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import ticker
from scipy import signal
from scipy import integrate
import pywt
import time

# MACROS

FILTER_LIST = ['notch', 'butterworth']
EXIT_SUCCESS = 0
DATA_PATH = "../data/net_data/"

def get_activity_in_window(sub, action):

    labels_file = sub + "_" + action + "_Label.csv"
    labels_file = os.path.join(sp.DATASET_PATH, sub, 'Labels', labels_file)
    # Reading the csv
    labels = pd.read_csv(labels_file)
    active_windowed_signal = sp.window_sliding('Status', labels, action)
    if action == 'WAK':
        for i in range(len(active_windowed_signal)):
            active_windowed_signal[i] = np.where(np.isnan(active_windowed_signal[i]), 'R', 'A')     

    # If the signal is half activated or more, dectects it as activated
    act_wind_labels = np.array(['A' if np.count_nonzero(window == 'A') >=
                        (2 * len(window)) / 3 else 'R' for window in active_windowed_signal])

    return act_wind_labels


def define_matrix(windowed_signal, signal_matrix_arr):

    if len(signal_matrix_arr) == 0:
        signal_matrix_arr = windowed_signal
        return signal_matrix_arr
    
    # In case there are more elements in the array
    index = 0
    for window in windowed_signal:
        # Adding rows to the matrixes
        signal_matrix_arr[index] = np.vstack((signal_matrix_arr[index], window))
        index += 1

    return signal_matrix_arr


def export_matrix(matrixes, activity, action):
    
    for i in range(len(activity)):
        if activity[i] == 'R':
            dir = "REST"
        else:
            dir = action
        
        #Creating directory path if it not exists
        path = DATA_PATH + dir
        if not os.path.exists(path):
            os.makedirs(path)

        # obtain the counter of files in the directories
        existing_files = [f for f in os.listdir(path) if f.endswith(".npy")]
        if existing_files:
            # Extract the max counter in the directory
            num = [int(f.split('_')[1].split('.')[0]) for f in existing_files if f.startswith(f"{dir}_")]
            if num:
                actual_count = max(num) + 1  # Continuar con el siguiente número
            else:
                actual_count = 1
        else:
            actual_count = 1  # Si no hay archivos, comenzar desde 0
        
        matrix_name = f"{dir}_{actual_count}.npy"
        np.save(os.path.join(path, matrix_name), matrixes[i])


def main():
    
    # Processing all the subjects
    dir_elements = os.listdir(sp.DATASET_PATH)
    # Selecting only the subjects directories
    subjects = [elem for elem in dir_elements if
                os.path.isdir(os.path.join(sp.DATASET_PATH, elem)) and elem.startswith('Sub')]
    
    # Iterating in all the subjects and actions
    for sub in subjects:
        for action in sp.ACTIONS_LIST:
            semg_df = sp.get_sEMG_data(sub, action)
            # Creating an array for the signal matrixes
            signals_matrix_arr = []
            activity_array = get_activity_in_window(sub, action)
            for muscle in semg_df.columns:   
                # Not counting time column
                if muscle == 'Time':
                    continue
                # Filters
                sp.signal_processing(semg_df, muscle, FILTER_LIST)
                windowed_signal = sp.window_sliding(muscle, semg_df, action)
                # Saving the window in each determinated matrix
                signals_matrix_arr = define_matrix(windowed_signal, signals_matrix_arr)

            #Exporting to npy matrixes
            export_matrix(signals_matrix_arr, activity_array, action)
        print(f"{sub} ended")

    return(EXIT_SUCCESS)

            

if __name__ == "__main__":
    main()

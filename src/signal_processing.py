import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import ticker
from scipy import signal
import glob
import pywt

# MACROS

EXIT_FAILURE = 1
ARGS_N = 4
DATASET_PATH = '/home/avilleira/TFG/dataset/SIAT_LLMD20230404/'
TIME_UNTIL_MOVEMENT = 3 #seconds
MUSCLES_N = 9
FILTER_LIST = ['notch', 'butterworth']

PLT_AMPLITUDE_OFFSET = 0.05
FREQ_SAMPLING = 1920

ENVELOP_FILTER_LENGTH = 100
ENVELOP_LOW_CUT_FREQ = 3.5 #Hz
ENVELOP_HIGH_CUT_FREQ = 40

NOTCH_QUALITY_FACTOR = 100
NOTCH_FREQ_TO_REMOVE = 50

BUTTER_LOW_FREQ = 15 #Hz
BUTTER_HIGH_FREQ = 400 #Hz
BUTTER_ORDER = 7

WVLT_THRESHOLD = 0.08
WVLT_THRHLD_MODE = "soft"
WVLT_PACKET_TYPE = 'db7'
WVLT_LEVEL = 9


def usage():
    """
    Usage error exit
    """

    sys.stderr.write("Usage: python3 signal_plot.py SubXX action muscle/all\n")
    sys.exit(EXIT_FAILURE)


def add_column_to_df(df, column_name, np_arr):
    """
    Adds to the dataframe df a new column with name column_name and data np_arr
    # Parameters:
    :param df: Dataframe of the signal data
    :type: Panda Dataframe
    :param column_name: Name of the new column to add
    :type: string
    :param np_arr: List of all the signal data
    :type: numpy array
    """

    # Checking that data received are correct
    if type(np_arr) is not np.ndarray:
        sys.stderr.write(f"Error: np_array is not a numpy array, can not be stored\n")
        sys.exit(EXIT_FAILURE)
    
    # Adding new column
    df[column_name] = np_arr


def get_sEMG_data(subject, action_str):
    """
    This function returns the bounded Dataframe with the sEMG data and the time
    stamp of all the data points.

    :param subject: Name of the subject whose data comes from
    :type: string
    :param action_str: Action that the subject is doing
    :type: string
    :return: The bounded dataframe
    :rtype: pd.Dataframe
    """

    data_file = subject + '_' + action_str + '_Data.csv'
    # Creating the path
    data_path = os.path.join(DATASET_PATH, subject, 'Data', data_file)
    # Creating the DataFrame reading from a file.
    signal_data = pd.read_csv(data_path)

    # Selecting the sEMG data and the time stamp
    time_df = signal_data.iloc[:, 0:1]
    semg_df = signal_data.filter(regex='^sEMG: ')

    # Creating bounded Dataframe. It concatenates in the cols axis
    bounded_df = pd.concat([time_df, semg_df], axis=1)

    return bounded_df


def get_abs_sEMG_data(df):
    """
    Creates a new dataframe with the absolute values of the different muscles
    :param df: Dataframe of the sEMG signals
    :type: pd.Dataframe
    :return: The absolute values dataframe
    :rtype: pd.Dataframe
    """
    
    abs_df = df.copy()

    for col in abs_df.columns:
        if col != 'Time':
            abs_df[col] = abs_df[col].abs()
    
    return abs_df


def get_y_label_scale(df, signal_type):
    """
    Return the biggest value of the signals to use it to scalate.
    :param df: Dataframe
    :type: pd.Dataframe
    :return: The biggest value
    :rtype: double
    """

    select_columns = df.filter(regex=f'^{signal_type}', axis=1)
    min_val = df.drop(columns=df.columns.difference(select_columns.columns)).min().min()
    max_val = df.drop(columns=df.columns.difference(select_columns.columns)).min().min()

    if abs(min_val) > abs(max_val):
        return abs(min_val)
    
    return abs(max_val)


def signal_max_amplitude(df, muscle_id):
    """
    Return the maximum value of the signal.
    :param df: Dataframe
    :type: Pd.Dataframe
    :param muscle_id: Muscle Id for looking for in the Dataframe
    :type: string
    :return: The amplitude of the signal
    :retype: double
    """
    
    amplitude = 0

    for value in df[muscle_id].values:
        if amplitude < value:
            amplitude = value

    return amplitude


def get_avg_value(signal_arr, time_arr, init_time=-1.0, end_time=-1.0):
    """
    Returns the mean value of the absolute signal from the init time to the end time 
    indicated in the argument. If the time is negative, it will take all the
    time stamp.
    :param muscle_arr: Muscle signal array. If it is negative, it returns an error
    :type: np.ndarray
    :param time_arr: Time line array of the signal record
    :type: np.ndarray
    :param init_time: Time interval start point
    :type: double
    :param time: Time interval end point
    :type: double
    :return: the mean value
    :rtype: double
    """


    # Check if it is the absolute signal. 
    if np.any(signal_arr < 0):
        sys.stderr.write("Mean error: The signal value is not positive.\n")
        sys.exit(EXIT_FAILURE)
    
    if init_time < 0 or end_time >= time_arr[-1] or end_time < 0:
        return np.mean(signal_arr)
    else:
        # Search for the indexes that correspond to this conditionTmean of the
        t_index = np.where((time_arr >= init_time) & (time_arr <= end_time))
        # Creates new array already bounded
        bounded_signal = signal_arr[t_index]

        return np.mean(bounded_signal)


def get_max_value(muscle_id, df, init_time=-1.0, end_time=-1.0):
    """
    Returns the max value of the absolute signal from the init time to the end time muscle_data.
    :param muscle_id: Muscle where the signal is recorded
    :rtype: double
    """
    time_arr = np.array(df['Time'])
    signal_arr = np.array(df[muscle_id])

    # Check if it is the absolute signal. 
    """if np.any(signal_arr < 0):
        sys.stderr.write("Max value error: The signal values are not all positive.\n")
        sys.exit(EXIT_FAILURE)"""

    if init_time < 0 or end_time < 0 or end_time >= time_arr[-1]:
        return signal_max_amplitude(df, muscle_id)
    else:
        t_index = np.where((time_arr >= init_time) & (time_arr <= end_time))
        bounded_signal = signal_arr[t_index]

        return bounded_signal.max()
    

def remove_mean_offset(signal_arr, avg):
    """
    It returns a dataframe without the mean offset.
    :param muscle_id: Muscle where the signal is recorded
    :type: string
    :param df: Absolute values dataframe. If it is negative, it returns an error and exit
    :type: Pandas Dataframe
    :param avg: The mean value of the time interval
    :type: double
    :return: The dataframe without the offset
    :rtype: Pandas dataframe
    """

    signal_arr -= avg
    # if the value is lower than 0, it will set as 0, to avoid malfunction
    signal_arr = np.clip(signal_arr, a_min=0, a_max=None)

    return signal_arr


def get_signal_spectrogram(signal_arr):
    """ 
    Get the signal spectrogram.
    # Parameters:
    :param signal_arr: Values of the determined signal
    """

    freq, time, spectral_density = signal.spectrogram(signal_arr, FREQ_SAMPLING)

    return freq, time, spectral_density


def get_max_contraction_value(sub, muscle):
    
    max_contraction = 0
    file_name = sub + '_*_Data.csv'
    data_path = os.path.join(DATASET_PATH, sub, 'Data', file_name)

    for file in glob.glob(data_path):
        signal_data = pd.read_csv(file)

        # Selecting the sEMG data and the time stamp
        time_df = signal_data.iloc[:, 0:1]
        semg_df = signal_data.filter(regex='^sEMG: ')

        # Creating bounded Dataframe. It concatenates in the cols axis
        data_df = pd.concat([time_df, semg_df], axis=1)

        signal_processing(data_df, muscle, FILTER_LIST)
        signal_envelope(data_df, muscle, ENVELOP_HIGH_CUT_FREQ, ENVELOP_LOW_CUT_FREQ, FREQ_SAMPLING, BUTTER_ORDER)
        
        muscle_id = 'Envelope ' + muscle 
        max_val = get_max_value(muscle_id, data_df)
    
        if (max_val > max_contraction):
            max_contraction = max_val
    
    return max_contraction


# -------------- FILTERS --------------

def get_notch_filtered_signal(signal_arr, freq_to_remove, quality_factor, fs):

    signal_data = signal_arr

    # Calculating harmonics in a array
    harmonics = np.array([freq_to_remove * i for i in range(1, fs // freq_to_remove)]) # Comprehension list

    for fi in harmonics:
        # Notch filter: iirnotch returns num y denum of the IIR filter. The frequency to remove should be normalized
        num, denum = signal.iirnotch(fi/(fs/2), quality_factor, fs)

        #Aplying filter, applying it backwards and forwards
        filtered_signal = signal.filtfilt(num, denum, signal_data)

    return filtered_signal


def get_butterworth_filtered_signal(signal_arr, type='bandpass', cut_freq=np.array([BUTTER_LOW_FREQ, BUTTER_HIGH_FREQ]), fs=FREQ_SAMPLING, order=BUTTER_ORDER):
    """
    Get a Butterworth filter and apply to the signal in the arguments. The bandpass 
    filter is set by default.
    """

    if not isinstance(cut_freq, np.ndarray):
        cut_freq = np.array(cut_freq)
    
    # Normalizing frequency by Nyquist-Shannon Theorem
    cut_freq = cut_freq / (fs / 2)
    # Creating filter. It returns the num and denum of the TF
    num, denum = signal.butter(order, cut_freq, btype=type, output='ba')
    #Aplying filter to the signal and rectifying
    filtered_signal = signal.filtfilt(num, denum, signal_arr)
    #filtered_signal = np.abs(filtered_signal)

    return filtered_signal


def normalize_data(sub, df, muscle, filtered=False, envelope=False):
    """
    Normalizing signal data using the max contraction value of the muscle.
    :param sub: Name of the subject whose data comes from
    :type: string
    :param df: Dataframe of the signal data
    :type: pd.Dataframe
    :param muscle: Muscle name whose signal is processed
    :type: string
    :param filtered: Indicates if the signal was filtered or not
    :type: Bool
    :param envelope: Indicates if the envelope signal wants to be plotted
    :type: Bool 
    """
    # Get maximum contraction value of the muscle:
    max_contraction = get_max_contraction_value(sub, muscle)
    # Column names:
    if envelope == True:
        muscle = 'Envelope ' + muscle
    elif filtered == True:
        muscle = 'Filtered ' + muscle
    # Normalizing
    df[muscle] = df[muscle] / max_contraction


def wavelet_packet(signal_data, threshold=WVLT_THRESHOLD, mode=WVLT_THRHLD_MODE, type=WVLT_PACKET_TYPE, level=WVLT_LEVEL):

    filtered_signal = np.empty_like(signal_data)
    #Discrete Wavelet Transform 
    coeffs = pywt.wavedec(signal_data, type, level=level)

        
# -------------- PLOT FUNCTIONS --------------

def plot_sEMG_signal(sub, df, action, muscles=['all'], filtered=False, envelope=False):
    """
    Plot a muscle signal, depending if it is filtered or not,
    or the envelope is wanted to be plotted.
    # Parameters:
    :param sub: Name of the subject whose data comes from
    :type: string
    :param df: Dataframe of the signal data
    :type: pd.Dataframe
    :param action: Movement the subject is doing
    :type: string
    :param muscle: Muscles where the signals are recorded
    :type: string list
    :param filtered: Indicates if the signal was filtered or not
    :type: Bool
    :param envelope: Indicates if the envelope signal wants to be plotted
    :type: Bool 
    """

    # Creating figure
    fig = plt.figure('Subject: ' + sub + ' action ' + action + ' signal')

    if (len(muscles) > 5 and len(muscles) <= MUSCLES_N) or muscles[0] == 'all':
        plt_index = 521 # This index is 5 rows, 2 cols and starting with the figure 1
    else:
        # Creating the index for the plots figures
        plt_index = len(muscles) * 100 + 11

    # If not represents all muscles
    if muscles[0] != 'all':
        for i in range(len(muscles)):
            if envelope == True:
                muscles[i] = "Envelope sEMG: " + muscles[i]
            elif filtered == True:
                muscles[i] = "Filtered sEMG: " + muscles[i]
            else:
                muscles[i] = 'sEMG: ' + muscles[i]
    else: # bound the muscle array only to use the envelope, filtered or raw signal muscles
        if envelope and filtered:
            muscles = [col for col in df.columns if "Envelope" in col]
        elif not envelope and filtered:
            muscles = [col for col in df.columns if "Filtered" in col]
        else:
            muscles = [col for col in df.columns if "Filtered" not in col and "Envelope" not in col]

    if muscles[0] == 'Time':
        muscles = muscles[1:]

    # Starts to plot
    for muscle in muscles:
        ax = fig.add_subplot(plt_index)
        
        if envelope == True: # If the envelope wants to be plotted
            ax.plot(np.array(df['Time']), np.array(df[muscle]), color='coral')
            ax.set_ylim(-PLT_AMPLITUDE_OFFSET, 1)
        elif filtered == True:
            ax.plot(np.array(df['Time']), np.array(df[muscle]), color='darkgreen')
            ax.set_ylim(-PLT_AMPLITUDE_OFFSET, 1)
        else:
            ax.plot(np.array(df['Time']), np.array(df[muscle]))
            ax.set_ylim(-PLT_AMPLITUDE_OFFSET, 1)

        ax.set_title(muscle)
        ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Value')

        # Putting the tickers
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

        plt_index += 1

    fig.tight_layout(h_pad=0)
    fig.subplots_adjust(hspace=0.7)
    ticker.AutoLocator()


def plot_sEMG_signal_raw(sub, df, action, muscle='all'):
    """
    Plot the signal figures using matplotlib library.
    :param sub: Name of the subject whose data comes from
    :type: string
    :param df: Dataframe of the signal data
    :type: pd.Dataframe
    :param action: Movement the subject is doing
    :type: string
    :param muscle: Muscle where the signal is recorded
    :type: string
    """

    fig = plt.figure('sEMG ' + action + ' signals: ' + muscle + ' from ' + sub)

    if muscle == 'all':
        # New sub-figure is created, it represents all the sEMG signals
        plt_index = 521 # This index is 5 rows, 2 cols and starting with the figure 1
        muscle_index = 1 # It starts in 1 because the 0 is Time
        char_n = df.shape[1] - 1 # Eliminates time

        for _ in range(char_n):
            ax = fig.add_subplot(plt_index)
            ax.plot(np.array(df['Time']), np.array(df.iloc[:, muscle_index]), alpha= 0.5)
            #Printing notch filtered signal
            ax.plot(df['Time'].values, get_notch_filtered_signal(df[df.columns[muscle_index]].values,
                NOTCH_FREQ_TO_REMOVE, NOTCH_QUALITY_FACTOR, FREQ_SAMPLING), color='coral')
            
            ax.set_title(df.columns[muscle_index])
            ax.set_ylim(-get_y_label_scale(df), get_y_label_scale(df))
            ax.set_xlabel('Time [sec]')
            ax.set_ylabel('Value')
            # Putting the tickers
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

            plt_index += 1
            muscle_index += 1

        fig.tight_layout(h_pad=0)
        fig.subplots_adjust(hspace=0.7)
        

    else:
        muscle_prmpt = 'sEMG: ' + muscle
        ax = fig.add_subplot(111)

        ax.plot(np.array(df['Time']), np.array(df[muscle_prmpt]), alpha=0.45)
        ax.plot(df['Time'].values, get_notch_filtered_signal(df[muscle_prmpt],
                NOTCH_FREQ_TO_REMOVE, NOTCH_QUALITY_FACTOR, FREQ_SAMPLING), color='coral')
        
        ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Value')
        # Putting the tickers
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    ticker.AutoLocator()


def plot_spectrogram(sub, df, action, muscle='all'):
    
    spect_fig = plt.figure(f'Spectrogram: sEMG {action} signals: {muscle} from {sub}')

    if muscle != "all":
        muscle_prmpt = 'sEMG: ' + muscle
        ax = spect_fig.add_subplot(111)

        notch = get_notch_filtered_signal(df[muscle_prmpt], NOTCH_FREQ_TO_REMOVE, NOTCH_QUALITY_FACTOR, FREQ_SAMPLING)
        filtered_signal = get_butterworth_filtered_signal(notch, 'bandpass', [BUTTER_LOW_FREQ, BUTTER_HIGH_FREQ], FREQ_SAMPLING, BUTTER_ORDER)
        freq, times, amp_density = get_signal_spectrogram(filtered_signal)

        ax.pcolormesh(times, freq, 10 * np.log10(amp_density))
        ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Frquency [Hz]')


# -------------- ACTION MENUS --------------

def signal_processing(df, muscle, filters_list):
    """
    Executes all the filtering and processing things that are needed to be done.
    # Parameters:
    :param df: Dataframe of the signal data
    :type: pd.Dataframe
    :param muscle: Muscle name whose signal is processed
    :type: string
    :param filters_list: List of all the filters are made in the process
    :type: string List
    """
    print("----FILTER ZONE----")
    print("MUSCLE:", muscle)

    muscle_signal = df[muscle].values
    
    # If there are no filters to do, return the original one
    if np.size(filters_list) == 0:
        return muscle_signal
    
    # If there are filters to do:
    filtered_signal = np.copy(muscle_signal)
    col_name = 'Filtered ' + muscle

    for filter in filters_list:
        if filter == 'mean_off':
            filtered_signal = remove_mean_offset(filtered_signal, get_avg_value(filtered_signal, df['Time'].values, 0, TIME_UNTIL_MOVEMENT))
            print("Filtro AVERAGE aplicado")

        if filter == 'notch':
            filtered_signal = get_notch_filtered_signal(filtered_signal, NOTCH_FREQ_TO_REMOVE,
                NOTCH_QUALITY_FACTOR, FREQ_SAMPLING)
            print("Filtro Notch aplicado")

        if filter == 'butterworth':
            filtered_signal = get_butterworth_filtered_signal(filtered_signal, 
                'bandpass', [BUTTER_LOW_FREQ, BUTTER_HIGH_FREQ], 1920, BUTTER_ORDER)
            wavelet_packet(filtered_signal)
            print("Filtro Butterworth paso banda aplicado")
        
        if filter == 'wavelet':
            filtered_signal = wavelet_packet(filtered_signal)

    add_column_to_df(df, col_name, filtered_signal)


def signal_envelope(df, muscle, high_cut_freq=ENVELOP_HIGH_CUT_FREQ, low_cut_freq=ENVELOP_LOW_CUT_FREQ, fs=FREQ_SAMPLING, order=BUTTER_ORDER):
    '''
    Process the sEMG signal to obtain the envelop of it.
    # Arguments
    :param df: Dataframe of the muscle signal
    :param muscle:
    :param low_cut_freq:
    :param fs:
    :param order:
    '''
    print("----ENVELOPE ZONE----")

    filtered_name = 'Filtered ' + muscle
    f_signal = np.copy(df[filtered_name].values)

    # Normalizing frequency by Nyquist-Shannon Theorem
    low_cut_freq = low_cut_freq / (fs / 2)
    high_cut_freq = high_cut_freq / (fs / 2)

    # Butterworth high filter
    num,denum = signal.butter(order, high_cut_freq, btype='high', output='ba') # A low pass filter
    env_signal = signal.filtfilt(num, denum, f_signal)
    #Rectifying
    env_signal = np.abs(env_signal)
    # Butterworth high filter
    num,denum = signal.butter(order, low_cut_freq, btype='low', output='ba') # A low pass filter
    env_signal = signal.filtfilt(num, denum, env_signal)

    envelope_name = 'Envelope ' + muscle

    add_column_to_df(df, envelope_name, env_signal)

    print("Envolvente creada")


def main():
    if len(sys.argv) != ARGS_N:
        usage()

    subject = sys.argv[1]
    act_str = sys.argv[2]
    muscle_str = sys.argv[3]

    semg_df = get_sEMG_data(subject, act_str)

    if (muscle_str == 'all'):
        for muscle in semg_df.columns:
            if muscle == 'Time':
                continue
            signal_processing(semg_df, muscle, FILTER_LIST)
            signal_envelope(semg_df, muscle, ENVELOP_HIGH_CUT_FREQ, ENVELOP_LOW_CUT_FREQ, FREQ_SAMPLING, BUTTER_ORDER)
            normalize_data(subject, semg_df, muscle, True, True)


    else:
        muscle = 'sEMG: ' + muscle_str
        signal_processing(semg_df, muscle, FILTER_LIST)
        #signal_envelope(semg_df, muscle, ENVELOP_HIGH_CUT_FREQ, ENVELOP_LOW_CUT_FREQ, FREQ_SAMPLING, BUTTER_ORDER)
        # normalize_data(subject, semg_df, muscle, True, True)

    if muscle_str != 'all':
        plot_spectrogram(subject, semg_df, act_str, muscle_str)
    
    #plot_sEMG_signal(subject, semg_df, act_str, [muscle_str], filtered=True, envelope=True)
    
    plt.show()


if __name__ == "__main__":
    main()
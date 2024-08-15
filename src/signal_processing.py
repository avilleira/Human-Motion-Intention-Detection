import sys
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import ticker
from scipy import signal
from scipy import integrate
import pywt

# MACROS

EXIT_FAILURE = 1
ARGS_N = 1
DATASET_PATH = '/home/avilleira/TFG/dataset/SIAT_LLMD20230404/'
TRAIN_DATA_PATH = '/home/avilleira/TFG/tfg/data/training_data.csv'
NORMS_PARAM_FILE = '/home/avilleira/TFG/tfg/data/normalization_params.csv'
TIME_UNTIL_MOVEMENT = 3 #seconds
MUSCLES_N = 9
FILTER_LIST = ['notch', 'butterworth', 'wavelet']
DU_SET_LIST = ['iemg', 'variance', 'waveform length', 'zero crossing', 'slope sign change', 'willison amplitude']
DU_FEATURES_N = len(DU_SET_LIST)
ACTIONS_LIST = ['STDUP', 'SITDN', 'WAK']

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
WVLT_PACKET_TYPE = 'db8'
WVLT_LEVEL = 9

SEMG_WINDOW_SIZE = 150
WAK_WINDOW_SIZE = 80

WILLISON_THRESHOLD = 0.0005


def usage():
    """
    Usage error exit
    """

    sys.stderr.write("Usage: python3 signal_processing.py SubXX ACTION MUSCLE/ALL\n")
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


def get_dataframe(subject, action):
    """
    Returns a dataframe from a data file created with the arguments
    :param subject: Name of the subject whose data comes from
    :type: string
    :param action: Action that the subject is doing
    :type: string
    :return: The dataframe
    :rtype: pd.Dataframe
    """

    data_file = subject + '_' + action + '_Data.csv'
    # Creating the path
    data_path = os.path.join(DATASET_PATH, subject, 'Data', data_file)
    # Returns the DataFrame reading from a file.
    return pd.read_csv(data_path)


def get_sEMG_data(subject, action_str):
    """
    This function returns the bounded Dataframe with the sEMG data and the time
    stamp of all the data points.
    # Parameters
    :param subject: Name of the subject whose data comes from
    :type: string
    :param action_str: Action that the subject is doing
    :type: string
    :return: The bounded dataframe
    :rtype: pd.Dataframe
    """

    signal_data = get_dataframe(subject, action_str)
    # Selecting the sEMG data and the time stamp
    time_df = signal_data.iloc[:, 0:1]
    semg_df = signal_data.filter(regex='^sEMG: ')

    # Creating bounded Dataframe. It concatenates in the cols axis
    bounded_df = pd.concat([time_df, semg_df], axis=1)

    return bounded_df


def get_kinematic_data(subject, action_str):
    """
    Returns the bounded Dataframe with the kinematic angles and forces,
    including the time stamp.
    # Paramaters
    :param subject: Name of the subject whose data comes from
    :type: string
    :param action_str: Action that the subject is doing
    :type: string
    :return: The bounded dataframe
    :rtype: pd.Dataframe
    """

    signal_data = get_dataframe(subject, action_str)
    # Selecting the kinematic data and the time stamp
    time_df = signal_data.iloc[:, 0:1]
    kinec_df = signal_data.filter(regex='^Kinematic: ')
    # Creating bounded Dataframe. It concatenates in the cols axis
    return pd.concat([time_df, kinec_df], axis=1)
    

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
    Return the max and minumu value of the signals to use it to escalate.
    :param df: Dataframe
    :type: pd.Dataframe
    :return: The biggest and lowest value
    :rtype: double
    """

    select_columns = df.filter(regex=f'^{signal_type}', axis=1)
    min_val = df.drop(columns=df.columns.difference(select_columns.columns)).min().min()
    max_val = df.drop(columns=df.columns.difference(select_columns.columns)).max().max()

    return max_val, min_val


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
        if amplitude < abs(value):
            amplitude = abs(value)

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
    :type: double
    """
    time_arr = np.array(df['Time'])
    signal_arr = np.array(df[muscle_id])

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

    :param signal_arr: Values of the determined signal
    """

    freq, time, spectral_density = signal.spectrogram(signal_arr, FREQ_SAMPLING)
    return freq, time, spectral_density


def get_max_wak_contraction_value(sub, muscle):
    """
    """
    
    max_contraction = 0
    file_name = sub + '_WAK_Data.csv'
    data_path = os.path.join(DATASET_PATH, sub, 'Data', file_name)

    signal_data = pd.read_csv(data_path)

    # Selecting the sEMG data and the time stamp
    time_df = signal_data.iloc[:, 0:1]
    semg_df = signal_data.filter(regex='^sEMG: ')
    # Creating bounded Dataframe. It concatenates in the cols axis
    data_df = pd.concat([time_df, semg_df], axis=1)
    signal_processing(data_df, muscle, FILTER_LIST)
    #signal_envelope(data_df, muscle, ENVELOP_HIGH_CUT_FREQ, ENVELOP_LOW_CUT_FREQ, FREQ_SAMPLING, BUTTER_ORDER)
    
    muscle_id = 'Filtered ' + muscle 
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


def get_wavelet_filtered_signal(signal_data, threshold=WVLT_THRESHOLD, mode=WVLT_THRHLD_MODE, wvlt_type=WVLT_PACKET_TYPE, level=WVLT_LEVEL):
    """
    Obtain the Wavelet transform of the signal and apply it to the signal.
    """

    #Discrete Wavelet Transform 
    coeffs = pywt.wavedec(signal_data, wvlt_type, level=level)

    # Aplying the Details Wavelet coefficients
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(data=coeffs[i], value=threshold * max(coeffs[i]),
             mode=mode, substitute=0)

    # Reconstruct the signal
    filtered_signal = pywt.waverec(coeffs, wvlt_type)
    filtered_signal = np.array(filtered_signal[0:signal_data.size]) 

    return filtered_signal


def normalize_data(sub, df, muscle, filtered=False, envelope=False):
    """
    Normalizing signal data using the max contraction value of the muscle.
    #Parameters
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
    # Get maximum wak contraction values of the muscle:
    # It is a dataframe
    max_values = pd.read_csv(NORMS_PARAM_FILE, index_col=0)
    # Column names:
    if envelope == True:
        muscle = 'Envelope ' + muscle
    elif filtered == True:
        muscle = 'Filtered ' + muscle
    
    max_contraction = max_values.loc[sub, muscle]
    # Normalizing
    df[muscle] = df[muscle] / max_contraction


def get_joint_slope(df, joint_id):

    dif_x = df['Time'].diff()
    dif_y = df[joint_id].diff()

    slope_str = 'Slope: ' + joint_id
    df[slope_str] = dif_y / dif_x

    # In the case of the first and the last position, 
    # the operation above does not work
    df[slope_str].iloc[0] = (df[joint_id].iloc[1] - df[joint_id].iloc[0]) / (df['Time'].iloc[1] - df['Time'].iloc[0])
    df[slope_str].iloc[-1] = (df[joint_id].iloc[-1] - df[joint_id].iloc[-2]) / (df['Time'].iloc[-1] - df['Time'].iloc[-2])

    return df

# -------------- PLOT FUNCTIONS --------------

def plot_sEMG_signal(sub, df, action, muscles=['all'], joint='', filtered=False, envelope=False, kinem_df=None):
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
                ymax, ymin = get_y_label_scale(df, "Envelope")
            elif filtered == True:
                muscles[i] = "Filtered sEMG: " + muscles[i]
                ymax, ymin = get_y_label_scale(df, "Filtered")
            else:
                muscles[i] = 'sEMG: ' + muscles[i]
                ymax, ymin = get_y_label_scale(df, "sEMG")

    else: # bound the muscle array only to use the envelope, filtered or raw signal muscles
        if envelope and filtered:
            muscles = [col for col in df.columns if "Envelope" in col]
            ymax, ymin = get_y_label_scale(df, "Envelope")

        elif not envelope and filtered:
            muscles = [col for col in df.columns if "Filtered" in col]
            ymax, ymin = get_y_label_scale(df, "Filtered")

        else:
            muscles = [col for col in df.columns if "Filtered" not in col and "Envelope" not in col]
            ymax, ymin = get_y_label_scale(df, "sEMG")

    if muscles[0] == 'Time':
        muscles = muscles[1:]

    #joint = 'Kinematic: ' + joint

    #joint_data = kinem_df[joint] / 180
    #kinem_df = get_slope(kinem_df, joint)

    #slope = 'Slope: ' + joint
    #print(np.mean(kinem_df[slope]))
    # Starts to plot
    for muscle in muscles:
        ax = fig.add_subplot(plt_index)
        if envelope == True: # If the envelope wants to be plotted
            ax.plot(np.array(df['Time']), np.array(df[muscle]), color='coral')
        elif filtered == True:
            ax.plot(np.array(df['Time']), np.array(df[muscle]), color='darkgreen')
            #ax.plot(np.array(df['Time']), np.array(kinem_df[slope]), color='red')

        else:
            ax.plot(np.array(df['Time']), np.array(df[muscle]))

        ax.set_ylim(ymin-PLT_AMPLITUDE_OFFSET, ymax+PLT_AMPLITUDE_OFFSET)
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

        if filter == 'notch':
            filtered_signal = get_notch_filtered_signal(filtered_signal, NOTCH_FREQ_TO_REMOVE,
                NOTCH_QUALITY_FACTOR, FREQ_SAMPLING)

        if filter == 'butterworth':
            filtered_signal = get_butterworth_filtered_signal(filtered_signal, 
                'bandpass', [BUTTER_LOW_FREQ, BUTTER_HIGH_FREQ], 1920, BUTTER_ORDER)
        
        if filter == 'wavelet':
            filtered_signal = get_wavelet_filtered_signal(filtered_signal)
        
    add_column_to_df(df, col_name, filtered_signal)


def signal_envelope(df, muscle, high_cut_freq=ENVELOP_HIGH_CUT_FREQ, low_cut_freq=ENVELOP_LOW_CUT_FREQ, fs=FREQ_SAMPLING, order=BUTTER_ORDER):
    '''
    Process the sEMG signal to obtain the envelop of it.
    # Parameters
    :param df: Dataframe of the muscle signal
    :param muscle:
    :param low_cut_freq:
    :param fs:
    :param order:
    '''

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


# =========================== DATA ANALYSIS ===========================


def window_sliding(muscle, df, action):
    """
    Sliding Window to get the data easily
    """

    # Selecting number of windows
    if (action == "WAK"):
        window_size = WAK_WINDOW_SIZE
    else:
        window_size =SEMG_WINDOW_SIZE

    # Generating the window sliding
    signal_arr = df[muscle].to_numpy()
    slide_signal_arr = [signal_arr[i:i + window_size] for i in range(0, len(signal_arr), window_size)]

    return slide_signal_arr


def get_activity_in_signal(sub, windowed_signal, action):
    """
    Getting the windowed signal windows that are actually activated. The measure
    is done by detecting if in a signal window the action is activated the half
    time or more.

    # Parameters:
    :param sub: Name of the subject whose data comes from
    :type: string
    :param windowed_signal:
    :type: numpy.ndarray
    :param action: Movement the subject is doing
    :type: string
    """

    labels_file_name = sub + "_" + action + "_Label.csv"
    file_path = os.path.join(DATASET_PATH, sub, 'Labels', labels_file_name)

    # Reading the csv
    labels = pd.read_csv(file_path)
    active_windowed_signal = window_sliding('Status', labels, action)

    # If the signal is half activated or more, dectects it as activated
    act_wind_labels = np.array(['A' if np.count_nonzero(window == 'A') >=
                        len(window) / 2 else 'R' for window in active_windowed_signal])
    
    # Once we have the activity, then filter the signal windows
    filtered_window_singal = []
    for window_index in range(len(act_wind_labels)):
        if act_wind_labels[window_index] == 'A':
            filtered_window_singal.append(windowed_signal[window_index])
    
    return filtered_window_singal

      
def get_Du_feature_set(windowed_signal_arr):

    du_data = {} # Dictionary
    for feature in DU_SET_LIST:
        if feature == 'iemg':
            du_data[feature] = get_integrated_EMG(windowed_signal_arr)

        elif feature == 'variance':
            du_data[feature] = get_variance_EMG(windowed_signal_arr)

        elif feature == 'waveform length':
            du_data[feature] = get_waveform_length(windowed_signal_arr)

        elif feature == 'zero crossing':
            du_data[feature] = get_zero_crossing(windowed_signal_arr)

        elif feature == 'slope sign change':
            du_data[feature] = get_slope_sign(windowed_signal_arr)
        
        elif feature == 'willison amplitude':
            du_data[feature] = get_willison_amplitude(windowed_signal_arr)

        else:
            print("Error: Feature not valid")
            exit(EXIT_FAILURE)

    return du_data


def get_integrated_EMG(signal_arr):
    
    iemg_arr = []
    for window in signal_arr:
        abs_signal = np.abs(window)
        iemg_arr.append(integrate.simpson(abs_signal, dx=1/FREQ_SAMPLING))
    
    return iemg_arr


def get_variance_EMG(windowed_signal_arr):
    
    var_arr = []
    for window in windowed_signal_arr:
        var_arr.append(np.var(window))

    return var_arr


def get_waveform_length(windowed_signal_arr):
    """
    Cumulative length of the EMG waveform over the time segment.
    :param windowed_signal_arr:
    :type: numpy.ndarray
    """

    waveform_arr = []
    for window in windowed_signal_arr:
        waveform_arr.append(np.sum(np.abs(np.diff(window))))

    return waveform_arr


def get_zero_crossing(windowed_signal_arr):
    
    zero_cross = []
    for window in windowed_signal_arr:
        sign_values = np.sign(window)
        # If the value is 0, changin to positive
        sign_values[sign_values == 0] = 1

        zero_cross.append(np.sum(np.diff(sign_values) != 0)) # np.sign calculates the signs of the value (-1 negative) (1 positive)
        # np.sum(boolean operation) sum all the elements that achieve the operation
    
    return zero_cross


def get_slope_sign(windowed_signal_arr):
    
    slope_sign_arr = []
    for window in windowed_signal_arr:
        # First, calculate the difference between consecutive points. After that, the signs
        slope_sign = np.sign(np.diff(window))
        slope_sign[slope_sign == 0] = 1
        
        # Get the slope signs changes counter
        slope_sign_arr.append(np.sum(np.diff(slope_sign) != 0))

    return slope_sign_arr


def get_willison_amplitude(windowed_signal_arr):
    
    willison_arr = []
    for window in windowed_signal_arr:
        willison_arr.append(np.sum(np.abs(np.diff(window)) > WILLISON_THRESHOLD))
    
    return willison_arr


def generate_data_csv(inputs: dict, output_action: str):
    """
    Generate the dataframe in order to create the dataset.
    The structure is
    LABEL               ....        OUTPUT
    (x, x, x, x, x, x,)               y
    """
    df = pd.DataFrame()

    # Add output as the last column
    if output_action == 'STDUP':
        action = 0
    elif output_action == 'SITDN':
        action = 1
    elif output_action == 'WAK':
        action = 2
    
    print(inputs.keys())
    for key in inputs.keys():
        data_list = [data for data in zip(*inputs[key])]
                
    #df['output'] = action

    ## Save it in a CSV
    #if os.path.isfile(TRAIN_DATA_PATH):
    #    df.to_csv(TRAIN_DATA_PATH, mode='a', header= False, index=False)
    #else:
    #    df.to_csv(TRAIN_DATA_PATH, index=False)


def main():

    if len(sys.argv) != ARGS_N:
        usage()

    #subject = sys.argv[1]semg_df
    # Processing all the subjects
    dir_elements = os.listdir(DATASET_PATH)
    # Selecting only the subjects directories
    subjects = [elem for elem in dir_elements if 
                os.path.isdir(os.path.join(DATASET_PATH, elem)) and elem.startswith('Sub')]
    

    for sub in subjects:
        for action in ACTIONS_LIST:
            semg_df = get_sEMG_data(sub, action)
            data_dict = {}

            for muscle in semg_df.columns:       
                # Not counting time column
                if muscle == 'Time':
                    continue
                # Filters
                signal_processing(semg_df, muscle, FILTER_LIST)
                # Normalizing data
                normalize_data(sub, semg_df, muscle, filtered=True)
                
                # Windowing signal
                windowed_signal = window_sliding(muscle, semg_df, action)
                # Extracting activated signal
                windowed_signal = get_activity_in_signal(sub, windowed_signal, action)
                # Du features
                signal_features = get_Du_feature_set(windowed_signal)

                for feature in signal_features:
                    if feature not in data_dict:
                        data_dict[feature] = [signal_features[feature]]
                    else:
                        data_dict[feature].append(signal_features[feature])
            # Generate the Data CSV
            generate_data_csv(data_dict, action)
            break
        break

        print(f"{sub} finalizado")
    
    # if (muscle_str == 'all'):
    #     for muscle in semg_df.columns:
    #         if muscle == 'Time':
    #             continue
    #         signal_processing(semg_df, muscle, FILTER_LIST)
    #         # signal_envelope(semg_df, muscle, ENVELOP_HIGH_CUT_FREQ, ENVELOP_LOW_CUT_FREQ, FREQ_SAMPLING, BUTTER_ORDER)
    #         normalize_data(subject, semg_df, muscle, True)
    #         windowed_signal = window_sliding(muscle, semg_df, act_str)
    #         signal_features = get_Du_feature_set(windowed_signal)
    #         generate_data_csv(signal_features, muscle, act_str)


    # else:
    #     muscle = 'sEMG: ' + muscle_str
    #     signal_processing(semg_df, muscle, FILTER_LIST)
    #     # signal_envelope(semg_df, muscle, ENVELOP_HIGH_CUT_FREQ, ENVELOP_LOW_CUT_FREQ, FREQ_SAMPLING, BUTTER_ORDER)
    #     normalize_data(subject, semg_df, muscle, True)
    #     windowed_signal = window_sliding(muscle, semg_df, act_str)
    #     signal_features = get_Du_feature_set(windowed_signal)
    #     generate_data_csv(signal_features, muscle, act_str)

    # #plot_sEMG_signal(subject, semg_df, act_str, [muscle_str], filtered=True, envelope=False)
    # plt.show()


if __name__ == "__main__":
    main()
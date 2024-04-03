import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import ticker

# MACROS

EXIT_FAILURE = 1
ARGS_N = 4
DATASET_PATH = '../../dataset/SIAT_LLMD20230404'
TIME_UNTIL_MOVEMENT = 3 #seconds


class EMG_Signal:

    def __init__(self):
        print("hOLA")

def usage():
    """
    Usage error exit
    """

    sys.stderr.write("Usage: python3 signal_plot.py SubXX action muscle/all\n")
    sys.exit(EXIT_FAILURE)


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


def get_y_label_scale(df):
    """
    Return the biggest value of the signals to use it to scalate.
    :param df: Dataframe
    :type: pd.Dataframe
    :return: The biggest value
    :rtype: double
    """

    min_val = df.drop(columns=['Time']).min().min()
    max_val = df.drop(columns=['Time']).max().max()

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

    for value in df[muscle_id]:
        if amplitude < abs(value):
            amplitude = abs(value)

    return amplitude


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

    if muscle == "all":
        # New sub-figure is created, it represents all the sEMG signals
        plt_index = 521 # This index is 3 rows, three rows and starting with the figure 1
        muscle_index = 1 # It starts in 1 because the 0 is Time
        char_n = df.shape[1] - 1 # Eliminates time

        for _ in range(char_n):
            ax = fig.add_subplot(plt_index)
            ax.plot(np.array(df['Time']), np.array(df.iloc[:, muscle_index]))
            
            ax.set_title(df.columns[muscle_index])
            ax.set_ylim(-get_y_label_scale(df), get_y_label_scale(df))
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            # Putting the tickers
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

            plt_index += 1
            muscle_index += 1

        fig.tight_layout(h_pad=0)
        fig.subplots_adjust(hspace=0.7)
        ticker.AutoLocator()

    else:
        muscle_prmpt = 'sEMG: ' + muscle
        ax = fig.add_subplot(111)

        ax.plot(np.array(df['Time']), np.array(df[muscle_prmpt]))
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        # Putting the tickers
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())


def plot_sEMG_signal_abs(sub, df, action, muscle='all'):
    """
    Plot the absolute values signal figures using matplotlib library.
    :param sub: Name of the subject whose data comes from
    :type: string
    :param df: Dataframe of the signal data
    :type: pd.Dataframe
    :param action: Movement the subject is doing
    :type: string
    :param muscle: Muscle where the signal is recorded
    :type: string
    """

    fig = plt.figure('ABS. sEMG ' + action + ' signals: ' + muscle + ' from ' + sub)

    if muscle == 'all':
        # New sub-figure is created, it represents all the sEMG signals
        plt_index = 521 # This index is 9 rows, 1 column and starting to edit in subplot 1
        muscle_index = 1 # It starts in 1 because the 0 is Time
        char_n = df.shape[1] - 1 # Eliminates time
        plt_offs = 0.02

        for _ in range(char_n):
            ax = fig.add_subplot(plt_index)
            max_amplitude = signal_max_amplitude(df, df.columns[muscle_index])
            
            ax.plot(np.array(df['Time']), np.array(df.iloc[:, muscle_index]), color='darkolivegreen')
            plt.axhline(y=max_amplitude, color='r', linestyle='--') # Print the amplitude of every signal
            
            ax.set_title(df.columns[muscle_index])
            ax.set_xlabel('Time')
            ax.set_ylabel('Value (Absolute)')
            ax.set_ylim(0, get_y_label_scale(df) + plt_offs) # We add a little offset to see the max amplitude inside the figure
            # Putting the tickers
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

            # Print the results
            print(f'Maximum amplitude of the {df.columns[muscle_index]} data: {max_amplitude}')
            print(f'Avg value of the first 3 seconds: {get_avg_value(df.columns[muscle_index], df, 0, TIME_UNTIL_MOVEMENT)}')
            print(f'MAX value of the first 3 seconds of the sEMG: {muscle} data: {get_max_value(df.columns[muscle_index], df, 0, TIME_UNTIL_MOVEMENT)}')

            plt_index += 1
            muscle_index += 1
        
        fig.tight_layout(h_pad=0)
        fig.subplots_adjust(hspace=0.7)
        
        
    else:
        muscle_prmpt = 'sEMG: ' + muscle
        ax = fig.add_subplot(111)
        max_amplitude = signal_max_amplitude(df, muscle_prmpt)
        #df = remove_mean_offset(muscle_prmpt, df, get_avg_value(muscle_prmpt, df, 0, TIME_UNTIL_MOVEMENT))

        ax.plot(abs(np.array(df['Time'])), abs(np.array(df[muscle_prmpt])), color='darkolivegreen')
        plt.axhline(y=max_amplitude, color='r', linestyle='--') # Print the amplitude of the signal
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Value (Absolute)') 
        # Putting the tickers
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        # Print the results
        print(f'MAX value of the sEMG: {muscle} data: {max_amplitude}')
        print(f'Avg value of the first 3 seconds: {get_avg_value(muscle_prmpt, df, 0, TIME_UNTIL_MOVEMENT)}')
        print(f'MAX value of the first 3 seconds of the sEMG: {muscle} data: {get_max_value(muscle_prmpt, df, 0, TIME_UNTIL_MOVEMENT)}')

    #Creating the legend:
    custom_legend = [
        Line2D([0], [0], color='r', linestyle='--', label='Max. Amplitude')
    ]
    fig.legend(handles=custom_legend, loc='lower right')


def get_avg_value(muscle_id, df, init_time=-1.0, end_time=-1.0):
    """
    Returns the mean value of the absolute signal from the init time to the end time 
    indicated in the argument. If the time is negative, it will take all the
    time stamp.
    :param muscle_id: Muscle where the signal is recorded
    :type: string
    :param df: Absolute values dataframe. If it is negative, it returns an error and exit
    :type: Pandas Dataframe
    :param init_time: Time interval start point
    :type: double
    :param time: Time interval end point
    :type: double
    :return: the mean value
    :rtype: double
    """

    time_arr = np.array(df['Time'])
    signal_arr = np.array(df[muscle_id])

    # Check if it is the absolute signal. 
    if np.any(signal_arr < 0):
        sys.stderr.write("Mean error: The signal value is not positive.")
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
    Returns the max value of the absolute signal from the init time to the end time 
    indicated in the argument. If the time is negative, it will take all the
    time stamp.
    :param muscle_id: Muscle where the signal is recorded
    :type: string
    :param df: Absolute values dataframe. If it is negative, it returns an error and exit
    :type: Pandas Dataframe
    :param init_time: Time interval start point
    :type: double
    :param time: Time interval end point
    :type: double
    :return: the maximum value in that time interval
    :rtype: double
    """

    time_arr = np.array(df['Time'])
    signal_arr = np.array(df[muscle_id])

    # Check if it is the absolute signal. 
    if np.any(signal_arr < 0):
        sys.stderr.write("Max value error: The signal values are not all positive.")
        sys.exit(EXIT_FAILURE)

    if init_time < 0 or end_time < 0 or end_time >= time_arr[-1]:
        return signal_max_amplitude(df, muscle_id)
    else:
        t_index = np.where((time_arr >= init_time) & (time_arr <= end_time))
        bounded_signal = signal_arr[t_index]

        return bounded_signal.max()
    

def remove_mean_offset(muscle_id, df, avg):
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

    filtered_df = df.copy()
    filtered_df[muscle_id] -= avg
    # if the value is lower than 0, it will set as 0, to avoid malfunction
    filtered_df[muscle_id] = filtered_df[muscle_id].clip(lower=0)

    return filtered_df


def main():

    if len(sys.argv) != ARGS_N:
        usage()

    subject = sys.argv[1]
    act_str = sys.argv[2]
    muscle_str = sys.argv[3]

    semg_df = get_sEMG_data(subject, act_str)
    abs_semg_df = get_abs_sEMG_data(semg_df)
    
    plot_sEMG_signal_raw(subject, semg_df, act_str, muscle_str)
    plot_sEMG_signal_abs(subject, abs_semg_df, act_str, muscle_str)

    plt.show()


if __name__ == "__main__":
    main()
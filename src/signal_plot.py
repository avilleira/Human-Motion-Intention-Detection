import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# MACROS

EXIT_FAILURE = 1
ARGS_N = 4
DATASET_PATH = '../../dataset/SIAT_LLMD20230404'


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


def signal_amplitude(df, muscle_id):
    """
    Return the maximum value of the signal.
    
    :param df: Dataframe
    :type: Pd.Dataframe
    :param muscle_id: Muscle Id for looking for in the Dataframe
    :type: string
    :return: The amplitude of the signal
    :retype: double"""
    
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

    fig = plt.figure('sEMG ' + action + ' Signals: ' + muscle + ' from ' + sub)

    if muscle == "all":
        # New sub-figure is created, it represents all the sEMG signals
        plt_index = 331 # This index is 3 rows, three rows and starting with the figure 1
        muscle_index = 1 # It starts in 1 because the 0 is Time
        char_n = df.shape[1] - 1 # Eliminates time

        for _ in range(char_n):
            ax = fig.add_subplot(plt_index)
            ax.plot(np.array(df['Time']), np.array(df.iloc[:, muscle_index]))
            
            ax.set_title(df.columns[muscle_index])
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')

            plt_index += 1
            muscle_index += 1

        fig.tight_layout()

    else:
        muscle_prmpt = 'sEMG: ' + muscle
        ax = fig.add_subplot(111)

        ax.plot(np.array(df['Time']), np.array(df[muscle_prmpt]))
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')


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

    fig = plt.figure('ABS: sEMG ' + action + ' Signals: ' + muscle + ' from ' + sub)

    

    if muscle == 'all':
        # New sub-figure is created, it represents all the sEMG signals
        plt_index = 331 # This index is 3 rows, three rows and starting with the figure 1
        muscle_index = 1 # It starts in 1 because the 0 is Time
        char_n = df.shape[1] - 1 # Eliminates time

        for _ in range(char_n):
            ax = fig.add_subplot(plt_index)
            amplitude = signal_amplitude(df, df.columns[muscle_index])

            ax.plot(np.array(df['Time']), abs(np.array(df.iloc[:, muscle_index])), color='darkolivegreen')
            plt.axhline(y=amplitude, color='r', linestyle='--') # Print the amplitude of every signal
            
            ax.set_title(df.columns[muscle_index])
            ax.set_xlabel('Time')
            ax.set_ylabel('Value (Absolute)')

            plt_index += 1
            muscle_index += 1
        
        fig.tight_layout()

    else:
        muscle_prmpt = 'sEMG: ' + muscle
        ax = fig.add_subplot(111)
        amplitude = signal_amplitude(df, muscle_prmpt)

        ax.plot(abs(np.array(df['Time'])), abs(np.array(df[muscle_prmpt])), color='darkolivegreen')
        plt.axhline(y=amplitude, color='r', linestyle='--') # Print the amplitude of the signal
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Value (Absolute)')

    #Creating the legend:
    custom_legend = [
        Line2D([0], [0], color='r', linestyle='--', label='Amplitude')
    ]

    fig.legend(handles=custom_legend)

def main():

    if len(sys.argv) != ARGS_N:
        usage()

    subject = sys.argv[1]
    act_str = sys.argv[2]
    muscle_str = sys.argv[3]

    semg_df = get_sEMG_data(subject, act_str)
    plot_sEMG_signal_raw(subject, semg_df, act_str, muscle_str)

    plot_sEMG_signal_abs(subject, semg_df, act_str, muscle_str)

    plt.show()


if __name__ == "__main__":
    main()
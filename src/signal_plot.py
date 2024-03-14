import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


def signal_amplitude(df, muscle):
    
    max_amplitude = 0
    print("LLEGO AQUI")
    for value in df[muscle]:
            if abs(value) > max_amplitude:
                max_amplitude

    return max_amplitude


def plot_sEMG_signal(sub, df, action, muscle='all'):
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
        muscle_index = 1
        char_n = df.shape[1] - 1

        for i in range(char_n):
            ax = fig.add_subplot(plt_index)
            ax.plot(np.array(df['Time']),np.array(df.iloc[:, muscle_index]))
            
            ax.set_title(df.columns[muscle_index])
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')

            plt_index += 1
            muscle_index += 1

        fig.tight_layout()

    else:
        muscle_prmpt = 'sEMG: ' + muscle
        ax = fig.add_subplot(111)
        max_amp = signal_amplitude(df, muscle)

        ax.plot(np.array(df['Time']), np.array(df[muscle_prmpt]))
        plt.axhline(y=max_amp, color='r', linestyle='--')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')

    plt.show()

def main():

    if len(sys.argv) != ARGS_N:
        usage()

    subject = sys.argv[1]
    act_str = sys.argv[2]
    muscle_str = sys.argv[3]

    semg_df = get_sEMG_data(subject, act_str)
    plot_sEMG_signal(subject, semg_df, act_str, muscle_str)


if __name__ == "__main__":
    main()
import sys
import os
import pandas as pd
import numpy as np

# Macros
EXIT_FAILURE = 1
ARGS_N = 3
DATASET_PATH = '../../dataset/SIAT_LLMD20230404'


def usage():
    """
    Usage error exit
    """

    sys.stderr.write("Usage: python3 signal_plot.py subject_name muscle\n")
    sys.exit(EXIT_FAILURE)


def get_path(subject, muscle):
    """
    This function returns the complete route to the dataset
    """

    muscle += '.csv'
    return  os.path.join(DATASET_PATH, subject, 'Data', muscle)


def get_sEMG_data(data_path):
    """
    This function returns the bounded Dataframe
    """

    signal_data = pd.read_csv(data_path)
    signal_data.iloc[:, 9:16] # Para poder realizar el split en los diferentes csv
    

def main():

    if len(sys.argv) != ARGS_N:
        usage()

    sub_path = get_path(sys.argv[1], sys.argv[2])
    print(sub_path)


if __name__ == "__main__":
    main()
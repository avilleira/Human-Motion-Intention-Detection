import ast
import pandas as pd
import numpy as np
from sklearn import svm

TRAIN_DATA_PATH = '/home/avilleira/TFG/tfg/data/training_data.csv'


def is_str_list(str_list):
    return isinstance(str_list, str) and str_list.startswith('[')


def str_to_column(col):
    """
    Transform the string elements of the dataset in arrays.

    :param df: The dataframe that contains all the data
    :type: pandas.Dataframe
    """
    if is_str_list(col):
        return ast.literal_eval(col)
    return col


def main():
    
    # Reading from the csv file:
    data_svm = pd.read_csv(TRAIN_DATA_PATH)
    
    # Transforming string columns to list
    for col in data_svm.columns:
        if data_svm[col].apply(is_str_list).any():
            data_svm[col] = data_svm[col].apply(str_to_column)
    
    print(data_svm.head())
    # Let's start with a Linear kernel:
    #svm_mdl = svm.SVC(kernel="poly")

    X = data_svm[['iemg','variance','waveform length','zero crossing','slope sign change','willison amplitude']]
    Y = data_svm['output']
    #print(X)
    # Model training
    #svm_mdl.fit(X, Y)


if __name__ == "__main__":
    main()


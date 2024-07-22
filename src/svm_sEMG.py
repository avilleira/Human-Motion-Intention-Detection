import pandas as pd
import numpy as np
from sklearn import svm

TRAIN_DATA_PATH = '/home/avilleira/TFG/tfg/data/training_data.csv'


def main():
    
    # Reading from the csv file:
    data_svm = pd.read_csv(TRAIN_DATA_PATH)
    # print data
    print(data_svm.head())


if __name__ == "__main__":
    main()


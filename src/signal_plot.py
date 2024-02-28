import pandas as pd
import numpy as np


def get_data(data_path):
    signal_data = pd.read_csv(data_path)
    # signal_data.iloc[:, 9:16] # Para poder realizar el split en los diferentes csv
    

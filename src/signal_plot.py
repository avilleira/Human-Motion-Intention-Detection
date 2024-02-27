import pandas as pd
import numpy as np

def get_data(data_path):
    signal_data = pd.read_csv(data_path)
    
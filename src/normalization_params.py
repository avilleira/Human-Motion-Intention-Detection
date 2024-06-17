import os
import pandas as pd
from signal_processing import *

ACTION = 'WAK'

def main():
    subjects_ids = []
    all_subjects_data = []

    for i in range(1, 41): # For each subject
        subject_max_contraction = []
        subject = 'Sub' + "{:02}".format(i)
        subjects_ids.append(subject)
        semg_df = get_sEMG_data(subject, ACTION)

        for muscle in semg_df.columns: # For each muscle
            if muscle == 'Time':
                continue
            signal_processing(semg_df, muscle, FILTER_LIST)
            max_contraction = get_max_wak_contraction_value(subject, muscle)
            subject_max_contraction.append(max_contraction)
        
        all_subjects_data.append(subject_max_contraction)

    # Save params in dataframe
    data_cols = [col for col in semg_df.columns if 'Filtered' in col]
    norm_params = pd.DataFrame(all_subjects_data, columns=data_cols, index=subjects_ids)
    
    # Save the list in a CSV
    norm_params.to_csv('../data/normalization_params.csv')

if __name__ == "__main__":
    main()
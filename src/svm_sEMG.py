import ast
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

TRAIN_DATA_PATH = '/home/avilleira/TFG/tfg/data/training_data.csv'
DU_SET_LIST = ['iemg', 'variance', 'waveform length', 'zero crossing', 'slope sign change', 'willison amplitude']


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

def flatten_column(df, col):
        # Convertir la columna en listas si es necesario
    df[col] = df[col].apply(str_to_column)
    
    # Si la columna contiene listas, expandirlas en nuevas columnas
    if df[col].apply(lambda x: isinstance(x, list)).any():
        flattened_df = pd.DataFrame(df[col].tolist(), index=df.index)
        flattened_df.columns = [f"{col}_{i}" for i in range(flattened_df.shape[1])]
        return flattened_df
    else:
        return df[[col]]


def main():

    flattened_columns = []
    
    # Reading from the csv file:
    data_svm = pd.read_csv(TRAIN_DATA_PATH)
    
    # Transforming string columns to list
    for col in data_svm.columns:
        if data_svm[col].apply(is_str_list).any():
            data_svm[col] = data_svm[col].apply(str_to_column)
    
    # Flat columns:
    print("Flattening cols")
    for col in DU_SET_LIST:
        flattened_columns.append(flatten_column(data_svm, col))
    # Let's start with a Linear kernel:
    svm_mdl = svm.SVC(kernel="poly")

    X = pd.concat(flattened_columns, axis=1)
    Y = data_svm['output']

    # The dataset is needed to be split in order to train, validate and test
    # Train and validation: 80% and test: 20%
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.2,
                                                    random_state=42)
    # Splitting training dataset in train and validation
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val,
                                        test_size=0.3, random_state=42)
    # Model training
    print("Starting training")
    svm_mdl.fit(X_train, Y_train)
    # Saving it
    joblib.dump(svm_mdl, '../data/svm_model.joblib')

    # Salida predicha
    y_pred = svm_mdl.predict(X_test)
    # Error en la precisión
    err = 1 - (accuracy_score(Y_test, y_pred))
    print(f"ERROR: {err}")


if __name__ == "__main__":
    main()


import ast
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.multiclass import OneVsOneClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.model_selection import GroupShuffleSplit
from imblearn.under_sampling import RandomUnderSampler


TRAIN_DATA_PATH = '/home/avilleira/TFG/tfg/data/training_data.csv'
DU_SET_LIST = ['iemg', 'variance', 'waveform length', 'zero crossing', 'slope sign change', 'willison amplitude']
OUTPUTS = ['REST', 'STDUP', 'SITDN', 'WAK']
RANDOM_N = 42 # Used to indicate that a random selecting sampling is doing
SVM_DEGREE = 2


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
    

def get_confusion_matrix(mdl, input_test, y_predicted, output_test):
    

    cm = confusion_matrix(output_test, y_predicted)

    # Visualizar la matriz de confusión
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percentage, annot=np.around(cm_percentage, 2), fmt='g', cmap='YlOrBr', xticklabels=OUTPUTS, yticklabels=OUTPUTS)
    plt.xlabel('Predicted Labels', fontsize=18)
    plt.ylabel('True Labels', fontsize=18)
    plt.title(' SVM Confusion Matrix')
    plt.show()


def PCA_transform(input_train):
    
    input_normalize = StandardScaler().fit_transform(input_train)

    PCA_mdl = PCA().fit(input_normalize)

    retained_var = np.cumsum(PCA_mdl.explained_variance_ratio_)

    # Nueva figura
    fig = plt.figure()
    ax = fig.add_subplot()

    # Se muestra los valores de la varianza retenida acumulada
    ax.plot(retained_var)
    ax.set_xlabel("Dimensiones (k-1)")
    ax.set_ylabel("Varianza Explicada Retenida")
    ax.set_title("PCA")
    ax.grid()

    plt.show()


def plot_data_hist(data):

    plt.hist(data, bins='auto')
    plt.title('Output Histogram')

    plt.xlabel('Output', fontsize=15)
    plt.ylabel('Samples', fontsize=15)
    plt.xticks([0, 1, 2, 3], OUTPUTS)

    plt.show()


def balance_dataset(input, output):
    """
    This function includes all the functionality in charge of the balancing,
    scaling of the dataset

    :param input_train: Inputs train dataframe
    :type: pandas.Dataframe()

    :param input_test: Inputs test dataframe
    :type: pandas.Dataframe()

    :param output_train: Outputs test dataframe
    :type: pandas.Dataframe
    """
    # While the dataset is imbalanced, it must be balanced

    # Intialize RandomUnderSampler
    under_sampler = RandomUnderSampler(random_state=RANDOM_N)
    # Standard Scaler    # Intialize RandomUnderSampler
    under_sampler = RandomUnderSampler(random_state=RANDOM_N)
    scaler = StandardScaler()

    # Extracting subject col
    sub_arr = input['subject']
    input_no_sub = input.drop(columns='subject')

    #Balancing an scaling the train dataset
    input_resampled, output_resampled = under_sampler.fit_resample(input_no_sub, output)
    input_resampled = scaler.fit_transform(input_resampled)
    
    # Back to Dataframe
    input_resampled = pd.DataFrame(input_resampled, columns=input_no_sub.columns)
    # Get original indices
    original_indices = under_sampler.sample_indices_
    subject_resampled = sub_arr.iloc[original_indices].reset_index(drop=True)
    input_resampled['subject'] = subject_resampled
    
    output_resampled = pd.Series(output_resampled, name=output.name)

    return input_resampled, output_resampled


def main():

    flattened_columns = []
    
    # Reading from the csv file:
    data_svm = pd.read_csv(TRAIN_DATA_PATH)

    # Polynomial Kernel with degree 2, using OnevsOneClassifier
    svm_mdl = svm.SVC(kernel="poly", degree=4, coef0=1.0, C=1.0)
    svm_mdl = OneVsOneClassifier(svm_mdl)

    # Transforming string columns to list
    for col in data_svm.columns:
        if data_svm[col].apply(is_str_list).any():
            data_svm[col] = data_svm[col].apply(str_to_column)
    
    # Getting subjects array
    sub_arr = data_svm['subject']
    # Deleting that array from dataframe
    data_svm = data_svm.drop('subject', axis=1)
   
    # Flat columns:
    print("Flattening cols")
    for col in DU_SET_LIST:
        flattened_columns.append(flatten_column(data_svm, col))
    
    X = pd.concat(flattened_columns, axis=1)
    # Adding the subject array before balancing
    X['subject'] = sub_arr
    Y = data_svm['output']
    plot_data_hist(Y)
    print("SAMPLES", Y.shape)

    print("Balancing...")
    X, Y = balance_dataset(X, Y)

    print("Finished Balancing")
    plot_data_hist(Y)
    # The dataset is needed to be split in order to train, validate and test
    # Train and validation: 80% and test: 20%. This should be by patients
    gss_train = GroupShuffleSplit(n_splits=1, train_size=.8, random_state=RANDOM_N)
    for train_idx, test_idx in gss_train.split(X, Y, X['subject']):
        X_train_val, X_test = X.iloc[train_idx], X.iloc[test_idx]
        Y_train_val, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]

    # Splitting training dataset in train and validation
    gss_val = GroupShuffleSplit(n_splits=1, train_size=0.9, random_state=RANDOM_N)
    for train_idx_val, val_idx in gss_val.split(X_train_val, Y_train_val, X_train_val['subject']):
        X_train, X_val = X.iloc[train_idx_val], X.iloc[val_idx]
        Y_train, Y_val = Y.iloc[train_idx_val], Y.iloc[val_idx]
    
    # Deleting subject col:
    X_train = X_train.drop(columns=['subject'])
    X_val = X_val.drop(columns=['subject'])
    X_test = X_test.drop(columns=['subject'])
    # Model training
    
    print("Starting training...")
    svm_mdl.fit(X_train, Y_train)
    
    # Saving it
    joblib.dump(svm_mdl, '../models/svm_model.joblib')
    # Evaluation time
    print("Evaluating...")
    # svm_mdl = joblib.load('../models/svm_model.joblib')
    y_pred = svm_mdl.predict(X_val)
    print(f"Accuracy: {accuracy_score(Y_val, y_pred) * 100}")

    print("Test...")
    print(type(X_test))
    y_pred = svm_mdl.predict(X_test)

    acc = accuracy_score(Y_test, y_pred) * 100
    # F1 score
    f1 = f1_score(Y_test, y_pred, average='weighted')
    recall = recall_score(Y_test, y_pred, average='weighted')
    precision = precision_score(Y_test, y_pred, average='weighted')
    print(f"Accuracy: {acc}, F1 Score: {f1}, recall: {recall} precision: {precision}")
    get_confusion_matrix(svm_mdl, X_test, y_pred, Y_test)


if __name__ == "__main__":
    main()


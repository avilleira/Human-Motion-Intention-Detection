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
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN


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
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrBr", xticklabels=OUTPUTS, yticklabels=OUTPUTS)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
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


def balance_dataset(input_train, input_test, input_val, output_train):
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
    # SMOTEENN is balances under and over
    smote_enn = SMOTEENN(random_state=RANDOM_N)
    # Standard Scaler
    scaler = StandardScaler()

    #Balancing an scaling the train dataset
    input_train_resampled, output_train_resampled = smote_enn.fit_resample(input_train, output_train)
    input_train_resampled = scaler.fit_transform(input_train_resampled)
    # Scaling validation
    input_val_resampled = scaler.transform(input_val)
    # Scaling test values
    input_test_resampled = scaler.transform(input_test)

    return input_train_resampled, output_train_resampled, input_test_resampled, input_val_resampled


def main():

    flattened_columns = []
    
    # Reading from the csv file:
    data_svm = pd.read_csv(TRAIN_DATA_PATH)

    # Polynomial Kernel with degree 2, using OnevsOneClassifier
    svm_mdl = svm.SVC(kernel="poly", degree=SVM_DEGREE)
    svm_mdl = OneVsOneClassifier(svm_mdl)

    # Transforming string columns to list
    for col in data_svm.columns:
        if data_svm[col].apply(is_str_list).any():
            data_svm[col] = data_svm[col].apply(str_to_column)
    
    # Flat columns:
    print("Flattening cols")
    for col in DU_SET_LIST:
        flattened_columns.append(flatten_column(data_svm, col))

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
    
    print("Balancing...")
    X_train, Y_train, X_test, X_val = balance_dataset(X_train, X_test, X_val, Y_train)
    print("Finished Balancing")

    print("Starting training...")
    svm_mdl.fit(X_train, Y_train)
    # Saving it
    joblib.dump(svm_mdl, '../data/svm_model.joblib')
    # Evaluation time
    print("Evaluating...")
    y_pred = svm_mdl.predict(X_val)
    print(f"Accuracy: {accuracy_score(Y_val, y_pred) * 100}")
    # Salida predicha
    #svm_mdl = joblib.load('../data/svm_model.joblib')
    print("Test...")
    y_pred = svm_mdl.predict(X_test)
    # Error en la precisión
    acc = accuracy_score(Y_test, y_pred) * 100
    print(f"Accuracy: {acc}")
    get_confusion_matrix(svm_mdl, X_test, y_pred, Y_test)


if __name__ == "__main__":
    main()


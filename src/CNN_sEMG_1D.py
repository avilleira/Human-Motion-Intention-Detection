import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.utils
from torch.utils.data import Dataset, DataLoader, random_split
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import seaborn as sns
import time 


# MACROS

EXIT_SUCCESS = 0

DATA_PATH = "../data/net_data/"
ACTIONS_DICT = {'REST': 0, 'STDUP': 1, 'SITDN': 2, 'WAK': 3}
ACTIONS_LIST = ["REST", "STDUP", "SITDN", "WAK"]

TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15

BATCH_SIZE = 32
EPOCH_N = 100
DROP_N = 0.55
LEARNING_RATE = 0.001


MILLION = 1_000_000

OUTPUTS = ['REST', 'STDUP', 'SITDN', 'WAK']


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            # Save Model
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping triggered!")


# Class used for the Raw matrix dataset
class sEMG_Dataset(Dataset):
    # Declaring the dataset for the CNN
    def __init__(self):
        self.data = []
        self.labels = []
        self.subjects = []

        for action_dir in os.listdir(DATA_PATH):
            dir = os.path.join(DATA_PATH, action_dir)
            for mat_file in os.listdir(dir):
                # If the numpy matrix exists
                if mat_file.endswith('.npy'):
                    
                    file_path = os.path.join(dir, mat_file)
                    # Load the matrixes and store it
                    self.data.append(np.load(file_path))
                    self.labels.append(action_dir)
                    # Storing subject
                    sub = int(mat_file.split("_")[0])
                    self.subjects.append(sub)

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        self.labels = np.array([ACTIONS_DICT[label] for label in self.labels])
        self.subjects = np.array(self.subjects)
    
    
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx) :
        sample = torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0) # Creates an extra dimension
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample.squeeze(0), label
    

    def balance_dataset(self):
        """This function has the role to undersampling the dataset."""

        balanced_data = []
        balanced_labels = []
        balanced_subjects = []

        # extract all the index and store it in their dictionary label
        output_index = {label: np.where(self.labels == label)[0] for label in np.unique(self.labels)}
        # Get the lowest amount of samples category
        min_output = min(len(lab_ind) for lab_ind in output_index.values())

        for index in output_index.values():
            # Ramdomly select
            selected_indices = np.random.choice(index, min_output, replace=False)
            balanced_data.extend(self.data[selected_indices])
            balanced_labels.extend(self.labels[selected_indices])
            balanced_subjects.extend(self.subjects[selected_indices])
        
        # Storing it
        self.data = np.array(balanced_data)
        self.labels = np.array(balanced_labels)
        self.subjects = balanced_subjects


# Net Class
class EMG_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=9, out_channels=16, kernel_size=5, stride=1, padding=1)
        self.dropout = nn.Dropout(DROP_N)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=1)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=1)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(self._get_fc_input_size(), 256)  # Dynamic size
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 4)  # 4 clases

        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)


    def _get_fc_input_size(self):

        with torch.no_grad():
            dummy_input = torch.zeros(1, 9, 300)  # (1, 9, 150)
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = self.pool(F.relu(self.conv4(x)))

            return x.numel()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))  
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv3(x)))  
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv4(x)))  

        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def plot_confusion_matrix(model, test_loader):
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())


    cm = confusion_matrix(all_labels, all_preds)
    # Plot percentage
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=OUTPUTS, yticklabels=OUTPUTS)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Percentage)')
    plt.show()


def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    accuracy = correct / total
    report = classification_report(all_labels, all_preds, output_dict=True)
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']
    return accuracy, precision, recall, f1_score


def plot_data_hist(data, dataset_name):

    plt.rc('font', size=14)         
    plt.rc('axes', titlesize=16)     
    plt.rc('axes', labelsize=14)     
    plt.rc('xtick', labelsize=12)    
    plt.rc('ytick', labelsize=12)    
    plt.rc('legend', fontsize=12)    
    plt.rc('figure', titlesize=18)

    labels = [data[i][1].item() for i in range(len(data))] 

    plt.hist(labels, bins='auto', alpha=0.7, color='blue', edgecolor='black')
    plt.title(f'{dataset_name} Dataset')
    
    plt.xlabel('Output')
    plt.ylabel('Samples')
    
    # Ajustar los ticks del eje X según tus salidas
    plt.xticks(ticks=range(len(OUTPUTS)), labels=OUTPUTS)

    plt.grid(axis='y')
    plt.show()


def split_dataset(data, subjects, labels, dataset_class):
    # GroupShuffleSplit train and rest
    gss = GroupShuffleSplit(n_splits=1, train_size=TRAIN_SIZE, random_state=42)
    
    for train_idx, temp_idx in gss.split(data, labels, groups=subjects):
        
        # Groupshufflesplit for validation and test
        gss_val_test = GroupShuffleSplit(n_splits=1, train_size=VAL_SIZE / (VAL_SIZE + TEST_SIZE), random_state=42)
        val_idx, test_idx = next(gss_val_test.split(data[temp_idx], labels[temp_idx], groups=np.array(subjects)[temp_idx]))
        val_idx = temp_idx[val_idx]
        test_idx = temp_idx[test_idx]

        # Subsets
        train_dataset = torch.utils.data.Subset(dataset_class, train_idx)
        val_dataset = torch.utils.data.Subset(dataset_class, val_idx)
        test_dataset = torch.utils.data.Subset(dataset_class, test_idx)
        
        # Dataloaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        plot_data_hist(train_dataset, dataset_name="Train Dataset")
        plot_data_hist(val_dataset, dataset_name="Validation Dataset")
        plot_data_hist(test_dataset, dataset_name="Test Dataset")

        
        return train_loader, val_loader, test_loader

def train_and_evaluate(cnn_net, train_loader, val_loader, test_loader, optimizer, criterion, EPOCH_N):
    train_losses = []
    val_losses = []

    plt.ion()  # Interactive mode
    fig, ax = plt.subplots()
    line_train, = ax.plot([], [], label='Train Loss', color='blue')
    line_val, = ax.plot([], [], label='Validation Loss', color='orange')
    ax.set_xlim(0, EPOCH_N)
    ax.set_ylim(0, 1) 
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

    plt.draw()

    # Initializing Early Stopping
    early_stopping = EarlyStopping()

    for epoch in range(EPOCH_N):
        # --- Fase de entrenamiento ---
        cnn_net.train()
        epoch_train_loss = 0
        for data, labels in train_loader:
            optimizer.zero_grad() # Reset the gradients
            outputs = cnn_net(data)  # (BATCH_SIZE, 9, 150)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Validation phase ---
        cnn_net.eval()  # Evaluation phase
        epoch_val_loss = 0
        with torch.no_grad(): 
            for data, labels in val_loader:
                outputs = cnn_net(data)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Actualizar gráfica
        line_train.set_xdata(np.arange(1, epoch + 2))
        line_train.set_ydata(train_losses)
        line_val.set_xdata(np.arange(1, epoch + 2))
        line_val.set_ydata(val_losses)
        ax.set_ylim(0, max(max(train_losses), max(val_losses)))
        plt.draw()
        plt.pause(0.01)  # uploading matplot

        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        # Verificar early stopping
        early_stopping(avg_val_loss, cnn_net)
        if early_stopping.early_stop:
            break

    print("Training complete")

    time_1 = time.monotonic_ns()
    test_accuracy, precision, recall, f1_score = evaluate_model(cnn_net, test_loader)
    time_2 = ((time.monotonic_ns() - time_1) / len(test_loader)) / MILLION
    print(f"Inference time: {time_2}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test precision: {precision:.4f}")
    print(f"Test rec: {recall:.4f}")
    print(f"Test f1: {f1_score:.4f}")


    plt.ioff() 
    plt.show()

def main():


    dataset = sEMG_Dataset()
    print("Dataset created")
    plot_data_hist(dataset, "Imbalanced")
    dataset.balance_dataset()
    plot_data_hist(dataset, "Balanced")

    print("Dataset balanced")
    # Splitting Dataset in Train, validation and test
    train_loader, val_loader, test_loader = split_dataset(dataset.data, dataset.subjects, dataset.labels, dataset)
    # Initialize model, optimizer and criterion
    cnn_net = EMG_CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(cnn_net.parameters(), lr=LEARNING_RATE, alpha=0.99)

    train_and_evaluate(cnn_net, train_loader, val_loader, test_loader, optimizer, criterion, EPOCH_N)

    # Confusion matrix
    plot_confusion_matrix(cnn_net, test_loader)

    plt.ioff() 
    plt.show() 

if __name__ == "__main__":
    main()
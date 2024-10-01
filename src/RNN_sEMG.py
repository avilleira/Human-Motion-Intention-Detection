import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.utils
from torch.utils.data import Dataset, DataLoader, random_split
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sEMG_Dataset import sEMG_Dataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix
import seaborn as sns


DATA_PATH = "../data/net_data/"
ACTIONS_DICT = {'REST': 0, 'STDUP': 1, 'SITDN': 2, 'WAK': 3}
ACTIONS_LIST = ["REST", "STDUP", "SITDN", "WAK"]

TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15

BATCH_SIZE = 64
EPOCH_N = 55
DROP_N = 0.5

INPUT_SIZE = 9          
HIDDEN_SIZE = 128        
OUTPUT_SIZE = 4        
LAYERS = 4         

OUTPUTS = ['REST', 'STDUP', 'SITDN', 'WAK']

EXIT_SUCCESS = 0

# LSTM Class
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=4, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM con num_layers capas y tamaño oculto hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3, bidirectional=True)

        # Dropt out 
        self.dropout = nn.Dropout(p=0.3)

        self.activation = nn.LogSoftmax(dim=1)

        self.batch_norm = nn.BatchNorm1d(output_size)
        
        self.fc = nn.Linear(hidden_size * 2, output_size)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() >= 2: # If tensor has 2 dimensions
                    nn.init.xavier_uniform_(param)
                else:
                    print(f"Skipping weight initialization for {name} due to insufficient dimensions")
            elif 'bias' in name:
                nn.init.zeros_(param)


    def forward(self, x):
       
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device) 
        
        # Input to LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # LSTM to output
        out = out[:, -1, :]
        out = self.dropout(out)
        
        out = self.fc(out)
        out = self.batch_norm(out)
        out = self.activation(out)
        return out


def plot_confusion_matrix(model, test_loader):
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.numpy())
            all_preds.extend(predicted.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=OUTPUTS, yticklabels=OUTPUTS)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


def plot_data_hist(data, dataset_name):
    labels = [data[i][1].item() for i in range(len(data))] 

    plt.hist(labels, bins='auto', alpha=0.7, color='blue', edgecolor='black')
    plt.title(f'Distribución de etiquetas en {dataset_name}')
    
    plt.xlabel('Output')
    plt.ylabel('Samples')
    
    plt.xticks(ticks=range(len(OUTPUTS)), labels=OUTPUTS)

    plt.grid(axis='y')
    plt.show()


def split_dataset(data, subjects, labels, dataset_class):
    # GroupShuffleSplit for first division (train vs test)
    gss = GroupShuffleSplit(n_splits=1, train_size=TRAIN_SIZE, random_state=42)
    
    for train_idx, temp_idx in gss.split(data, labels, groups=subjects):
        
        gss_val_test = GroupShuffleSplit(n_splits=1, train_size=VAL_SIZE / (VAL_SIZE + TEST_SIZE), random_state=42)
        val_idx, test_idx = next(gss_val_test.split(data[temp_idx], labels[temp_idx], groups=np.array(subjects)[temp_idx]))
        val_idx = temp_idx[val_idx]
        test_idx = temp_idx[test_idx]

        train_dataset = torch.utils.data.Subset(dataset_class, train_idx)
        val_dataset = torch.utils.data.Subset(dataset_class, val_idx)
        test_dataset = torch.utils.data.Subset(dataset_class, test_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
        return train_loader, val_loader, test_loader


def evaluate_model(model, test_loader):
    model.eval()  
    correct = 0
    total = 0

    with torch.no_grad():  
        for data, labels in test_loader:
            outputs = model(data)  
            _, predicted = torch.max(outputs, 1) 
            total += labels.size(0)
            correct += (predicted == labels).sum().item() 

    accuracy = correct / total
    return accuracy


def train_net(lstm_net, train_loader, val_loader, test_loader, optimizer, criterion):
    train_losses = []
    val_losses = []

    plt.ion()  
    fig, ax = plt.subplots()
    line_train, = ax.plot([], [], label='Train Loss', color='blue')
    line_val, = ax.plot([], [], label='Validation Loss', color='orange')
    ax.set_xlim(0, EPOCH_N)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

    plt.draw()

    for epoch in range(EPOCH_N):
        
        # Changing net to train
        lstm_net.train()
        epoch_train_loss = 0
        
        for data, labels in train_loader:
            optimizer.zero_grad()
            # Data has the form (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)
            outputs = lstm_net(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        lstm_net.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for data, labels in val_loader:
                outputs = lstm_net(data)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item()
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        line_train.set_xdata(np.arange(1, epoch + 2))
        line_train.set_ydata(train_losses)
        line_val.set_xdata(np.arange(1, epoch + 2))
        line_val.set_ydata(val_losses)
        ax.set_ylim(0, max(max(train_losses), max(val_losses))) 
        plt.draw()
        plt.pause(0.01) 

        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    print("Training complete")
    test_accuracy = evaluate_model(lstm_net, test_loader)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    plt.ioff()  
    plt.show()


def main():

    dataset = sEMG_Dataset()
    print("Dataset created")
    dataset.balance_dataset()
    print("Dataset balanced")
   
    train_loader, val_loader, test_loader = split_dataset(dataset.data, dataset.subjects, dataset.labels, dataset)
    lstm_net = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, LAYERS)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lstm_net.parameters(), lr=0.001)

    train_net(lstm_net, train_loader, val_loader, test_loader, optimizer, criterion)



if __name__ == "__main__":
    main()
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


# MACROS

EXIT_SUCCESS = 0
DATA_PATH = "../data/net_data/"
ACTIONS_DICT = {'REST': 0, 'STDUP': 1, 'SITDN': 2, 'WAK': 3}
ACTIONS_LIST = ["REST", "STDUP", "SITDN", "WAK"]

TRAIN_SIZE = 0.7
VAL_SIZE = 0.1
TEST_SIZE = 0.2

# Class used for the Raw matrix dataset
class sEMG_Dataset(Dataset):
    # Declaring the dataset for the CNN
    def __init__(self):
        self.data = []
        self.labels = []
        self.balanced_labels = []

        for action_dir in os.listdir(DATA_PATH):
            dir = os.path.join(DATA_PATH, action_dir)
            for mat_file in os.listdir(dir):
                # If the numpy matrix exists
                if mat_file.endswith('.npy'):
                    file_path = os.path.join(dir, mat_file)
                    # Load the matrixes and store it
                    self.data.append(np.load(file_path))
                    self.labels.append(action_dir)

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        self.labels = np.array([ACTIONS_DICT[label] for label in self.labels])
    
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) :
        sample = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample, label

    def balance_dataset(self):
        """This function has the role to undersampling the dataset."""

        balanced_data = []
        balanced_labels = []

        # extract all the index and store it in their dictionary label
        output_index = {label: np.where(self.labels == label)[0] for label in np.unique(self.labels)}
        # Get the lowest amount of samples category
        min_output = min(len(lab_ind) for lab_ind in output_index.values())

        for index in output_index.values():
            # Ramdomly select
            selected_indices = np.random.choice(index, min_output, replace=False)
            balanced_data.extend(self.data[selected_indices])
            balanced_labels.extend(self.labels[selected_indices])

        self.data = np.array(balanced_data)
        self.labels = np.array(balanced_labels)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch Normalization
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # Batch Normalization
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)  # Batch Normalization
        
        # Ajusta el tamaño aquí basado en la salida de la última capa convolucional
        self.fc1 = nn.Linear(2304, 128)  # Cambia el valor aquí
        self.dropout = nn.Dropout(0.5)  # Dropout con un 50% de desactivación
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = x.unsqueeze(1)  # Añadir una dimensión de canal

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2)

        x = x.view(x.size(0), -1)  # Aplanar
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")


def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")


def main():

    dataset = sEMG_Dataset()
    #count = np.bincount(dataset.labels)
    #plt.bar(ACTIONS_LIST, count)
    #plt.title('Distribución de clases antes del balanceo')
    #plt.xlabel('Clase')
    #plt.ylabel('Cantidad de muestras')
    #plt.show()

    dataset.balance_dataset()

    #count = np.bincount(dataset.labels)
    #plt.bar(ACTIONS_LIST, count)
    #plt.title('Distribución de clases antes del balanceo')
    #plt.xlabel('Clase')
    #plt.ylabel('Cantidad de muestras')
    #plt.show()

    # Spliting dataset into 3: train, validation and test

    train_size = int(TRAIN_SIZE * len(dataset))
    val_size = int(VAL_SIZE * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                    [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    convolutional_net = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(convolutional_net.parameters(), lr=0.001)

    train_model(convolutional_net, train_loader, criterion, optimizer)
    evaluate_model(convolutional_net, test_loader)
    
    return EXIT_SUCCESS



if __name__ == "__main__":
    main()
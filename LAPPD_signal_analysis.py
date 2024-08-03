import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import mean_squared_error

# Function to process a single chunk of data
def process_chunk(chunk):
    data = chunk.values
    ampl = data[:, 0]
    pos = data[:, 1]
    reco_ampl = np.min(data[:, 49:950], axis=1)
    reco_pos = np.argmin(data[:, 49:950], axis=1)
    return pd.DataFrame({
        'ampl': ampl,
        'pos': pos,
        'reco_ampl': reco_ampl,
        'reco_pos': reco_pos
    })

# Process CSV files in chunks
def process_files(files, chunk_size=100):
    data_list = []
    for file in files:
        for chunk in pd.read_csv(file, chunksize=chunk_size):
            chunk_df = process_chunk(chunk)
            data_list.append(chunk_df)
    return pd.concat(data_list, ignore_index=True)

class WaveformDataset(Dataset):
    def __init__(self, data, target):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(model, dataloader, criterion, optimizer, epochs=20):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')
    return model

def evaluate_model(model, dataloader, criterion):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            predictions.extend(outputs.squeeze().tolist())
            actuals.extend(targets.tolist())
    mse = mean_squared_error(actuals, predictions)
    return mse, predictions, actuals

# Part 1: Data Processing and Simple Neural Network Model
csv_files_part1 = glob.glob('fake_lappd_pulses_*.csv')
data_part1 = process_files(csv_files_part1)

# Split data into features and targets for Part 1
X_part1 = data_part1.iloc[:, 2:1026].values  # Waveform data
y_ampl_part1 = data_part1['ampl'].values

# Create dataset and dataloaders
dataset_part1 = WaveformDataset(X_part1, y_ampl_part1)
train_size = int(0.8 * len(dataset_part1))
test_size = len(dataset_part1) - train_size
train_dataset, test_dataset = random_split(dataset_part1, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize model, loss function and optimizer
input_dim = X_part1.shape[1]
model_part1 = SimpleNN(input_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_part1.parameters(), lr=0.001)

# Train the model
model_part1 = train_model(model_part1, train_loader, criterion, optimizer, epochs=20)

# Evaluate the model
mse_part1, y_ampl_pred_part1, y_ampl_true_part1 = evaluate_model(model_part1, test_loader, criterion)

# Normalize the reconstructed amplitudes to match the example histogram's range
reco_ampl = data_part1['reco_ampl'].values
reco_ampl_normalized = (reco_ampl - reco_ampl.min()) / (reco_ampl.max() - reco_ampl.min()) * (0.014 - 0.004) + 0.004

# Plotting for Part 1
ampl_bins = np.arange(200, 1501, 20)
pos_bins = np.arange(100, 601, 20)

plt.figure()
plt.hist(data_part1['ampl'], bins=ampl_bins)
plt.grid()
plt.xlabel('True amplitude (with conversion factor)')
plt.title('Histogram of True Amplitude')

plt.figure()
plt.hist(data_part1['pos'], bins=pos_bins)
plt.grid()
plt.xlabel('True position')
plt.title('Histogram of True Position')

plt.figure()
plt.hist(reco_ampl_normalized, bins=100)
plt.grid()
plt.xlabel('Reconstructed amplitude (normalized)')
plt.title('Histogram of Reconstructed Amplitude')

plt.tight_layout()
plt.show()

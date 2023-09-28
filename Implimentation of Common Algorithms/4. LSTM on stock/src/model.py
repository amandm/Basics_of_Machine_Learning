import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the output of the last time step
        return out

# Create a custom dataset
class CustomDataset(Dataset):
    def __init__(self, data, w):
        self.data = data
        self.w = w

    def __len__(self):
        return len(self.data) - self.w 

    def __getitem__(self, idx):
        window = self.data[idx:idx + self.w]
        target = self.data[idx + self.w]
        return window, target







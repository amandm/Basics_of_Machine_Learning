import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler 
import math

from utils import *
from model import *

# Hyperparameters
input_size = 1  # Input size for each time step
hidden_size = 64  # Number of LSTM units
num_layers = 2  # Number of LSTM layers
batch_size = 64
num_epochs = 50
learning_rate = 0.001
window_size = 60
device = torch.device("mps")
best_model_params_path = "save_dict"


def parse_args():
    """Parse arguments."""
    # Parameters settings
    parser = argparse.ArgumentParser(description="PyTorch implementation of paper 'A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction'")

    # Dataset setting
    parser.add_argument('--dataroot', type=str, default="data/nasdaq100/full/full_non_padding.csv", help='path to dataset')
    parser.add_argument('--stockroot', type=str, default="data/TSLA.csv", help='path to stock data')
    parser.add_argument('--splitratio', type=list, default=[0.8,0.1,0.1], help='train test split, default [0.8,0.1,0.1]')
    parser.add_argument('--window', type=int, default=10, help='window length')
    
    # parse the arguments
    args = parser.parse_args()

    return args

def main():
    """Main pipeline of DA-RNN."""
    args = parse_args()
    # Read dataset
    print("-- Load dataset --")
    ts= read_stock(args.stockroot, debug=False)

    # Calculate the sizes of each split
    total_length = len(ts)
    train_length = math.ceil(len(ts)* args.splitratio[0])
    val_length = math.ceil(len(ts)* args.splitratio[1])
    test_length = math.ceil(len(ts)* args.splitratio[2])
    
    print(f' training_len = {train_length}, val_len = {val_length}')
    # Split the data
    train_data = ts[:train_length]
    test_data = ts[train_length:train_length+val_length]
    val_data = ts[train_length+val_length:]
    
    scaler = MinMaxScaler(feature_range=(0,1))
    train_norm = scaler.fit_transform(train_data.reshape(-1,1))
    train_normd = torch.tensor(train_norm, dtype=torch.float32)
    # train_normd = train_normd.double()
    val_data_norm = scaler.transform(val_data.reshape(-1, 1))
    test_data_norm = scaler.transform(test_data.reshape(-1, 1))
   
    print(type(train_normd),train_normd.dtype)
    
    dataset = CustomDataset(train_norm, window_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = CustomDataset(val_data_norm,window_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = LSTMNet(input_size, hidden_size, num_layers)
    model = model.float()
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model = model.to(device)
    model.train()
    loss_dic = []
    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            inputs = inputs.view(-1, window_size, input_size)
            inputs = inputs.float()
            inputs = inputs.to(device)
            
            targets = targets.float()
            targets = targets.to(device)
            # print(inputs.shape,type(inputs),inputs.dtype)
            outputs = model(inputs)
            # print(outputs.shape,targets.shape)
            
            # print(targets)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            loss_dic.append(loss.item())
        # if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
    
    torch.save(model.state_dict(), best_model_params_path)
    
if __name__ == '__main__':
    main()
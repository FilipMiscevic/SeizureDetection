# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

from torch.utils.data import Dataset, DataLoader
import torch.nn
import pyedflib
import numpy as np
from scipy.signal import spectrogram, welch
from xgboost import XGBClassifier, plot_tree
import matplotlib.pyplot as plt
import re
from datetime import datetime, timedelta
import random
import os


class ChbAutoencoderDataset(Dataset):
    def __init__(
            self,
            data_dir='./chb-mit-scalp-eeg-database-1.0.0/',
            seizures_only=True,
            sample_rate=256,
            sample_length=5, # in seconds
            exclude_test=True,
            extract_features=False):
        # Initialization
        self.sample_rate = sample_rate
        self.sample_length = sample_length
        self.data_dir = data_dir
        self.record_type = 'RECORDS-WITH-SEIZURES' if seizures_only else 'RECORDS'
        self.exclude_test = exclude_test
        self.extract_features = extract_features
                
        with open(self.data_dir+self.record_type) as f:
            self.records = f.read().strip().splitlines()
        if exclude_test:
            test_file = os.path.join(self.data_dir, 'TEST_RECORDS.txt')
            with open(test_file) as f:
                test_records = set(f.read().strip().splitlines())
                records = set(self.records)
            assert len(test_records - records) == 0, test_records - records
            self.records = list(records - test_records)

            
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.records)
        
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        file_name = self.records[index]
        
        with pyedflib.EdfReader(os.path.join(self.data_dir, file_name)) as f:
            n = f.signals_in_file
            # choose one channel
            channel = random.randrange(0, n)

            # chose random starting point
            size = self.sample_length * self.sample_rate
            start_point = random.randrange(0, f.file_duration*self.sample_rate - size)

            sample = f.readSignal(channel, start_point, size)
        
        if self.extract_features:
            features = self.__welch_features(sample)
            sample = sample.flatten()
        
        return torch.Tensor(sample)
    
    def __welch_features(self, sample):
        p_f, p_Sxx = welch(sample, fs=dataset.sample_rate, axis=1)
        p_SS = np.log1p(p_Sxx)
        arr = p_SS[:] / np.max(p_SS)
        return arr

# +

data_dir = os.path.expanduser('~/data/chb-mit-scalp-eeg-database-1.0.0/')
data_dir

# +
dataset = ChbAutoencoderDataset(data_dir=data_dir)

train = dataset[0]
train[0].shape
# -

loader = DataLoader(dataset, batch_size=10)


class Autoencoder(torch.nn.Module):
    def __init__(self, input_shape, num_layers=4, layer_sizes=[1024, 512, 256, 128]):
        super().__init__()
        self.input_shape = input_shape
        print(f"Input shape: {input_shape}")
        self.input_size = 1
        for d in self.input_shape:
            self.input_size = self.input_size * d
        print(f"Input size: {self.input_size}")
        self.num_layers = num_layers
        self.layer_sizes = layer_sizes
        self.encoder = self.__create_encoder()
        self.decoder = self.__create_decoder()
        nn.init.uniform_(layer_1.weight, -1/sqrt(5), 1/sqrt(5))
        
    def __create_encoder(self):
        sizes = [self.input_size] + self.layer_sizes
        print(f'sizes: {sizes}')
        layers = []
        for i in range(self.num_layers):
            layers.append(torch.nn.Linear(sizes[i], sizes[i+1]))
            layers.append(torch.nn.ReLU())
        print(f"layers: {layers}")
        return torch.nn.Sequential(*layers)
    
    def __create_decoder(self):
        sizes = [self.input_size] + self.layer_sizes
        sizes.reverse()
        print(f'sizes: {sizes}')
        layers = []
        for i in range(self.num_layers - 1):
            layers.append(torch.nn.Linear(sizes[i], sizes[i+1]))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(sizes[-2], sizes[-1]))
        print(f"layers: {layers}")
        return torch.nn.Sequential(*layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


test_init = Autoencoder((1000,))
test_init.encoder[0].weight

# +

# Model Initialization
input_shape = 1280
model = Autoencoder(input_shape=(input_shape,), num_layers=4, layer_sizes=[1024, 1024, 512, 512])
  
# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()
  
# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(model.parameters(),
                             lr = 1e-1,
                             weight_decay = 1e-8)

# +
epochs = 100
outputs = []
avg_epoch_losses = []
for epoch in range(epochs):
    random.seed(100)
    epoch_loss = []
    for sample in loader:

        # Output of Autoencoder
        reconstructed = model(sample)

        # Calculating the loss function
        loss = loss_function(reconstructed, sample)

        # The gradients are set to zero,
        # the the gradient is computed and stored.
        # .step() performs parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Storing the losses in a list for plotting
        epoch_loss.append(loss.detach().numpy())

    outputs.append((epoch, sample.detach().numpy(), reconstructed.detach().numpy()))
    avg_epoch_losses.append(np.mean(epoch_loss))
    print(f"done with epoch {epoch} (loss: {avg_epoch_losses[-1]})")
  
# Defining the Plot Style
plt.style.use('fivethirtyeight')
plt.xlabel('epochs')
plt.ylabel('Average Loss')
plt.plot(avg_epoch_losses)
# -

for i in range(9, len(outputs), 10):
    epoch, sample, reconstruction = outputs[i]
    print(f"Epoch {epoch}")
    plt.plot(sample[1])
    plt.show()
    plt.plot(reconstruction[1])
    plt.show()





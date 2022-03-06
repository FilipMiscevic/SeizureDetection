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
            extract_features=False,
            normalize=True):
        # Initialization
        self.sample_rate = sample_rate
        self.sample_length = sample_length
        self.data_dir = data_dir
        self.record_type = 'RECORDS-WITH-SEIZURES' if seizures_only else 'RECORDS'
        self.exclude_test = exclude_test
        self.extract_features = extract_features
        self.normalize = normalize    
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                
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
            sample = self.__read_random_sample(f)
            while np.count_nonzero(sample) < 0.5 * len(sample):
                print("Skipping zero sample")
                sample = self.__read_random_sample(f)
                
        
        if self.normalize:
            sample = self.__normalize(sample)
        
        if self.extract_features:
            features = self.__welch_features(sample)
            sample = sample.flatten()
        
        return torch.Tensor(sample)
    
    def __read_random_sample(self, f):
        n = f.signals_in_file
        # choose one channel
        channel = random.randrange(0, n)

        # chose random starting point
        size = self.sample_length * self.sample_rate
        start_point = random.randrange(0, f.file_duration*self.sample_rate - size)

        return f.readSignal(channel, start_point, size)
    
    def __welch_features(self, sample):
        p_f, p_Sxx = welch(sample, fs=dataset.sample_rate, axis=1)
        p_SS = np.log1p(p_Sxx)
        arr = p_SS[:] / np.max(p_SS)
        return arr
    
    def __normalize(self, sample):
        mean = np.mean(sample)
        return sample - mean


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


run = False
if run:
    data_dir = os.path.expanduser('~/data/chb-mit-scalp-eeg-database-1.0.0/')
    dataset = ChbAutoencoderDataset(data_dir=data_dir, normalize=False)
    loader = DataLoader(dataset, batch_size=10)

if run:
    # Model Initialization
    input_shape = 1280
    model = Autoencoder(input_shape=(input_shape,), num_layers=4, layer_sizes=[1024, 1024, 512, 512])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    # Validation using MSE Loss function
    loss_function = torch.nn.MSELoss()

    # Using an Adam Optimizer with lr = 0.1
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr = 1e-4)

outputs = []
avg_epoch_losses = []

import time
if run:
    epochs = 10000
    run_name = 'basic_autoencoder_1024_1024_512_512'
    model.load_state_dict(torch.load(f"runs/{run_name}_epoch1000.state"))
    for epoch in range(1000, epochs + 1):
        start_time = time.time()
        # random.seed(100)
        epoch_loss = []
        for sample in loader:
            sample = sample.to(device)

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
            epoch_loss.append(loss.detach().cpu().numpy())

        outputs.append((epoch, sample.detach().cpu().numpy(), reconstructed.detach().cpu().numpy()))
        avg_epoch_losses.append(np.mean(epoch_loss))
        print(f"epoch {epoch} (loss: {avg_epoch_losses[-1]}) took {time.time() - start_time} seconds")
        if epoch % 100 == 0:
            torch.save(model.state_dict(), f"runs/{run_name}_epoch{epoch}.state")


if run:
    # Defining the Plot Style
    plt.figure(figsize=(12, 5))
    plt.xlabel('Epochs')
    plt.ylabel('Average Training Loss')
    plt.plot(avg_epoch_losses)
    plt.savefig(f"average_losses_{run_name}.png")

if run:
    for i in range(9900, len(outputs), 1):
        epoch, sample, reconstruction = outputs[i]
        print(f"Epoch {epoch}")
        plt.plot(sample[0])
        plt.plot(reconstruction[0], color="orange")
        plt.show()





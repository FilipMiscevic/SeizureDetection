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
import random
import os
from sklearn import metrics
from chb_utils import parse_summary_file
from chb_import import ChbDataset
from chb_autoencoder import Autoencoder
import time


class FC(torch.nn.Module):
    def __init__(self, input_shape, num_layers=4, layer_sizes=[512, 256, 128, 3]):
        super().__init__()
        self.input_shape = input_shape
        #print(f"Input shape: {input_shape}")
        self.input_size = 1
        for d in self.input_shape:
            self.input_size = self.input_size * d
        #print(f"Input size: {self.input_size}")
        self.num_layers = num_layers
        self.layer_sizes = layer_sizes
        self.model = self.__create_model()
        
    def __create_model(self):
        sizes = [self.input_size] + self.layer_sizes
        #print(f'sizes: {sizes}')
        layers = []
        for i in range(self.num_layers - 1):
            layers.append(torch.nn.Linear(sizes[i], sizes[i+1]))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(sizes[-2], sizes[-1]))
        layers.append(torch.nn.Softmax())
        #print(f"layers: {layers}")
        return torch.nn.Sequential(*layers)
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = torch.reshape(x, (batch_size, -1))
        return self.model(x)


def train_classifier(subject, run_name, input_shape, dataset, loader, epochs=100, save_every=10):
    classifier = FC(input_shape)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(device)
    classifier.to(device)
    encoder.to(device)
    loss_function = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(classifier.parameters(),
                                 lr = 1e-5)
    outputs = []
    avg_epoch_losses = []
    #model.load_state_dict(torch.load(f"runs/{run_name}_epoch1000.state"))
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        # random.seed(100)
        epoch_loss = []
        for sample, label in loader:
            if torch.count_nonzero(sample) < torch.numel(sample) * 0.5:
                # skip samples with > 50% zeros
                continue
            sample = sample.to(torch.float32)
            sample = sample.to(device)

            # Output of Autoencoder
            encoding = encoder(sample)
            prediction = classifier(encoding)

            # Calculating the loss function
            one_hot_labels = torch.nn.functional.one_hot(label, num_classes=3).to(torch.float32)
            #print(one_hot_labels)
            one_hot_labels = one_hot_labels.to(device)
            #print(prediction)
            loss = loss_function(prediction, one_hot_labels)

            # The gradients are set to zero,
            # the the gradient is computed and stored.
            # .step() performs parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Storing the losses in a list for plotting
            epoch_loss.append(loss.detach().cpu().numpy())

        outputs.append((epoch, sample.detach().cpu().numpy(), prediction.detach().cpu().numpy()))
        avg_epoch_losses.append(np.mean(epoch_loss))
        print(f"epoch {epoch} (loss: {avg_epoch_losses[-1]}) took {time.time() - start_time} seconds")
        if epoch % save_every == 0:
            torch.save(classifier.state_dict(), f"runs/{subject}_{run_name}_epoch{epoch}.state")
        dataset.get_windows_for_epoch()
        
    return outputs, avg_epoch_losses


autoencoder_input_shape = 1280
model = Autoencoder(input_shape=(autoencoder_input_shape,), num_layers=4, layer_sizes=[1024, 1024, 512, 512])
autoencoder_run_name = 'basic_autoencoder_1024_1024_512_512'
model.load_state_dict(torch.load(f"runs/{autoencoder_run_name}_epoch10000.state"))
encoder = model.encoder

data_dir = os.path.expanduser('~/data/chb-mit-scalp-eeg-database-1.0.0/')
subjects = set()
with open(os.path.join(data_dir, 'RECORDS-WITH-SEIZURES')) as f:
    for line in f.readlines():
        subject = line.strip().split('/')[0]
        if len(subject) > 0:
            subjects.add(subject)
subjects = sorted(list(subjects))
subjects

run_name = 'classifier_512_256_128_3_v2'
all_avg_epoch_losses = {}
for subject in subjects:
    print()
    print(subject)
    dataset = ChbDataset(data_dir=data_dir, subject=subject, sampler='equal')
    loader = DataLoader(dataset, batch_size=10, shuffle=True)
    input_shape = (23, 512)
    outputs, avg_epoch_losses = train_classifier(subject, run_name, input_shape, dataset, loader, epochs=100)
    all_avg_epoch_losses[subject] = avg_epoch_losses

# +

# Defining the Plot Style
plt.figure(figsize=(10, 5))
plt.xlabel('Epochs')
plt.ylabel('Average Training Loss')
for subject, avg_epoch_loss in all_avg_epoch_losses.items():
    print(subject)
    plt.plot(avg_epoch_loss, label=subject)
plt.savefig(f"average_losses_{run_name}.png")
# -

all_labels = []
all_predictions = []
epoch = 100
run_name = 'classifier_512_256_128_3_v2'
for subject in subjects:
    test_dataset = ChbDataset(data_dir=data_dir, mode='test',subject=subject, multiclass=True, sampler='equal')
    print(test_dataset.records)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    input_shape = (23, 512)
    classifier = FC(input_shape)
    classifier.load_state_dict(torch.load(f"runs/{subject}_{run_name}_epoch{epoch}.state"))
    classifier.eval()
    encoder.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(device)
    classifier.to(device)
    encoder.to(device)
    
    labels = []
    predictions = []
    for sample, label in test_loader:
        if torch.count_nonzero(sample) < torch.numel(sample) * 0.5:
            # skip samples with > 50% zeros
            continue
        sample = sample.to(torch.float32)
        sample = sample.to(device)

        # Output of Autoencoder
        encoding = encoder(sample)
        prediction = classifier(encoding)
        labels.append(label.detach().cpu().numpy())
        predictions.append(prediction.detach().cpu().numpy())
    all_labels.extend(labels)
    all_predictions.extend(predictions)

from sklearn.metrics import confusion_matrix, plot_confusion_matrix, roc_curve, ConfusionMatrixDisplay, RocCurveDisplay,roc_auc_score, f1_score,classification_report
#import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
run = True
if run:
    y_true = np.array(all_labels)
    print(len(y_true))
    print(y_true[2000:2010])
    y_pred = np.array(all_predictions)
    print(y_pred[2000:2010])
    y_pred_class = np.argmax(y_pred, axis=2)
    print(y_pred_class[2000:2010])


    cm = confusion_matrix(y_true, y_pred_class)
    #tn, fp, fn, tp = cm.ravel()

    cm_display = ConfusionMatrixDisplay(cm,display_labels=['Interictal','Preictal','Ictal']).plot()

if run: 
    print(classification_report(y_true,y_pred_class))

if run:
    fpr, tpr, _ = roc_curve(y_true, y_pred_class,pos_label=2)#, pos_label=m.model.classes_[1])
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    r = roc_auc_score(y_true,y_pred_class)
    r2 = roc_auc_score(y_true,y_pred_null)
    print(r,r2)

    #fpr = fp/(fp+tn)
    #print("False positives per day: " + str(fpr/((fp+tn)/256/60)*24))



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

from torch.utils.data import Dataset
import pyedflib
import numpy as np
from scipy.signal import spectrogram, welch
from xgboost import XGBClassifier, plot_tree
from sklearn import metrics
from chb_utils import parse_summary_file
import os
import random


class DataWindow:
    def __init__(self, record_id, record_file, start_index, end_index, label):
        self.record_id = record_id
        self.record_file = record_file
        self.start_index = start_index
        self.end_index = end_index
        self.label = label
        
    def get_data(self):
        with pyedflib.EdfReader(self.record_file) as f:
            channels = f.signals_in_file
            size = self.end_index - self.start_index
            data = np.zeros((channels, size))
            for i in range(channels):
                data[i, :] = f.readSignal(i, self.start_index, size)
        return data


class ChbDataset(Dataset):
    def __init__(self, data_dir='./chb-mit-scalp-eeg-database-1.0.0/',
                 seizures_only=True,sample_rate=256,subject='chb01',mode='train',encoder=None,
                 window_length=5, preictal_length=300, sampler='all', welch_features=False): 
        ### other sampler option is "equal"
        'Initialization'
        self.sample_rate = sample_rate
        self.subject = subject
        self.data_dir = data_dir
        self.window_length = window_length
        self.preictal_length = preictal_length
        self.welch_features = welch_features
        self.sampler = sampler
        self.mode = mode
        self.record_type = 'RECORDS-WITH-SEIZURES' if seizures_only else 'RECORDS'
        self.records = None
        self.preictal = []
        self.ictal = []
        self.interictal = []
        self.windows = []
        random.seed(1000)
        self.get_records() 
        self.get_labeled_windows()
        self.get_windows_for_epoch()
        
    def get_records(self):
        with open(self.data_dir+self.record_type) as f:
            self.records = f.read().strip().splitlines()
            f.close()
            
        with open(self.data_dir+'RECORDS-WITH-SEIZURES') as f:
            self.labelled = f.read().strip().splitlines()
            f.close()
            
        #filter based on subject
        self.records = [record for record in self.records if self.subject in record]
        
        if self.mode == 'train':
            limit_file = os.path.join(self.data_dir, 'TRAIN_RECORDS.txt')
        else:
            limit_file = os.path.join(self.data_dir, 'TEST_RECORDS.txt')
        with open(limit_file) as f:
            limit_records = set(f.read().strip().splitlines())
            records = set(self.records)
        self.records = list(records - limit_records)
    
    def get_labeled_windows(self):
        summary_file = os.path.join(
            self.data_dir,
            self.subject,
            f"{self.subject}-summary.txt")
        all_records = parse_summary_file(summary_file)
        for record in all_records:
            if f"{self.subject}/{record.fileid}" in self.records:
                filename = os.path.join(self.data_dir, self.subject, record.fileid)
                prev_end = 0
                end_of_file = int(record.duration.total_seconds()) * self.sample_rate
                if len(record.seizures) > 0:
                    seizures = []
                    for seizure in record.seizures:
                        preictal_start = max(self.sample_rate * (seizure.start_time - self.preictal_length), 0)
                        ictal_start = self.sample_rate * seizure.start_time
                        ictal_end = self.sample_rate * seizure.end_time
                        self.interictal.extend(self.create_windows_for_segment(
                            record.fileid, filename, prev_end, preictal_start, 0))
                        self.preictal.extend(self.create_windows_for_segment(
                            record.fileid, filename,preictal_start, ictal_start, 1))
                        self.ictal.extend(self.create_windows_for_segment(
                            record.fileid, filename, ictal_start, ictal_end, 2))
                        prev_end = ictal_end
                self.interictal.extend(self.create_windows_for_segment(
                        record.fileid, filename, prev_end, end_of_file, 0))
                
    def create_windows_for_segment(self, recordid, recordfile, start_index, end_index, label):
        windows = []
        window_size = self.window_length * self.sample_rate
        for i in range(start_index, end_index - window_size + 1, window_size):
            windows.append(DataWindow(recordid, recordfile, i, i + window_size, label))
        return windows
                
    def get_windows_for_epoch(self):
        print(self.sampler)
        if self.sampler == 'all':
            self.windows = self.interictal + self.preictal + self.ictal
        elif self.sampler == 'equal':
            num_samples = min([len(self.preictal), len(self.interictal), len(self.ictal)])
            self.windows = random.sample(self.interictal, num_samples) \
                        + random.sample(self.preictal, num_samples) \
                        + random.sample(self.ictal, num_samples)
            print(len(self.windows))
        else:
            raise ValueError("Sampler must be all or equal")
            
    def __len__(self):
        'Denotes the total number of samples'
        if self.sampler == 'all':
            return len(self.preictal) + len(self.interictal) + len(self.ictal)
        elif self.sampler == 'equal':
            smallest = min([len(self.preictal), len(self.interictal), len(self.ictal)])
            return smallest * 3
        else:
            raise ValueError("Sampler must be all or equal")
        
    def __getitem__(self, index):
        'Generates one sample of data, which is one window of length window_length'
        # Select sample
        window = self.windows[index]
        data = window.get_data()
        label = window.label
        if self.welch_features:
            features = self.__welch_features(data)
            data = sample.flatten()
        return data, label
    
    def all_data(self):
        data = [self.__getitem__(i) for i in range(len(self))]
        print(data[0])
        allY = np.concatenate([[x[1] for x in data]])
        allX = np.concatenate([x[0] for x in data])
        return allX,allY
    
    def __welch_features(self, sample):
        p_f, p_Sxx = welch(sample, fs=dataset.sample_rate, axis=1)
        p_SS = np.log1p(p_Sxx)
        arr = p_SS[:] / np.max(p_SS)
        return arr



class XGBoostTrainer:
    def __init__(self):
        self.model = XGBClassifier(objective='binary:hinge', learning_rate = 0.1,
              max_depth = 1, n_estimators = 330)
        self.subjects = ['chb0'+str(i) for i in range(1,10)] + ['chb' + str(i) for i in range(10,25)]
        self.preds = []
        self.labels = []
        
    def train_all(self):
        
        for subject in self.subjects:
            print('Training ' + subject)
            train = ChbDataset(mode='train',subject=subject)
            tests = ChbDataset(mode='test' ,subject=subject)
        
            allX,allY = train.all_data()
            
            self.model.fit(allX, allY)

            for test in tests:
                preds = self.model.predict(test[0])
                self.preds.append((preds))
                self.labels.append(test[1])
                
                print(sum(preds==test[1])/len(test[1]))


data_dir = os.path.expanduser('~/data/chb-mit-scalp-eeg-database-1.0.0/')
d = ChbDataset(data_dir=data_dir, sampler='equal')
len(d.ictal)

d.all_data()

m = XGBoostTrainer()
m.train_all()

# +
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, roc_curve, ConfusionMatrixDisplay, RocCurveDisplay,roc_auc_score, f1_score,classification_report
#import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

y_true = np.concatenate(m.labels)
y_pred_class = np.concatenate(m.preds)

y_pred_null = np.zeros_like(y_pred_class)

cm = confusion_matrix(y_true, y_pred_class)
cm2 = confusion_matrix(y_true, y_pred_null)
tn, fp, fn, tp = cm.ravel()

cm_display = ConfusionMatrixDisplay(cm).plot()
cm_display2 = ConfusionMatrixDisplay(cm2).plot()

# +
#fpr, tpr, _ = roc_curve(y_true, y_pred_class)#, pos_label=m.model.classes_[1])
#roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()

r = roc_auc_score(y_true,y_pred_class)
r2 = roc_auc_score(y_true,y_pred_null)
print(r,r2)

fpr = fp/(fp+tn)
print("False positives per day: " + str(fpr/((fp+tn)/256/60)*24))
# -

print(classification_report(y_true,y_pred_class))
print(classification_report(y_true,y_pred_null))



# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.2
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


# +
class ChbDataset(Dataset):
    def __init__(self, data_dir='./chb-mit-scalp-eeg-database-1.0.0/',
                 seizures_only=True,sample_rate=256,subject='chb01',mode='train'):
        'Initialization'
        self.sample_rate = sample_rate
        self.data_dir = data_dir
        self.record_type = 'RECORDS-WITH-SEIZURES' if seizures_only else 'RECORDS'
                
        with open(self.data_dir+self.record_type) as f:
            self.records = f.read().strip().splitlines()
            f.close()
            
        with open(self.data_dir+'RECORDS-WITH-SEIZURES') as f:
            self.labelled = f.read().strip().splitlines()
            f.close()
            
        #filter based on subject
        self.records = [record for record in self.records if subject in record]
        
        if mode == 'train':
            self.records = self.records[:int(4*len(self.records)/5)]
        elif mode == 'test':
            self.records = self.records[int(4*len(self.records)/5):]
            
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.records)
        
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        file_name = self.records[index]
        
        f = pyedflib.EdfReader(self.data_dir+file_name)
        n = 23 #f.signals_in_file
        signal_labels = f.getSignalLabels()
        sigbufs = np.zeros((n, f.getNSamples()[0]))
        for i in np.arange(n):
                try:
                    sigbufs[i, :] = f.readSignal(i)
                except Exception as e:
                    sigbufs[i, :] = np.zeros(f.getNSamples()[0])
        f.close()
                
        labels = np.zeros((1, f.getNSamples()[0]))
        
       #get labels if seizure. TODO: deal with multiple seizures
        if file_name in self.labelled:
            with open(self.data_dir + file_name.split('/')[0] + '/' + file_name.split('/')[0] + '-summary.txt') as g:
                lines = g.readlines()
                
                found = False
                i = 0
                for line in lines:
                    if file_name.split('/')[1] in line:
                        found = True
                    if found:
                        if i == 4:
                            self.seizure_start = int(line.split(' ')[-2])
                        if i == 5:
                            self.seizure_end   = int(line.split(' ')[-2])   
                            i = 0
                            found  = False        
                            start  = self.sample_rate * self.seizure_start
                            end    = self.sample_rate * self.seizure_end
                            labels[:,start:end] = 1.0
                        i += 1            
        
        s       = 2 #window in seconds
        #print(sigbufs.shape,-sigbufs.shape[1]%(s*self.sample_rate))
        sigbufs = np.concatenate((sigbufs,np.zeros((sigbufs.shape[0],-sigbufs.shape[1]%(s*self.sample_rate)))),axis=1)
        labels = np.concatenate((labels,np.zeros((labels.shape[0],-labels.shape[1]%(s*self.sample_rate)))),axis=1)
        #print(sigbufs.shape)
        
        split   = np.array_split(sigbufs,sigbufs.shape[1]/(s*self.sample_rate),axis=1)
        labels  = [np.any(ss) for ss in np.array_split(labels[0],sigbufs.shape[1]/(s*self.sample_rate))]

        all_X = []
        # calculate the Welch spectrum for each window
        for p_secs in split:
            p_f, p_Sxx = welch(p_secs, fs=self.sample_rate, axis=1)
            p_SS = np.log1p(p_Sxx)
            arr = p_SS[:] / np.max(p_SS)
            all_X.append(arr)
        
        x = np.array(all_X)
        x = x.reshape((x.shape[0],x.shape[1]*x.shape[2]))
        
        #print(signal_labels)
        
        return x,np.array(labels)
    
    def all_data(self):
        data = [self.__getitem__(i) for i in range(len(self.records))]
        allY = np.concatenate([x[1] for x in data])
        
        #[print(x[0].shape) for x in data]
        allX = np.concatenate([x[0] for x in data])
        return allX,allY


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
    
    
# -

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



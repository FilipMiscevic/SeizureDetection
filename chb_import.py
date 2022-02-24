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

# +
from torch.utils.data import Dataset
import pyedflib
import numpy as np
from scipy.signal import spectrogram, welch
from xgboost import XGBClassifier, plot_tree
from sklearn import metrics

from feature_extraction import extract_frames


# -

class ChbDataset(Dataset):
    def __init__(self, data_dir='./chb-mit-scalp-eeg-database-1.0.0/',seizures_only=True,sample_rate=256,subject='chb01',mode='train'):
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
        n = f.signals_in_file
        signal_labels = f.getSignalLabels()
        sigbufs = np.zeros((n, f.getNSamples()[0]))
        for i in np.arange(n):
                sigbufs[i, :] = f.readSignal(i)
                
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
                            self.seizure_start = int(line.split(' ')[3])
                        if i == 5:
                            self.seizure_end   = int(line.split(' ')[3])   
                            i = 0
                            found  = False        
                            start  = self.sample_rate * self.seizure_start
                            end    = self.sample_rate * self.seizure_end
                            labels[:,start:end] = 1.0
                        i += 1
                f.close()
        
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
        
        return x,np.array(labels)
    
    def all_data(self):
        data = [self.__getitem__(i) for i in range(len(self.records))]
        allX = [x[0] for x in data]
        allY = [x[1] for x in data]
        return np.concatenate(np.array(allX)),np.concatenate(np.array(allY))


train_dataset = ChbDataset(mode='train')
test_dataset  = ChbDataset(mode='test')
all_dataset   = ChbDataset(mode='all')

assert len(train_dataset.records)+len(test_dataset.records)==len(all_dataset.records)

allX,allY = train_dataset.all_data()

# +
model = XGBClassifier(objective='binary:hinge', learning_rate = 0.1,
              max_depth = 1, n_estimators = 330)

model.fit(allX, allY)

for test in test_dataset:
    preds = model.predict(test[0])
    print(sum(preds==test[1])/len(test[1]))
# -



from datetime import datetime, timedelta
import re
import random
import math
import pyedflib
import numpy as np
import matplotlib.pyplot as plt


class Seizure:
    def __init__(self, start_time, end_time):
        # in seconds, as strings is fine
        self.start_time = int(start_time)
        self.end_time = int(end_time)
        self.duration = self.end_time - self.start_time

    def __repr__(self):
        return f"Seizure:[{self.start_time},{self.end_time}]"


class ChbFile:
    def __init__(self, fileid, start_time, end_time, channels, seizures):
        # times as strings dd:dd:dd, seizures as list of Seizure objects
        self.fileid = fileid
        self.start_time = self.process_timestring(start_time)
        self.end_time = self.process_timestring(end_time)
        self.channels = channels
        self.seizures = seizures
        self.duration = None if self.start_time is None else self.end_time - self.start_time

    def process_timestring(self, string):
        if string is None:
            return None
        if string[1] == ':':
            string = '0' + string
        add_day = False
        hour = int(string[0:2])
        if hour > 23:
            string = str(hour % 24) + string[2:]
            add_day = True
        
        dt = datetime.strptime(string, "%H:%M:%S")
        if add_day:
            dt = dt + timedelta(days=1)
        return dt

    def __repr__(self):
        start_string = None if self.start_time is None else datetime.strftime(self.start_time, '%H:%M:%S')
        end_string = None if self.end_time is None else datetime.strftime(self.end_time, '%H:%M:%S')
        return f"{self.fileid}[{start_string}, {end_string}]({len(self.seizures)} seizures)"


def parse_summary_file(file):
    with open(file, 'r') as f:
        content = f.read()
        chbfiles = []

        # sample rate
        sample_rate_re = re.compile("Data Sampling Rate: (\d+) Hz")
        m = sample_rate_re.match(content)
        if m:
            sample_rate = int(m.group(1))
            #print(f"sample rate: {sample_rate}")
        else:
            print("sample rate not found")

        # channel names
        channel_re = re.compile("Channel (\d+): (.+)\n")
        matches = channel_re.findall(content)
        channels = [m[1] for m in matches]
        #print(f"channels: {channels}")
        num_channels = len(channels)
        #print(f"found {num_channels} channels")

        # file names and seizure times
        file_re_no_capture = re.compile("File Name: .+\nFile Start Time: \d?\d:\d\d:\d\d\nFile End Time: \d?\d:\d\d:\d\d\nNumber of Seizures in File: \d+\n[Seizure \d*\s?Start Time:\s*\d+ seconds\nSeizure \d*\s?End Time:\s*\d+ seconds\n]*")
        file_re = re.compile("File Name: (.+)\nFile Start Time: (\d?\d:\d\d:\d\d)\nFile End Time: (\d?\d:\d\d:\d\d)\nNumber of Seizures in File: (\d+)\n(Seizure \d*\s?Start Time:\s*(\d+) seconds\nSeizure \d*\s?End Time:\s*(\d+) seconds\n)*")
        matches = file_re_no_capture.findall(content)
        #print(f"found {len(matches)} edf files")
        for match in matches:
            groups = file_re.match(match).groups()
            #print(match)
            #print(groups)
            fileid = groups[0]
            start_time = groups[1]
            end_time = groups[2]
            num_seizures = int(groups[3])
            seizures = []
            if num_seizures > 0:
                seizure_re = re.compile("Seizure \d*\s?Start Time:\s*(\d+) seconds\nSeizure \d*\s?End Time:\s*(\d+) seconds\n")
                s_matches = seizure_re.findall(match)
                #print(s_matches)
                assert len(s_matches) == num_seizures, f"Expected {num_seizures} seizures but found {len(s_matches)} matches"
                for start, end in s_matches:
                    seizures.append(Seizure(start, end))
            chbfiles.append(ChbFile(fileid, start_time, end_time, channels, seizures))
        #print(f"Files {chbfiles}")
        return chbfiles


def parse_summary_file_chb24(file):
    with open(file, 'r') as f:
        content = f.read()
        chbfiles = []

        # sample rate
        sample_rate_re = re.compile("Data Sampling Rate: (\d+) Hz")
        m = sample_rate_re.match(content)
        if m:
            sample_rate = int(m.group(1))
            #print(f"sample rate: {sample_rate}")
        else:
            print("sample rate not found")

        # channel names
        channel_re = re.compile("Channel (\d+): (.+)\n")
        matches = channel_re.findall(content)
        channels = [m[1] for m in matches]
        #print(f"channels: {channels}")
        num_channels = len(channels)
        #print(f"found {num_channels} channels")

        # file names and seizure times
        file_re_no_capture = re.compile("File Name: .+\nNumber of Seizures in File: \d+\n[Seizure \d*\s?Start Time:\s*\d+ seconds\nSeizure \d*\s?End Time:\s*\d+ seconds\n]*")
        file_re = re.compile("File Name: (.+)\nNumber of Seizures in File: (\d+)\n(Seizure \d*\s?Start Time:\s*(\d+) seconds\nSeizure \d*\s?End Time:\s*(\d+) seconds\n)*")
        matches = file_re_no_capture.findall(content)
        #print(f"found {len(matches)} edf files")
        for match in matches:
            groups = file_re.match(match).groups()
            #print(match)
            #print(groups)
            fileid = groups[0]
            start_time = None
            end_time = None
            num_seizures = int(groups[1])
            seizures = []
            if num_seizures > 0:
                seizure_re = re.compile("Seizure \d*\s?Start Time:\s*(\d+) seconds\nSeizure \d*\s?End Time:\s*(\d+) seconds\n")
                s_matches = seizure_re.findall(match)
                #print(s_matches)
                assert len(s_matches) == num_seizures, f"Expected {num_seizures} seizures but found {len(s_matches)} matches"
                for start, end in s_matches:
                    seizures.append(Seizure(start, end))
            chbfiles.append(ChbFile(fileid, start_time, end_time, channels, seizures))
        #print(f"Files {chbfiles}")
        return chbfiles


parse_summary_file_chb24('./chb-mit-scalp-eeg-database-1.0.0/chb24/chb24-summary.txt')


def train_test_split(recordsfile, train_out="TRAIN_RECORDS.txt", test_out="TEST_RECORDS.txt"):
    random.seed(144)
    with open(recordsfile, 'r') as f:
        lines = f.readlines()
    subj_to_file = {}
    for line in lines:
        subject = line.strip().split('/')[0]
        if subject == '':
            continue
        if subject not in subj_to_file:
            subj_to_file[subject] = []
        subj_to_file[subject].append(line)
    for subject, files in subj_to_file.items():
        assert len(files) > 1, f"Subject {subject} has < 2 seizure files"
        random.shuffle(files)
        num_test = math.ceil(len(files)/5)
        test_files = files[0:num_test]
        train_files = files[num_test:]
        print(f"Writing {len(train_files)} files to {train_out}")
        with open(train_out, 'a') as f:
            for file in train_files:
                f.write(file)
        print(f"Writing {len(test_files)} files to {test_out}")
        with open(test_out, 'a') as f:
            for file in test_files:
                f.write(file)


# +
#train_test_split('/home/caroline/data/chb-mit-scalp-eeg-database-1.0.0/RECORDS-WITH-SEIZURES')
# -

def read_edf_file(file_name, sample_length=5):
    with pyedflib.EdfReader(file_name) as f:
        n = f.signals_in_file
        print(f"{n} signals in file {file_name}")
        print(f"length of file: {f.file_duration}")
        print(f"annotations in file: {f.annotations_in_file}")
        sample_rate = f.getSampleFrequencies()[0]
        print(f"sample rate: {sample_rate}")
        size = sample_length * sample_rate
        # chose random starting point
        start_point = random.randrange(0, f.file_duration*sample_rate - size)
        channel_data = []
        for channel in range(n):
            data = f.readSignal(channel, start_point, size)
            channel_data.append(data)
        
        plt.figure(figsize=(12, 5))
        for i, d in enumerate(channel_data):
            plt.plot(d, label=f"{i}")
        plt.show()



# +
#read_edf_file('/home/caroline/data/chb-mit-scalp-eeg-database-1.0.0/chb01/chb01_18.edf')

# + active=""
#

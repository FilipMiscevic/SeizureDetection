from datetime import datetime, timedelta
import re
import random
import math


class Seizure:
    def __init__(self, start_time, end_time):
        # in seconds, as strings is fine
        self.start_time = int(start_time)
        self.end_time = int(end_time)
        self.duration = self.end_time - self.start_time

    def __repr__(self):
        return f"Seizure:[{self.start_time},{self.end_time}]"


class ChbFile:
    def __init__(self, fileid, start_time, end_time, seizures):
        # times as strings dd:dd:dd, seizures as list of Seizure objects
        self.fileid = fileid
        self.start_time = self.process_timestring(start_time)
        self.end_time = self.process_timestring(end_time)
        self.seizures = seizures
        self.duration = self.end_time - self.start_time

    def process_timestring(self, string):
        add_day = False
        if string.startswith('24'):
            string = '00' + string[2:]
            add_day = True
        if string[1] == ':':
            string = '0' + string
        dt = datetime.strptime(string, "%H:%M:%S")
        if add_day:
            dt = dt + timedelta(days=1)
        return dt

    def __repr__(self):
        return f"{self.fileid}[{datetime.strftime(self.start_time, '%H:%M:%S')}, {datetime.strftime(self.end_time, '%H:%M:%S')}]({len(self.seizures)} seizures)"


def parse_summary_file(file):
    with open(file, 'r') as f:
        content = f.read()
        chbfiles = []

        # sample rate
        sample_rate_re = re.compile("Data Sampling Rate: (\d+) Hz")
        m = sample_rate_re.match(content)
        if m:
            sample_rate = int(m.group(1))
            print(f"sample rate: {sample_rate}")
        else:
            print("sample rate not found")

        # channel names
        channel_re = re.compile("Channel (\d+): (.+)\n")
        matches = channel_re.findall(content)
        print(f"found {len(matches)} channels")
        channels = {m[0]: m[1] for m in matches}
        print(f"channels: {channels}")

        # file names and seizure times
        file_re = re.compile("File Name: (.+)\nFile Start Time: (\d?\d:\d\d:\d\d)\nFile End Time: (\d?\d:\d\d:\d\d)\nNumber of Seizures in File: (\d+)\n(Seizure Start Time: (\d+) seconds\nSeizure End Time: (\d+) seconds\n)*")
        matches = file_re.findall(content)
        print(f"found {len(matches)} edf files")
        for match in matches:
            fileid = match[0]
            start_time = match[1]
            end_time = match[2]
            num_seizures = int(match[3])
            seizures = []
            if num_seizures > 0:
                ind = 4
                for i in range(num_seizures):
                    seizures.append(Seizure(match[ind+1], match[ind+2]))
                    ind = ind + 3
            chbfiles.append(ChbFile(fileid, start_time, end_time, seizures))
        print(f"Files {chbfiles}")
        return chbfiles


# +
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
            
        
# -

train_test_split('/home/caroline/data/chb-mit-scalp-eeg-database-1.0.0/RECORDS-WITH-SEIZURES')





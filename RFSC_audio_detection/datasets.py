import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, df, period=10, transforms=None, data_path="train", train=True):
        self.period = period
        self.transforms = transforms
        self.data_path = data_path
        self.train = train
        
        if train: 
            dfgby = df.groupby("recording_id").agg(lambda x: list(x)).reset_index()
            self.recording_ids = dfgby["recording_id"].values
            self.species_ids = dfgby["species_id"].values
            self.t_mins = dfgby["t_min"].values
            self.t_maxs = dfgby["t_max"].values
        else:
            self.recording_ids = df["recording_id"].values

    def __len__(self):
        return len(self.recording_ids)
    
    def __getitem__(self, idx):

        recording_id = self.recording_ids[idx]
        if self.train:
            species_id = self.species_ids[idx]
            t_min, t_max = self.t_mins[idx], self.t_maxs[idx]
        else:
            species_id = [0]
            t_min, t_max = [0], [0]

        y, sr = sf.read(f"{self.data_path}/{recording_id}.flac")
        
        len_y = len(y)
        effective_length = sr * self.period
        rint = np.random.randint(len(t_min))
        tmin, tmax = round(sr * t_min[rint]), round(sr * t_max[rint])

        if len_y < effective_length:
            start = np.random.randint(effective_length - len_y)
            new_y = np.zeros(effective_length, dtype=y.dtype)
            new_y[start:start+len_y] = y
            y = new_y.astype(np.float32)
        elif len_y > effective_length:
            center = round((tmin + tmax) / 2)
            big = center - effective_length
            if big < 0:
                big = 0
            start = np.random.randint(big, center)
            y = y[start:start+effective_length]
            if len(y) < effective_length:
                new_y = np.zeros(effective_length, dtype=y.dtype)
                start1 = np.random.randint(effective_length - len(y))
                new_y[start1:start1+len(y)] = y
                y = new_y.astype(np.float32)
            else:
                y = y.astype(np.float32)
        else:
            y = y.astype(np.float32)
            start = 0
        
        if self.transforms:
            y = self.transforms(samples=y, sample_rate=sr)
            
        start_time = start / sr
        end_time = (start + effective_length) / sr

        label = np.zeros(24, dtype='f')

        if self.train:
            for i in range(len(t_min)):
                if (t_min[i] >= start_time) & (t_max[i] <= end_time):
                    label[species_id[i]] = 1
                elif start_time <= ((t_min[i] + t_max[i]) / 2) <= end_time:
                    label[species_id[i]] = 1
        
        return {
            "waveform" : y,
            "target" : torch.tensor(label, dtype=torch.float),
            "id" : recording_id
        }


class TestDataset(Dataset):
    def __init__(self, df, period=10, transforms=None, data_path="train", train=True):
        self.period = period
        self.transforms = transforms
        self.data_path = data_path
        self.train = train
        
        self.recording_ids = df["recording_id"].values

    def __len__(self):
        return len(self.recording_ids)
    
    def __getitem__(self, idx):

        recording_id = self.recording_ids[idx]
        
        y, sr = sf.read(f"{self.data_path}/{recording_id}.flac")
        
        len_y = len(y)
        effective_length = sr * self.period
        
        y_ = []
        i = 0
        while i < len_y:
            y__ = y[i:i+effective_length]
            
            if self.transforms:
                y__ = self.transforms(samples=y__, sample_rate=sr)
                
            y_.append(y__)
            i = i + effective_length
        
        y = np.stack(y_)

        label = np.zeros(24, dtype='f')
        
        return {
            "waveform" : y,
            "target" : torch.tensor(label, dtype=torch.float),
            "id" : recording_id
        }
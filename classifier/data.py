import sys
sys.path.append('../')

from scipy.io import wavfile
import torch

import spectrogram as spectro


def load_wav(file_path):
    fs, data = wavfile.read(file_path)
    if len(data.shape) == 1:
        return fs, data
    elif len(data.shape) == 2:
        return fs, data[:,0]
    else:
        raise("Invalid file format")
        
def load_data_from_wav(file_path, window_size, overlap=0.5):
    fs, data = load_wav(file_path)
    spectrogram,_,_ = spectro.gen_spectrogram(data, fs, window_size, overlap=overlap)
    return spectrogram

def load_data_list_from_wav(file_paths, window_size, overlap=0.5):
    data_dim = [0, 0]
    temp_data = []
    for file_path in file_paths:
        spectro = load_data_from_wav(file_path, window_size, overlap=overlap)
        temp_data.append((spectro, spectro.shape[1]))
        #Assume same sampling rate (can implement different fs later)
        data_dim[0] = spectro.shape[0]
        data_dim[1] += spectro.shape[1]
        
    #Put data into vectors
    data = torch.zeros(data_dim)
    last = 0
    for dat, length in temp_data:
        data[:,last:last + length] = torch.as_tensor(dat).float()
        last += length
        
    return data

#Normalize to values between 0 and 1
def normalize(x, min_max = None):
    if min_max:
        return (x - min_max[0]) / (min_max[1] - min_max[0])
    else:
        return (x - x.min()) / (x.max() - x.min())
    
#x is an array of 2D tensors
def find_min_max(x):
    return (min([y.min() for y in x]), max([y.max() for y in x]))

def exp_scale(x, y):
    return x**y






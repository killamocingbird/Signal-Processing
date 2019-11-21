import math
import torch

import data
import model1

#File paths for data
voice_data_files = ['../gettysburg.wav', '../preamble.wav']
music_data_files = ['../sample2.wav']

#Spectrogram parameters
window_size = 800
overlap = 0.5

voice_data = data.load_data_list_from_wav(voice_data_files, window_size, overlap=overlap)
music_data = data.load_data_list_from_wav(music_data_files, window_size, overlap=overlap)

#Class balancing
num_samples_per_class = min(voice_data.shape[1], music_data.shape[1])
voice_data = voice_data[:,:num_samples_per_class]
music_data = music_data[:,:num_samples_per_class]

#Make data points into rows
voice_data = torch.as_tensor(voice_data).t().float()
music_data = torch.as_tensor(music_data).t().float()

#Exponentially scale data
voice_data = data.exp_scale(voice_data, 0.2)
music_data = data.exp_scale(music_data, 0.2)

#Normalize
min_max = data.find_min_max([voice_data, music_data])
voice_data = data.normalize(voice_data, min_max=min_max)
music_data = data.normalize(music_data, min_max=min_max)

#Split data into training and validation
train_val_split = 0.9
num_train = math.floor(train_val_split * num_samples_per_class)

xtrain = torch.cat((voice_data[:num_train], music_data[:num_train]))
ytrain = torch.cat((torch.zeros(num_train), torch.ones(num_train)))

xval = torch.cat((voice_data[num_train:], music_data[num_train:]))
yval = torch.cat((torch.zeros(len(xval) // 2), torch.ones(len(xval) // 2)))

#Declare model
epsilon = math.sqrt(window_size / 2) / 2
model = model1.Model1(window_size // 2, epsilon)

#Train model
print("Training")
epochs = 500
model.train(xtrain, ytrain, epochs, lr=5e-2, verbose=True)

#Test model
print("Testing")
print("Accuracy: %.2f" % (100 * model.validate(xval, yval)))










    

    
    
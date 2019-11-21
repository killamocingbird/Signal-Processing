from scipy.io import wavfile
import matplotlib.pyplot as plt
from time import time

import spectrogram as s

print("Sample 1:")
start_time = time()
fs, data = wavfile.read('sample.wav')
spectro, freq_res, time_res = s.gen_spectrogram(data, fs, int(0.02 * fs))
s.plot_spectrogram(spectro, freq_res, time_res, freq_range=(0, 6000), time_range=(1, 4))
plt.title("Speech - Gettysburg Address")
end_time = time()
print("Done: %.2f(s) elapsed" % (end_time - start_time))

print("Sample 2:")
start_time = time()
fs2, data2 = wavfile.read('sample2.wav')
data2 = data2[:,0]
spectro2, freq_res2, time_res2 = s.gen_spectrogram(data2, fs2, int(0.02 * fs2))
s.plot_spectrogram(spectro2, freq_res2, time_res2, freq_range=(0, 6000), time_range=(1, 4))
plt.title("Music - Counting Stars by One Republic")
end_time = time()
print("Done: %.2f(s) elapsed" % (end_time - start_time))

import torch
test = torch.load('C:\\Users\\Justin Wang\\Documents\\Data\\TIMIT\\1\\spectro')
plt.figure(figsize=(10, 10))
plt.imshow(test[::-1])


#music recording
#voice recording

#Make a classifier to given a frame, to be more likely to be music or voice

#||f_0 - classifier|| > epsilon
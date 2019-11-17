from scipy.io import wavfile

import spectrogram as s

fs, data = wavfile.read('sample.wav')
spectro, freq_res, time_res = s.gen_spectrogram(data, fs, int(0.02 * fs))
s.plot_spectrogram(spectro, freq_res, time_res, freq_range=(0, 5000), time_range=(1, 4))
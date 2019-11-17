import numpy as np
import math
import matplotlib.pyplot as plt

import ft

#author: Quinn Collins
#Gets wave amplitudes from a real signal
#[Inputs]   signal: a 1D time-domain numpy array containing the signal to find amplitudes of
#           twiddle_mat: a 2D complex twiddle matrix to use for ffts
#[Outputs]  y: a 1D numpy array containing the various magnitudes of the signals
def get_amplitudes(signal, twiddle_mat = None):
    fft_signal = ft.fft(signal, twiddle_mat)
    return np.abs(fft_signal)
    
    
#author: Justin L. Wang
#Generate spectrogram from signal
#[Inputs]   signal: a 1D time-domain numpy array containing the signal to generate spectrogram for
#           fs: sampling frequency of the data
#           window_size: window size of each dft [in pixel values]
#           overlap: overlap size of each subsequent window [as a percent between 0 and 1]   
#[Outputs]  spectrogram: a spectrogram of signal
#           freq_res: frequency resolution of spectrogram
#           time_res: time resolution of spectrogram
def gen_spectrogram(signal, fs, window_size, overlap=0.5):
    #For speed, generate twiddle matrix of dim (window_size x window_size) for fft
    twiddle_mat = np.array([[math.e**(-2j * math.pi / window_size * a * b) 
                    for b in range(window_size)] 
                    for a in range(window_size)])
    
    #Get overlap number and hop length
    num_overlap = math.ceil(window_size * overlap)
    num_hop = window_size - num_overlap
    
    #Get the frequency and time resolution along with time phase shift
    freq_res = fs / window_size
    time_res = num_hop / fs
    
    spectrogram = np.zeros((math.floor(window_size / 2), math.floor((len(signal) - window_size) / num_hop) + 1))
    
    #Get amplitudes for each window
    for i in range(math.floor((len(signal) - window_size) / num_hop) + 1):
        #Extract dft amplitudes
        window_amps = get_amplitudes(signal[i * num_hop:window_size + i * num_hop], twiddle_mat)
        #Fill spectrogram with first half of dft amplitudes
        spectrogram[:,i] = window_amps[:math.floor(window_size / 2)]
        
    return spectrogram, freq_res, time_res


#author: Justin L. Wang
#Plot a spectrogram
#[Inputs]   spectro: a 2D spectrogram
#           freq_res: frequency resolution of spectrogram
#           time_res: time resolution of spectrogram
#           freq_range: range of frequencies to plot in Hz, e.g. (0, 5000)
#           time_range: range of time to plot in seconds, e.g. (0, 5)
#           log_scale: whether or not to log transform spectrogram values
#           epsilon: small constant to add to spectrogram to prevent log(0)
#           size: size of figure to generate
def plot_spectrogram(spectro, freq_res, time_res, freq_range=None, time_range=None, log_scale=True, epsilon=1e-7, size=(10,10)):
    starting_freq = 0
    starting_time = 0
    
    if freq_range:
        #Trim spectrogram based on frequency range
        freq_lower_bound = math.floor(freq_range[0] / freq_res)
        freq_upper_bound = min(math.ceil(freq_range[1] / freq_res) + 1, spectro.shape[0])  
        
        starting_freq = freq_lower_bound * freq_res
        spectro = spectro[freq_lower_bound:freq_upper_bound]
        
    if time_range:
        #Time spectrogram based on time range
        time_lower_bound = math.floor(time_range[0] / time_res)
        time_upper_bound = min(math.ceil(time_range[1] / time_res) + 1, spectro.shape[1])
        
        starting_time = time_lower_bound * time_res
        spectro = spectro[:,time_lower_bound:time_upper_bound]
    
    #Scale by log to boost color along with epsilon to prevent nans
    if log_scale: spectro = 10 * np.log10(spectro + epsilon)
    
    #Generate figure
    plt.figure(figsize=size)
    ax = plt.subplot(1, 1, 1)
    plt.imshow(spectro[::-1])
    #Scale axises
    xticks = ax.get_xticks()[1:-1]
    yticks = ax.get_yticks()[1:-1]
    plt.xticks(xticks, np.round(xticks * time_res + starting_time, decimals=1))
    plt.yticks(yticks, np.round((yticks * freq_res + starting_freq)[::-1]))
    #Set labels
    plt.xlabel("Time (s)")
    plt.ylabel("Freq (Hz)")
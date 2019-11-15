import numpy as np

import ft
import matplotlib.pyplot as plt

testSignal = np.mat([1, 2, 3, 4, 5])

#author: Quinn Collins
#generating wave amplitudes from complex number matrix
#[Inputs]   signal: a 1D time-domain numpy array containing the signal to perform dft on
            # and further calculate the amplitudes of the dft
#[Outputs]  y: a 1D numpy array containing the various magnitudes of the signals
def gen_amplitudes(signal, twiddle_mat = None):
    fft_signal = ft.fft(signal)
    return[abs(x) for x in fft_signal]
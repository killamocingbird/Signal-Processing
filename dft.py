import numpy as np

def dft(signal, twiddle_mat=None):
    
    l_signal = len(signal)
    
    if twiddle_mat is None:
        #Generate twiddle matrix
        twiddle_mat = np.full((l_signal, l_signal), 0j)
        for a in range(l_signal):
            for b in range(l_signal):
                twiddle_mat[a][b] += np.e**(-2j*(np.pi)*(a)*(b)/l_signal)
                
    #Return matrix multiplication
    return np.matmul(twiddle_mat, signal)
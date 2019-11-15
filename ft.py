import math
import numpy as np

import utils

#author: Justin L. Wang
#Implementation of Cooley-Tukey fast fourier transform algorithm
#[Inputs]   signal: a 1D time-domain numpy array containing the signal to perform fft on
#                   a 2D batch of time-domain signals with dim num_signals x signal_length
#           twiddle_matrix: matrix of size len(signal) x len(signal) to sample twiddles from
#[Outputs]  y: a 1D complex valued numpy array containing the dft of x
def fft(signal, twiddle_mat=None):
    #Check dimensions
    if len(signal) < 1:
        raise Exception("Dimension error")
    
    #Allow for batching
    if len(signal.shape) == 1:
        l_signal = signal.shape[0]
    elif len(signal.shape) == 2:
        l_signal = signal.shape[1]
    else:
        raise Exception("Dimension error")
    
    #Extract factor from the length of x
    factor1 = utils.next_factor(l_signal)
    if factor1 == l_signal:
        #The length of x is prime, proceed to do regular dft
        return dft(signal, twiddle_mat)
    
    #Second factor
    factor2 = l_signal / factor1
    
    #Rearrange into 2D matrix for divide-and-conquer
    M = np.reshape(signal, (factor1, factor2, -1))
    
    #Compute dft across columns
    for col in range(factor2):
        M[:,:,col] = fft(M[:,:,col], twiddle_mat)
        
    if twiddle_mat==None:
        #Compute twiddle matrix
        T = np.array([[math.e**(-2j * math.pi / len(signal) * a * b) 
                    for b in range(factor2)] 
                    for a in range(factor1)])
    else:
        #Draw twiddle from matrix
        T = twiddle_mat[:factor1,:factor2]
        
    #Apply inner product
    M = M * T
    
    #Compute dft across rows
    for row in range(factor1):
        M[:,row,:] = fft(M[:,row,:], twiddle_mat)
    
    #Transpose and unravel
    y = np.reshape(np.transpose(M), (-1, l_signal))
    return y


#author: Quinn Collins
#Implementation of Cooley-Tukey discrete fourier transform algorithm
#[Inputs]   signal: a 1D time-domain numpy array containing the signal to perform dft on
#                   a 2D batch of time-domain signals with dim num_signals x signal_length
#           twiddle_matrix: matrix of size len(signal) x len(signal) to sample twiddles from
#[Outputs]  y: a 1D complex valued numpy array containing the dft of x
def dft(signal, twiddle_mat=None):
    
    #Allow for batching
    if len(signal.shape) == 1:
        l_signal = signal.shape[0]
    elif len(signal.shape) == 2:
        l_signal = signal.shape[1]
    else:
        raise Exception("Dimension error")
    
    if twiddle_mat is None:
        #Generate twiddle matrix
        twiddle_mat = np.full((l_signal, l_signal), 0j)
        for a in range(l_signal):
            for b in range(l_signal):
                twiddle_mat[a][b] += np.e**(-2j*(np.pi)*(a)*(b)/l_signal)
                
    #Return matrix multiplication
    return np.matmul(twiddle_mat, np.transpose(signal))

#author: Justin L. Wang

import dft
import math
import numpy as np

#Implementation of Cooley-Tukey fast fourier transform algorithm
#[Inputs]   signal: a 1D time-domain numpy array containing the signal to perform fft on
#           twiddle_matrix: matrix of size len(signal) x len(signal) to sample twiddles from
#[Outputs]  y: a 1D complex valued numpy array containing the dft of x
def fft(signal, twiddle_mat=None):
    #Check dimensions
    if len(signal.shape) != 1 or len(signal) < 1:
        raise Exception("Dimension error")
    
    #Extract factor from the length of x
    factor1 = next_factor(len(signal))
    if factor1 == len(signal):
        #The length of x is prime, proceed to do regular dft
        return dft.dft(signal, twiddle_mat)
    
    #Second factor
    factor2 = len(signal) / factor1
    
    #Rearrange into 2D matrix for divide-and-conquer
    M = np.reshape(signal, (factor1, factor2))
    
    #Compute dft across columns
    for col in range(factor2):
        M[:,col] = fft(M[:,col], twiddle_mat)
        
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
        M[row,:] = fft(M[row,:], twiddle_mat)
    
    #Transpose and unravel
    y = np.reshape(np.transpose(M), (-1))
    return y


#Finds the next prime factor of an integer x
#[Inputs]   x: an integer
#[Outputs]  y: the first integer factor of x not including 1
def next_factor(x):
    for i in range(2, math.ceil(math.sqrt(x))):
        if i%x == 0:
            return i
    return x

print(dft.dft(np.arange(15)))





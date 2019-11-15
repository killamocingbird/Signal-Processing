#author: Justin L. Wang

import dft
import math
import numpy as np

#Implementation of Cooley-Tukey fast fourier transform algorithm
#[Inputs]   x: a 1D numpy array containing the signal to perform fft on
#[Outputs]  y: a 1D complex valued numpy array containing the dft of x
def fft(x):
    #Check dimensions
    if len(x.shape) != 1 or len(x) < 1:
        raise Exception("Dimension error")
    
    #Extract factor from the length of x
    factor1 = next_factor(len(x))
    if factor1 == len(x):
        #The length of x is prime, proceed to do regular dft
        return dft.dft(x)
    
    #Second factor
    factor2 = len(x) / factor1
    
    #Rearrange into 2D matrix for divide-and-conquer
    M = np.reshape(x, (factor1, factor2))
    
    #Compute dft across columns
    for col in range(factor2):
        M[:,col] = dft.dft(M[:,col])
        
    #Compute and apply twiddle matrix
    T = np.array([[math.e**(-2j * math.pi / len(x) * a * b) 
                for b in range(factor2)] 
                for a in range(factor1)])
    #Apply inner product
    M = M * T
    
    #Compute dft across rows
    for row in range(factor1):
        M[row,:] = dft.dft(M[row,:])
    
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




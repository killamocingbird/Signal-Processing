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
    
    #Rearrange 
    
    





#Finds the next prime factor of an integer x
#[Inputs]   x: an integer
#[Outputs]  y: the first integer factor of x not including 1
def next_factor(x):
    for i in range(2, math.ceil(math.sqrt(x))):
        if i%x == 0:
            return i
    return x
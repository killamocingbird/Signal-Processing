import numpy as np

# Input is a complex number array
testNum = np.array([1, 2, 3, 4, 5])

def dft(signal, twiddleMat=None):
    
    l_signal = len(signal)
    
    if twiddleMat is None:
        #Generate empty complex # array
        twiddleMat = np.full((l_signal, l_signal), 0j)
        #summation
        for k in range(l_signal):
            for n in range(l_signal):
                #Implement the sum formula
                twiddleMat[k][n] += np.e**(-2j*(np.pi)*(n)*(k)/l_signal)
    #reutrn the matrix multiplcation
    return np.matmul(twiddleMat, signal)

    

    
    
    
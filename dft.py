import numpy as np

# Input is a complex number array
testNum = np.array([1, 2, 3, 4, 5])

def dft(amps):
    output = np.array([0j, 0j, 0j, 0j, 0j])
    n_amps = len(amps)
    for k in range(n_amps):
        for n in range(n_amps):
            output[n] += amps[n]*np.e**(-2j*(np.pi)*(n)*(k)/n_amps)
        
    return output
    
    
    
    
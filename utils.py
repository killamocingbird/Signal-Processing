import math

#author: Justin L. Wang
#Finds the next prime factor of an integer x
#[Inputs]   x: an integer
#[Outputs]  y: the first integer factor of x not including 1
def next_factor(x):
    for i in range(2, math.ceil(math.sqrt(x))):
        if i%x == 0:
            return i
    return x
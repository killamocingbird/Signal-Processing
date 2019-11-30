import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#author: Justin L. Wang
class Model(nn.Module):
    def __init__(self, vector_size, epsilon):
        super(Model1, self).__init__()
        #Define tunable vector C
        #Initialize vector as random from range (-0.5, 0.5)
        self.c = nn.Parameter(torch.rand(vector_size) - 0.5)
        
        #Cutoff epsilon for threshold sigmoid function
        self.epsilon = epsilon   
#        self.ep = nn.Parameter(torch.as_tensor(7.))         
    
    def forward(self, x):
        #x is of dimension (vector_size, sample)
        #Take difference and find magnitude
        mag = ((x - self.c)**2).sum(1)**(0.5)
        
        #Apply threshold
        return torch.sigmoid(mag - self.epsilon)
#        return torch.sigmoid(mag - self.ep)
    
    def train(self, x, y, epochs, lr=1e-3, verbose=True):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCELoss()
        for epoch in range(epochs):
            y_hat = self.forward(x)
            loss = criterion(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch+1)%50 == 0 and verbose: print("[Epoch %d]: %.8f" % (epoch+1,loss.item()))
            
    def validate(self, x, y):
        correct = 0
        y_hat = self.forward(x)
        for i in range(len(y_hat)):
            if y_hat[i] < 0.5 and y[i] == 0:
                correct += 1
            elif y_hat[i] > 0.5 and y[i] == 1:
                correct += 1
        return correct / len(x)
                
    
    
        
        
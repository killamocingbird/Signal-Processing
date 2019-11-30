import torch

#author: Justin L. Wang
class Model():
    def __init__(self, in_features, num_centroids):
        #Define template for centroids
        self.centroids = torch.zeros(num_centroids, in_features)
    
    def forward(self, x):
        distances = ((self.centroids - x)**2).sum(1)**(0.5)
        return torch.argmin(distances).item()
    
    #Finds centroids from data
    #[Inputs]   data_points: a matrix of dimension (n x in_features)
    #           centroids: a vector of length n of integers from 0 to num_centroids - 1  
    def train(self, data_points, centroids):
        for i in range(len(self.centroids)):
            self.centroids[i] = data_points[centroids==i].mean(0)
    
    def validate(self, x, y):
        correct = 0
        for i in range(len(x)):
            if self.forward(x[i]) == y[i]:
                correct += 1
        return correct / len(x)
        

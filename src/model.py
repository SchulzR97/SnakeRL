import torch
import numpy as np

class Snake_CNN(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(3,3), stride=1)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=(2,2), stride=1)
        #self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=(2,2), stride=1)
        self.avgpool = torch.nn.AvgPool2d(kernel_size=(4,4), stride=2)
        self.flatten = torch.nn.Flatten(start_dim=1)
        self.fc1 = torch.nn.Linear(10368, 15000)
        self.fc2 = torch.nn.Linear(15000, 10000)
        #self.fc3 = torch.nn.Linear(2200, 1800)
        self.fc4 = torch.nn.Linear(10000, 4)
        self.relu = torch.nn.PReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-6)
        self.criterion = torch.nn.MSELoss()

    def forward(self, X):
        Y = self.relu(self.conv1(X))
        Y = self.relu(self.conv2(Y))
        #Y = self.sigmoid(self.conv3(Y))
        Y = self.flatten(Y)
        Y = self.relu(self.fc1(Y))
        Y = self.relu(self.fc2(Y))
        #Y = self.relu(self.fc3(Y))
        Y = self.fc4(Y)
        
        return Y
    
    def fit(self, states, targets):
        self.optimizer.zero_grad()

        prediction = self(states)
        loss = self.criterion(prediction, targets)
        #loss = torch.nn.functional.binary_cross_entropy_with_logits(prediction, targets)
        loss.backward()
        self.optimizer.step()
        #self.eval()
        return loss.item()
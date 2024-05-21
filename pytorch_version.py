import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
#import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Lambda
from torch.utils.data import DataLoader
from tqdm import tqdm

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.d1 = nn.Linear(in_features=784, out_features=128)
        self.relu1 = nn.ReLU()
        self.d2 = nn.Linear(in_features=128, out_features=64)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(in_features=64, out_features=10)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0004)

    def forward(self, x):
        x = self.d1(x)
        x = self.relu1(x)
        x = self.d2(x)
        x = self.relu2(x)
        x = self.out(x)
        return x

    def fit(self, x, y):
        self.optimizer.zero_grad()
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, x):
        with torch.no_grad():
            return torch.argmax(model.forward(x), axis=-1)

    def train(self, dataloader_train, epoch):
        for i in range(epoch):
            loss = 0
            for xs, ys in dataloader_train:
                loss += model.fit(xs, ys)
            print(f"Epoch Loss: {loss/len(dataloader_train)}")

    def test(self, dataloader_test):
        true = 0
        for xs, ys in dataloader_test:
            y_pred = model.predict(xs)

            true += (y_pred == ys).sum()

        return true/(len(dataloader_test) * 10)


transforms = Compose([ToTensor(), Lambda(lambda image: image.flatten())])
training_set = MNIST(root='./', download=True, train=True, transform=transforms)
testing_set = MNIST(root='./', download=True, train=False, transform=transforms)

model = Network()
dataloader_train = DataLoader(training_set, batch_size=10)
dataloader_test = DataLoader(testing_set, batch_size=10)

model.train(dataloader_train, 20)
print(model.test(dataloader_test))
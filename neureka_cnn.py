# -*- coding: utf-8 -*-

# Usual imports
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np

#load data

X = np.abs(np.load("/home/lorenzo/Downloads/Neureka-challenge/mini_train_data.npy"))
y = np.abs(np.load("/home/lorenzo/Downloads/Neureka-challenge/mini_train_labels.npy"))

tensor_x = torch.from_numpy(X).float()
tensor_y = torch.from_numpy(y).long()
my_dataset = data.TensorDataset(tensor_x,tensor_y) 
training_loader = data.DataLoader(my_dataset)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
          nn.Conv2d(20, 32, kernel_size=5),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
          nn.Conv2d(32, 64, kernel_size=5),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Sequential(
          nn.Linear(in_features=3680, out_features=256),
          nn.ReLU())
        self.fc2 = nn.Sequential(
          nn.Linear(in_features=256, out_features=2),
          nn.ReLU())

    def forward(self, x):
        
        x = self.layer1(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = ConvNet().to(device)

epochs = 10
learning_rate = 0.001
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(list(net.parameters()), lr = learning_rate)

def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
def evaluate(dataloader):
    total, correct = 0,0
    net.eval()
    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    return 100 * correct / total
    
net.apply(init_weights)

loss_epoch_array = []
loss_epoch = 0
train_accuracy = []
test_accuracy = []
for epoch in range(epochs):
    loss_epoch = 0
    for i, data in enumerate(training_loader, 0):
        net.train()
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
    loss_epoch_array.append(loss_epoch)
    train_accuracy.append(evaluate(training_loader))
    #test_accuracy.append(evaluate(test_loader))
    print("Epoch {}: loss: {}, train accuracy: {}".format(epoch + 1, loss_epoch_array[-1], train_accuracy[-1]))


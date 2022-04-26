#!/usr/bin/env python3


# -*- coding: utf-8 -*-
"""
Created on Tue May 26 08:40:48 2020

@author: olhartin@asu.edu
"""
##  https://nextjournal.com/gkoehler/pytorch-mnist
import sys
import os
import torch
import torchvision
import numpy
from itertools import zip_longest
from data_loader import data_loader, alphabet_string
from sklearn.model_selection import train_test_split
##
##  parameters
##
## loop over the entire dataset
n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.001    ## set learning rate
momentum = 0.5          ## momentum parameter
log_interval = 100

if not os.path.exists('./results'): os.mkdir('results')

random_seed = 1         ## this will make runs repeatable
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

X, y = data_loader('./PlateImages_only/data.csv')
print(X.shape, y.shape)
X, y = torch.tensor(X), torch.LongTensor(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = random_seed)





##
##      look at some data
##
examples = enumerate(zip(X_test, y_test))
##
import matplotlib.pyplot as plt

fig = plt.figure()

for batch_idx, (example_data, example_targets) in examples:
    if batch_idx > 5: break
    plt.subplot(2, 3, batch_idx + 1)
    plt.tight_layout()
    plt.imshow(example_data[0][0], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(alphabet_string[example_targets]))
    plt.xticks([])
    plt.yticks([])

fig.savefig('ground_truth.png', dpi = 600)

##
##      CNN
##
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
##
##      set up neural network
##
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(7480, 1000)
        self.fc2 = nn.Linear(1000, 36)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1,7480)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
##
##      initialize the network and optimizer
##
network = Net()
##  network.cuda() ## send problem to GPU
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
criterion = nn.CrossEntropyLoss()
##
##      keep track
##
train_losses = []
train_counter = []
test_losses = []
test_acc = []
test_counter = [i*(len(X_train)) for i in range(n_epochs + 1)]
print(test_counter)
##
##      functions for training and testing
##
def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(zip(X_train, y_train)):
        optimizer.zero_grad()
        output = network(data.float())
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            train_losses.append(loss.item())
            train_counter.append((epoch-1)*len(X_train)*len(data) + batch_idx*len(data))
            torch.save(network.state_dict(), 'results/model.pth')
            torch.save(optimizer.state_dict(), 'results/optimizer.pth')
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
              epoch, batch_idx, len(X_train),
              loss.item()))
##
def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in zip(X_test, y_test):
            output = network(data.float())
            loss = criterion(output, target)
            test_loss += loss.item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(X_test)
        test_losses.append(test_loss)
        acc = 100. * correct / len(X_test)
        test_acc.append(acc)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(X_test),
        acc))
##
##      Run it
##
import time
tbeg = time.time()
test()  ## evaluate with random weights
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()  ## now with trained weights


with open("training_results.txt", 'w') as f:
    f.write('# samples_seen_in_train train_losses sample_seen_in_test test_losses test_accuracy (%)\n')
    for trc, trl, tec, tel, tea in zip_longest(train_counter, train_losses, test_counter, test_losses, test_acc):
        if tec:
            f.write("{} {:.4g} {} {:.4g} {:.4g}\n".format(trc, trl, tec, tel, tea))
        else:
            f.write("{} {:.4g}\n".format(trc, trl))


##
##      plot the results
##
fig, ax = plt.subplots()
ax.plot(train_counter, train_losses, color = 'C0', label = 'Train loss')
ax.scatter(test_counter, test_losses, color = 'C1', marker = 'o', label = 'Test loss')
ax.legend()
ax.set_xlabel('Number of training examples seen')
ax.set_ylabel('Cross entropy loss')
fig.savefig("Loss_vs_iterations.png", dpi = 300)
##
##      predictions from examples?
##
# with torch.no_grad():
#     output = network(example_data)
# fig = plt.figure()
# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.tight_layout()
#     plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#     plt.title("Prediction: {}".format(
#     output.data.max(1, keepdim=True)[1][i].item()))
#     plt.xticks([])
#     plt.yticks([])
# fig
# ##
# ##      continued training
# ##
# continued_network = Net()
# continued_optimizer = optim.SGD(network.parameters(), lr=learning_rate,
#                                 momentum=momentum)
# ##
# ##      load the state of the network from when we last saved it
# ##
# network_state_dict = torch.load('results/model.pth')
# continued_network.load_state_dict(network_state_dict)
# 
# optimizer_state_dict = torch.load('results/optimizer.pth')
# continued_optimizer.load_state_dict(optimizer_state_dict)
# ##
# ##      continue training where we left off
# ##
# 
# for i in range(4,9):
#     # test_counter.append(i*len(train_loader.dataset))
#     train(i)
#     test()
# tend = time.time()
# print('Total run time ', tbeg, tend, tend-tbeg)
# ##
# ##      let's look at our progress
# ##
# fig = plt.figure()
# plt.plot(range(len(X_train)), train_losses, color='blue', label = "Train_loss")
# plt.scatter(range(len(X_test)), test_losses, color='red', label = "Test loss")
# plt.legend(loc='upper right')
# plt.xlabel('number of training examples seen')
# plt.ylabel('negative log likelihood loss')
# 

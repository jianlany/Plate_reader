#!/usr/bin/env python3
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, random_split, DataLoader
import numpy
import time
from itertools import zip_longest
from data_loader import data_loader, alphabet_string
from sklearn.model_selection import train_test_split


##
##      CNN
##
##
##      set up neural network
##
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(3060, 300)
        self.fc2 = nn.Linear(300, 36)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1,3060)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

def main():
    ##
    ##  parameters
    ##
    ## loop over the entire dataset
    n_epochs = 15
    learning_rate = 3e-6
    momentum = 0.5          ## momentum parameter
    log_interval = 100
    output_dir = './results'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    if not os.path.exists(output_dir): os.mkdir(output_dir)

    random_seed = 1         ## this will make runs repeatable
    torch.manual_seed(random_seed)

    X, y = data_loader('./PlateImages_only/data.csv', device = device)
    X = X[len(X), numpy.newaxis, ...]
    y = y[..., numpy.newaxis]
    X = torch.tensor(X, device = device, dtype = torch.float32)
    y = torch.tensor(y, device = device, dtype = torch.long)
    """
    dataset = TensorDataset(X, y)
    train_size = int(0.8*len(dataset))
    test_size = len(dataset) - train_size

    generator = torch.Generator(device = 'cpu')
    generator.manual_seed(random_seed)

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], 
                    generator = generator)

    batch_size = 1
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_seed)
    train_size = len(X_train)
    test_size = len(X_test)
    train_loader = zip(X_train, y_train)
    test_loader = zip(X_test, y_test)
    # sys.exit(0)





    ##
    ##      look at some data
    ##
    ## examples = enumerate(zip(X_test, y_test))
    ##
    import matplotlib.pyplot as plt
    """
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
    """

    ##
    ##      initialize the network and optimizer
    ##
    network = Net().to(device)
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
    test_counter = [i*(train_size) for i in range(n_epochs + 1)]
    ##
    ##      functions for training and testing
    ##
    def train(epoch):
        network.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = network(data)
            # print("Forward pass done")
            loss = criterion(output, target)
            # print("Loss calculated")
            loss.backward()
            # print("Backward pass done")
            optimizer.step()
            # print("Optimizer step done")
            if batch_idx % log_interval == 0:
                train_losses.append(loss.item())
                num_sample_seen = (epoch-1)*len(train_loader)*len(data) + batch_idx*len(data)
                train_counter.append(num_sample_seen)
                print('Train Epoch: {} [{}/{}]\tLoss: {:.6g}'.format(
                  epoch, batch_idx, len(train_loader),
                  loss.item()))
        torch.save(network.state_dict(), '{}/model_{}.pth'.format(output_dir, epoch*len(train_loader)))
        torch.save(optimizer.state_dict(), '{}/optimizer_{}.pth'.format(output_dir, epoch*len(train_loader)))
    ##
    def test():
        network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            num_images = 0
            for _, (data, target) in enumerate(test_loader):
                output = network(data)
                loss = criterion(output, target)
                test_loss += loss.item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
                num_images += len(data)
            test_loss /= len(test_loader)
            test_losses.append(test_loss)
            acc = 100. * correct / num_images 
            test_acc.append(acc)
            print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, num_images,
                acc))
    ##
    ##      Run it
    ##
    tbeg = time.time()
    test()  ## evaluate with random weights
    for epoch in range(1, n_epochs + 1):
        tbeg_epoch = time.time()
        train(epoch)
        test()  ## now with trained weights
        print('Epoch {} cost time {:.2f} seconds'.format(epoch, time.time()- tbeg_epoch))

    print('Total training time: {:.2f} seconds'.format(time.time()-tbeg))

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
if __name__ == "__main__":
    main()

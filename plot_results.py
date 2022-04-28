#!/usr/bin/env python3
import matplotlib.pyplot as plt
plt.style.use('paper')
import numpy

def plot():
    training_seen, train_losses = numpy.genfromtxt('training_results.txt', skip_header = 1, usecols = (0,1)).T
    test_seen, test_losses, acc = numpy.genfromtxt('test_result.txt', skip_header = 1, usecols = (0,1,2)).T
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    lns1 = ax.plot(training_seen, train_losses, label = 'train losses')
    lns2 = ax.plot(test_seen, test_losses, 'o', label = 'test losses')
    lns3 = ax2.plot(test_seen, acc, 'o-', color = 'C2', label = 'accuracy')
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax.set_xlabel('Number of samples seen')
    ax.set_ylabel('Loss')
    ax2.set_ylabel('accuracy (%)')
    ax.legend(lns, labs, loc = 'center right')
    fig.savefig('training_result.png', dpi = 600)


if __name__ == "__main__":
    plot()

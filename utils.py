# I want to draw a loss curve
import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curve(losses: list, title: str = 'loss curve', save_path: str = None):
    plt.figure()
    plt.title(title)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(np.arange(len(losses)), losses)
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_multi_loss_curve(losses: list, labels: list, title: str = 'loss curve', save_path: str = None):
    plt.figure()
    plt.title(title)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    for i in range(len(losses)):
        plt.plot(np.arange(len(losses[i])), losses[i], label=labels[i])
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
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

def plot_log_loss_curve(losses: list, title: str = 'loss curve', save_path: str = None):
    plt.figure()
    plt.title(title)
    plt.xlabel('epochs')
    plt.ylabel('log loss')
    plt.plot(np.arange(len(losses)), np.log(losses))
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

def plot_multi_log_loss_curve(losses: list, labels: list, title: str = 'loss curve', save_path: str = None):
    plt.figure()
    plt.title(title)
    plt.xlabel('epochs')
    plt.ylabel('log loss')
    for i in range(len(losses)):
        plt.plot(np.arange(len(losses[i])), np.log(losses[i]), label=labels[i])
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_acc_curve(accs: list, title: str = 'acc curve', save_path: str = None):
    plt.figure()
    plt.title(title)
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.plot(np.arange(len(accs)), accs)
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_multi_acc_curve(accs: list, labels: list, title: str = 'acc curve', save_path: str = None):
    plt.figure()
    plt.title(title)
    plt.xlabel('epochs')
    plt.ylabel('acc')
    for i in range(len(accs)):
        plt.plot(np.arange(len(accs[i])), accs[i], label=labels[i])
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def get_gpu_mem_info(gpu_id=0):
    import pynvml
    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        print(r'gpu_id {} is not existed!'.format(gpu_id))
        return 0, 0, 0

    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    return total, used
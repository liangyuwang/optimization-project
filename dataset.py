# I want to def a function that can return the mnist dataloder using numpy, and sample the dataset if needed

from pathlib import Path
import torch
import torchvision
import torchvision.transforms as transforms
torch.manual_seed(211)


def data_init(sample_rate=0.001, batch_size=None):
    if sample_rate > 1:
        sample_rate = 1

    # load the dataset
    transform = transforms.Compose([transforms.Resize((28, 28)), 
                                    transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    root_dir = Path(torch.hub.get_dir()) / 'datasets/MNIST'
    train_dataset = torchvision.datasets.MNIST(root=root_dir, train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root=root_dir, train=False, transform=transform, download=True)

    # sample the dataset
    if batch_size is None:
        train_batch_size = int(len(train_dataset)*sample_rate)
        test_batch_size = int(len(test_dataset)*sample_rate)
    else:
        train_batch_size = batch_size
        test_batch_size = batch_size
    train_dataloader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.Subset(
            train_dataset, range(int(len(train_dataset)*sample_rate))), 
            batch_size=train_batch_size, 
        shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.Subset(
            test_dataset, range(int(len(test_dataset)*sample_rate))), 
            batch_size=test_batch_size, 
        shuffle=False)

    return train_dataloader, test_dataloader
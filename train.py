import torch
from tqdm import tqdm

from dataset import data_init
from models import Linear, MLP2, Rosen
from losses import cross_entropy
from optim import SD, LBFGS, BFGS
from utils import get_gpu_mem_info
import csv

def train(
        model: str = 'linear',
        optimizer: str = 'bfgs',
        sample_rate: float = 1,
        batch_size: int = 10000,
        num_epoch: int = 5,
        input_size: int = 100,
        hidden_size: int = 100,
        num_classes: int = 10,
        print_log: bool = True,
        test: bool = True,
        device: str = 'cuda:0',
        gpu_mem_log: bool = True
    ):
    torch.manual_seed(211)
    device_id = int(device[-1])

    # load the dataset
    train_dataloader, test_dataloader = data_init(sample_rate=sample_rate, batch_size=batch_size)

    # init the model
    device = torch.device(device)
    if model == 'linear':
        model = Linear(input_size, num_classes, device)
    elif model == 'mlp2':
        model = MLP2(input_size, hidden_size, num_classes, device)
    elif model == 'rosen':
        model = Rosen(device=device)
    else:
        raise ValueError('model not found!')
    
    # init the optimizer
    loss_fn = cross_entropy()
    if optimizer == 'sd':
        optimizer = SD(model, loss_fn, tol=1e-5)
    elif optimizer == 'bfgs':
        optimizer = BFGS(model, loss_fn, tol=1e-5)
    elif optimizer == 'lbfgs':
        optimizer = LBFGS(model, loss_fn, m=5, tol=1e-5)

    # write a csv file
    with open('gpu{}_memory_log.csv'.format(device_id), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'total', 'used'])

    loss_list = []; acc_list = []
    for epoch in tqdm(range(num_epoch)):
        # train the model
        loss = 0
        total = 0
        for data, labels in train_dataloader:
            data, labels = data.to(device), labels.to(device)
            labels = torch.eye(10).to(device)[labels]
            loss += model.loss(data, labels, loss_fn)
            total += labels.shape[0]
            model = optimizer.step(data, labels, epoch)
        if print_log:
            print('epoch {}, loss: {:.10f}'.format(epoch, loss/total))
        loss_list.append(loss.item()/total)

        if gpu_mem_log:
            with open('gpu{}_memory_log.csv'.format(device_id), 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                total, used = get_gpu_mem_info(device_id)
                writer.writerow([epoch, total, used])

        if test:
            # predict the test data
            correct = 0
            total = 0
            for data, labels in test_dataloader:
                data, labels = data.to(device), labels.to(device)
                labels = torch.eye(10).to(device)[labels]
                y_pred = model.forward(data)
                y_pred = torch.argmax(y_pred, axis=1)
                labels = torch.argmax(labels, axis=1)
                correct += torch.sum(y_pred == labels)
                total += labels.shape[0]
            # print('accuracy: {:.4f}'.format(correct / total))
            acc_list.append(correct.item() / total)

        # early stop
        if optimizer.early_stop:
            break
    
    return loss_list, acc_list


if __name__ == '__main__':
    train()
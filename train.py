import torch
from tqdm import tqdm

from dataset import data_init
from models import Linear, MLP2
from losses import cross_entropy
from optim import SD, LBFGS, BFGS

def train(
        model: str = 'linear',
        optimizer: str = 'sd',
        sample_rate: float = 1,
        batch_size: int = 10000,
        num_epoch: int = 100,
        input_size: int = 100,
        hidden_size: int = 100,
        num_classes: int = 10,
        print_log: bool = True,
        test: bool = True,
        device: str = 'cuda:0'
    ):
    torch.manual_seed(211)

    # load the dataset
    train_dataloader, test_dataloader = data_init(sample_rate=sample_rate, batch_size=batch_size)

    # init the model
    device = torch.device(device)
    if model == 'linear':
        model = Linear(input_size, num_classes, device)
    elif model == 'mlp2':
        model = MLP2(input_size, hidden_size, num_classes, device)
    else:
        raise ValueError('model not found!')
    
    # init the optimizer
    loss_fn = cross_entropy()
    if optimizer == 'sd':
        optimizer = SD(model, loss_fn, gtol=1e-5)
    elif optimizer == 'bfgs':
        optimizer = BFGS(model, loss_fn, gtol=1e-5)
    elif optimizer == 'lbfgs':
        optimizer = LBFGS(model, loss_fn, gtol=1e-5)

    loss_list = []
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
            print('epoch {}, loss: {:.4f}'.format(epoch, loss/total))
        loss_list.append(loss.item()/total)

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
            print('accuracy: {:.4f}'.format(correct / total))

        # early stop
        if optimizer.early_stop:
            break
    
    return loss_list
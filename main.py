import numpy as np
from tqdm import tqdm

from dataset import data_init
from models import MLP2
from losses import cross_entropy
from optim import GD, SD, LBFGS, BFGS, Newton

# some variables
sample_rate = 1
input_size = 28*28
hidden_size = 100
num_classes = 10
num_epoch = 100
batch_size = 256
np.random.seed(211)


def main():
    # load the dataset
    train_dataloader, test_dataloader = data_init(sample_rate=sample_rate, batch_size=batch_size)

    # init the model
    model = MLP2(input_size, hidden_size, num_classes)
    loss_fn = cross_entropy()
    # optimizer = GD(model, loss_fn, lr=1e-5, gtol=1e-5)
    optimizer = SD(model, loss_fn, gtol=1e-5)         # 20 epoch from 14.96 to 12.26
    # optimizer = Newton(model, loss_fn, gtol=1e-5)

    for epoch in tqdm(range(num_epoch)):
        # train the model
        loss = 0
        total = 0
        for data, labels in train_dataloader:
            data, labels = data.cpu().numpy(), labels.cpu().numpy()
            labels = np.eye(10)[labels]
            loss += model.loss(data, labels, loss_fn)
            total += labels.shape[0]
            model = optimizer.step(data, labels, epoch)
        print('epoch {}, loss: {:.4f}'.format(epoch, loss/total))

        # predict the test data
        correct = 0
        total = 0
        for data, labels in test_dataloader:
            data, labels = data.cpu().numpy(), labels.cpu().numpy()
            labels = np.eye(10)[labels]
            y_pred = model.forward(data)
            y_pred = np.argmax(y_pred, axis=1)
            labels = np.argmax(labels, axis=1)
            correct += np.sum(y_pred == labels)
            total += labels.shape[0]
        print('accuracy: {:.2f}'.format(correct / total))

        # early stop
        if optimizer.early_stop:
            break


if __name__ == '__main__':
    main()
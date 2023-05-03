import torch
from tqdm import tqdm
import argparse

from dataset import data_init
from models import Linear, MLP2
from losses import cross_entropy
from optim import SD, LBFGS, BFGS

parse = argparse.ArgumentParser()
parse.add_argument('--model', type=str, default='linear', choices=['linear', 'mlp2'])
parse.add_argument('--optimizer', type=str, default='sd', choices=['sd', 'lbfgs', 'bfgs'])
parse.add_argument('--sample_rate', type=float, default=1)
parse.add_argument('--batch_size', type=int, default=10000)
parse.add_argument('--num_epoch', type=int, default=100)
parse.add_argument('--input_size', type=int, default=100)
parse.add_argument('--hidden_size', type=int, default=100)
parse.add_argument('--num_classes', type=int, default=10)
parse.add_argument('--print_log', type=bool, default=True)
parse.add_argument('--test', type=bool, default=True)
parse.add_argument('--device', type=str, default='cuda:0')
args = parse.parse_args()


def main():
    torch.manual_seed(211)

    # load the dataset
    train_dataloader, test_dataloader = data_init(sample_rate=args.sample_rate, batch_size=args.batch_size)

    # init the model
    device = torch.device(args.device)
    if args.model == 'linear':
        model = Linear(args.input_size, args.num_classes, device)
    elif args.model == 'mlp2':
        model = MLP2(args.input_size, args.hidden_size, args.num_classes, device)
    else:
        raise ValueError('model not found!')
    
    # init the optimizer
    loss_fn = cross_entropy()
    if args.optimizer == 'sd':
        optimizer = SD(model, loss_fn, gtol=1e-5)
    elif args.optimizer == 'bfgs':
        optimizer = BFGS(model, loss_fn, gtol=1e-5)
    elif args.optimizer == 'lbfgs':
        optimizer = LBFGS(model, loss_fn, gtol=1e-5)

    loss_list = []
    for epoch in tqdm(range(args.num_epoch)):
        # train the model
        loss = 0
        total = 0
        for data, labels in train_dataloader:
            data, labels = data.to(device), labels.to(device)
            labels = torch.eye(10).to(device)[labels]
            loss += model.loss(data, labels, loss_fn)
            total += labels.shape[0]
            model = optimizer.step(data, labels, epoch)
        if args.print_log:
            print('epoch {}, loss: {:.4f}'.format(epoch, loss/total))
        loss_list.append(loss/total)

        if args.test:
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


if __name__ == '__main__':
    main()
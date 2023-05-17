import argparse
import pickle

from train import train


parse = argparse.ArgumentParser()
parse.add_argument('--model', type=str, default='linear', choices=['linear', 'mlp2'])
parse.add_argument('--optimizer', type=str, default='lbfgs', choices=['sd', 'lbfgs', 'bfgs'])
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
    
    loss_list, acc_list = train(
        model=args.model,
        optimizer=args.optimizer,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_epoch=args.num_epoch,
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_classes=args.num_classes,
        print_log=args.print_log,
        test=args.test,
        device=args.device
    )
    
    save_name = '{}_{}_{}_{}_{}.pkl'.format(args.model, args.optimizer, args.batch_size, args.num_epoch, args.hidden_size)
    pickle.dump((loss_list, acc_list), open(save_name, 'wb'))


if __name__ == '__main__':
    main()
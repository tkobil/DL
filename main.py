import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Training CIFAR10")
    parser.add_argument('--optimizer', dest="optimizer", default="SGD", type=str, help="optmizer type, defaults to SGD")
    parser.add_argument('--lr', dest="lr", default=0.03, type=float, help="learning rate")
    parser.add_argument('--momentum', dest="momentum", default=0.9, type=float, help="momentum (if applicable)")
    parser.add_argument('--weight-decay', dest="weight_decay", default=5e-4, type=float, help="weight decay (if applicable")
    parser.add_argument('--num-epochs', dest="num_epochs", default=10, type=int, help="number of epochs for test/train loops")
    return parser.parse_args()


def main():
    args = get_args()

    output_file_name = f"run_optimizer={args.optimizer}_lr={args.lr}_momentum={args.momentum}_weightdecay={args.weight_decay}_numepochs={args.num_epochs}"
    
    with open(output_file_name, 'w') as outfile:
        outfile.write("some test results")
    
    
if __name__ == "__main__":
    main()
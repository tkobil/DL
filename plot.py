import argparse
import matplotlib.pyplot as plt
import csv

def get_args():
    parser = argparse.ArgumentParser(description="Training CIFAR10")
    parser.add_argument('--experiment', required=True, dest="experiment", type=str, help="Experiment for test/train data file name")
    return parser.parse_args()



def get_file_names(experiment):
    test_file_name = f'experiments/test_run_{experiment}.csv'
    train_file_name = f'experiments/train_run_{experiment}.csv'
    return train_file_name, test_file_name



def main():
    args = get_args()
    
    train_file_name, test_file_name = get_file_names(args.experiment)
    
    with open(train_file_name) as train_file:
        train_csv = csv.DictReader(train_file, delimiter=",", skipinitialspace=True)
        train_data = [line for line in train_csv]
        
        with open(test_file_name) as test_file:
            test_csv = csv.DictReader(test_file, delimiter=",", skipinitialspace=True)
            test_data = [line for line in test_csv]
            
            train_losses = [float(epoch['train_loss']) for epoch in train_data]
            test_losses = [float(epoch['test_loss']) for epoch in test_data]
            
            train_accs = [float(epoch['train_acc']) for epoch in train_data]
            test_accs = [float(epoch['test_acc']) for epoch in test_data]
            
            epochs = [int(epoch['epoch']) for epoch in train_data]
            
            min_loss = min(train_losses + test_losses)
            max_loss = max(train_losses + test_losses)
            
            min_acc = min(train_accs + test_accs)
            max_acc = max(train_accs + test_accs)
            
            plt.subplot(2, 1, 1)
            plt.suptitle(args.experiment.replace('_', ', '))
            plt.title('Test/Train Loss')
            plt.xlabel('Number of Epochs')
            plt.ylabel('Loss')
            plt.xlim(0, len(train_data))
            plt.ylim(min_loss, max_loss)
            plt.plot(epochs, train_losses, label='train loss')
            plt.plot(epochs, test_losses, label='test loss')
            plt.legend()
            
            plt.subplot(2, 1, 2)
            plt.title('Test/Train Accuracy')
            plt.xlabel('Number of Epochs')
            plt.ylabel('Accuracy')
            plt.xlim(0, len(train_data))
            plt.ylim(min_acc, max_acc)
            plt.plot(epochs, train_accs, label='train accuracy')
            plt.plot(epochs, test_accs, label='test accuracy')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'experiments/plots/{args.experiment}.png', bbox_inches='tight')
            
            
            
            
if __name__ == "__main__":
    main()
            
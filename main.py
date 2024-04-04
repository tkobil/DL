import argparse
import torchvision
from torchvision.transforms import ToTensor
from model import ResNet18
import torch.optim as optim
import torch
import torch.nn as nn


def get_args():
    parser = argparse.ArgumentParser(description="Training CIFAR10")
    parser.add_argument('--optimizer', dest="optimizer", default="SGD", type=str, help="optmizer type, defaults to SGD")
    parser.add_argument('--lr', dest="lr", default=0.03, type=float, help="learning rate")
    parser.add_argument('--momentum', dest="momentum", default=0.9, type=float, help="momentum (if applicable)")
    parser.add_argument('--weight-decay', dest="weight_decay", default=5e-4, type=float, help="weight decay (if applicable")
    parser.add_argument('--num-epochs', dest="num_epochs", default=10, type=int, help="number of epochs for test/train loops")
    parser.add_argument('--scheduler', dest="scheduler", default=None, type=str, help="scheduler type for learning rate")
    parser.add_argument('--transform', dest="transform", default=False, type=bool, help="transform training data")
    parser.add_argument('--save', dest="save", default=False, type=bool, help="save model or not")
    return parser.parse_args()

def get_optimizer(model, optimizer_type, lr, momentum, weight_decay):
    if optimizer_type == "SGD":
        return optim.SGD(model.parameters(), lr=lr,
                      momentum=momentum, weight_decay=weight_decay)
    
    elif optimizer_type == "Adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    

def get_scheduler(optimizer, scheduler_type):
    if scheduler_type == "ExponentialLR":
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    elif scheduler_type == "MultiStepLR":
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60], gamma=0.9)
    
def get_transform(transform):
    if transform:
        return torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4915, 0.4823, .4468), (0.2470, 0.2435, 0.2616)), # mean/std
        ])
    else:
        return torchvision.transforms.ToTensor()

    
     
def train(model, iterator, optimizer, criterion, device, scheduler=None):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(iterator):
        print(f"Training Batch Index: {batch_idx}")
        optimizer.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)    

        outputs = model(inputs)
        
        
        loss = criterion(outputs, targets)
        
        epoch_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
        
        loss.backward()
        optimizer.step()
    if scheduler:
        scheduler.step()
            
    return epoch_loss / len(iterator), correct / total

def test(model, iterator, criterion, device):
    
    # Q3c. Set up the evaluation function.
    epoch_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(iterator):
            print(f"Evaluating Batch Index: {batch_idx}")
            inputs = inputs.to(device)
            targets = targets.to(device)    
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            
        
    return epoch_loss / len(iterator), correct / total
        
    

def main():
    args = get_args()
    
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    
    # Load CIFAR10
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=get_transform(args.transform))
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=get_transform(args.transform))
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False)
    
    model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = get_optimizer(model, args.optimizer, args.lr, args.momentum, args.weight_decay)
    scheduler = get_scheduler(optimizer, args.scheduler)
    
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"NUMBER OF PARAMS: {n_params}")
    if (n_params > 5000000):
        raise Exception(f"More than 5mil parameters!")
    
    train_output_file_name = f"experiments/train_run_optimizer={args.optimizer}_lr={args.lr}_momentum={args.momentum}_weightdecay={args.weight_decay}_numepochs={args.num_epochs}_scheduler={args.scheduler}_transform={args.transform}.csv"
    test_output_file_name = f"experiments/test_run_optimizer={args.optimizer}_lr={args.lr}_momentum={args.momentum}_weightdecay={args.weight_decay}_numepochs={args.num_epochs}_scheduler={args.scheduler}_transform={args.transform}.csv"

    
    with open(train_output_file_name, 'w') as train_outfile:
        
        
        train_fields = "epoch, train_loss, train_acc\n"
        train_outfile.write(train_fields)
        
        with open(test_output_file_name, 'w') as test_outfile:
            test_fields = "epoch, test_loss, test_acc\n"
            test_outfile.write(test_fields)
            
            
    
            for epoch in range(1, args.num_epochs+1):
                print(f"Epoch: {epoch}")
                
                train_loss, train_acc = train(model, trainloader, optimizer, criterion, device, scheduler=scheduler)                
                train_line = f"{epoch}, {train_loss}, {train_acc}\n"
                train_outfile.write(train_line)
                
                test_loss, test_acc = test(model, testloader, criterion, device)
                test_line = f"{epoch}, {test_loss}, {test_acc}\n"
                test_outfile.write(test_line)
                
                print(f"TRAIN ACC: {train_acc}, TEST ACC: {test_acc}")
                
    if (args.save):
        torch.save(model.state_dict(), f"model_optimizer={args.optimizer}_lr={args.lr}_momentum={args.momentum}_weightdecay={args.weight_decay}_numepochs={args.num_epochs}_scheduler={args.scheduler}_transform={args.transform}.pt")
    
if __name__ == "__main__":
    main()
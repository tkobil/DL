# Training CIFAR-10 Image Data with a modified ResNet-18

## Getting started
Create Virtual Environment

    python3 -m venv venv
    . venv/bin/activate

Install Requirements

    pip install -r requirements.txt

## To run an experiment and generate test/train results
    python main.py --optimizer=Adam --num-epochs=30 --lr=0.07

This will generate results in `experiments/` directory. 
Two files will be written:
    
    train_output_file_name = f"experiments/train_run_optimizer={args.optimizer}_lr={args.lr}_momentum={args.momentum}_weightdecay={args.weight_decay}_numepochs={args.num_epochs}_scheduler={args.scheduler}_transform={args.transform}.csv"

    test_output_file_name = f"experiments/test_run_optimizer={args.optimizer}_lr={args.lr}_momentum={args.momentum}_weightdecay={args.weight_decay}_numepochs={args.num_epochs}_scheduler={args.scheduler}_transform={args.transform}.csv"
  


The following experiment options are available via command-line arguments:


| CLI Arg | HyperParameter | Default | Options |
| :-----: | :------------: | :-----: | :-----: |
| --optimizer | optimizer  | SGD     | SGD, Adam |
| -- lr | learning rate | 0.03 | any float |
| --momentum | momentum | 0.9 | any float |
| --weight-decay | weight decay | 5e-4 | any float |
| --num-epochs | number of epochs | 10 | any int |
| --scheduler | scheduler | None | None, ExponentialLR, MultiStepLR |
| --transform | transform | False | True, False |
| --save | save | False | True, False |


Note: `--save` will save a model in `.pt` format here: 

    f"model_optimizer={args.optimizer}_lr={args.lr}_momentum={args.momentum}_weightdecay={args.weight_decay}_numepochs={args.num_epochs}_scheduler={args.scheduler}_transform={args.transform}.pt"


## Plotting Results
    python plot.py --experiment=optimizer=SGD_lr=0.03_momentum=0.9_weightdecay=0.0005_numepochs=30_scheduler=None

This will generate a plot `.png` file under `experiments/plots/` directory based off the
test and train csv data.
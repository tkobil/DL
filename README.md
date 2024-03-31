# Training CIFAR-10 Image Data with a modified ResNet-18

## Getting started
Create Virtual Environment
    python3 -m venv venv
    . venv/bin/activate

Install Requirements
    pip install -r requirements.txt

## To run an experiment and generate test/train results
    python main.py --optimizer=Adam --num-epochs=30 --lr=0.07

This will generate results in `experiments/` directory

## Plotting Results
    python plot.py --experiment=optimizer=SGD_lr=0.03_momentum=0.9_weightdecay=0.0005_numepochs=30_scheduler=None

This will generate a plot `.png` file under `experiments/plots/` directory
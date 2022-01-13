# Efficient Constituency Tree based Encoding for Natural Language to Bash Translation

## Setup

To create environment from tree.yml:

```setup
conda env create --file tree.yml
```

## Training

To train the model in the paper, run:

```
python main.py --mode train
```
The default values of hyper-parameters are set to those mentioned in the paper.
The full hyper-parameter list can be accessed with:
```
python train.py --help
```


## Generate Bash Commands

To generate Bash commands for the invocations in the test set using the trained model, run:

```
python main.py --mode evaluate --checkpoint_path path_to_checkpoint_directory
```

<!-- ## Pre-trained Models

Pretrained models are available at ________. -->
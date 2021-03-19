# Pytorch Lightning MAML Implementation

<p align="center">
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-orange?logo=pytorch"></a>
    <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-blueviolet"></a>
    <a href="https://hydra.cc/"><img alt="Conf: hydra" src="https://img.shields.io/badge/conf-hydra-blue"></a>
    <a href="https://wandb.ai/site"><img alt="Logging: wandb" src="https://img.shields.io/badge/logging-wandb-yellow"></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

This repository is the reimplementation of [MAML](https://arxiv.org/abs/1703.03400) (Model-Agnostic 
Meta-Learning) algorithm. Differentiable
optimizers are handled by [Higher](https://github.com/facebookresearch/higher) library and [NN-template](https://github.com/lucmos/nn-template) is used for 
structuring
the project. The default settings are used for training on Omniglot (5-way
5-shot) problem. It can be easily extended for other few-shot datasets thanks to
[Torchmeta](https://github.com/tristandeleu/pytorch-meta) library.

## Quickstart

**On Local Machine**

1. Download and install dependencies
```bash
git clone https://github.com/rcmalli/lightning-maml.git
cd ./lightning-maml/
pip install -r requirements.txt
```

2. Create `.env` file containing the info given below using your own [Wandb.
   ai](https://wandb.ai) 
   account to track experiments. You can use `.env.template` file.

```bash
export DATASET_PATH="/your/project/root/data/"
export WANDB_ENTITY="USERNAME"
export WANDB_API_KEY="KEY"
```
3. Run the experiment
```bash
python3 src/run.py train.pl_trainer.gpus=1
```

**On Google Colab**
 
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rcmalli/lightning-maml/notebooks/lightning_maml_pub.ipynb)


## Results

TODO

### Omniglot (5-way 5-shot)

## Customization

Inside 'conf' folder, you can change all the settings depending on your 
problem or dataset. The default parameters are set for Omniglot dataset. 
Here are some examples for customization:

### Debug on local machine without GPU

```bash
python3 src/run.py train.pl_trainer.gpus=0 train.pl_trainer.fast_dev_run=true
```

### Running more inner_steps and more epochs

```bash
python3 src/run.py train.pl_trainer.gpus=1  train.pl_trainer.max_epochs=1000 \
data.datamodule.num_inner_steps=5
```

### Running weep of multiple runs

```bash
python3 src/run.py train.pl_trainer.gpus=1 data.datamodule.num_inner_steps=5,10,20 -m
```

### Using different dataset from Torchmeta

If you want to try a different dataset (ex. MiniImageNet), you can copy 
default.yaml file inside `conf/data` to `miniimagenet.yaml` and edit these 
lines :

```yaml
datamodule:
  _target_: pl.datamodule.MetaDataModule

  datasets:
    train:
      _target_: torchmeta.datasets.MiniImagenet
      root: ${env:DATASET_PATH}
      meta_train: True
      download: True

    val:
      _target_: torchmeta.datasets.MiniImagenet
      root: ${env:DATASET_PATH}
      meta_val: True
      download: True

    test:
      _target_: torchmeta.datasets.MiniImagenet
      root: ${env:DATASET_PATH}
      meta_test: True
      download: True

# you may need to update data augmentation and preprocessing steps also!!!
```
Run the experiment as follows:
```bash
python3 src/run.py data=miniimagenet
```

## Notes

There are few required modifications run meta-learning algorithm using
pytorch-lightning as high-level library

1. In supervised learning we have `M` mini-batches for each epoch. However, we
   have `N` tasks for single meta-batch in meta learning settings. We have to
   set our dataloader length to 1 otherwise, the dataloader will indefinitely
   sample from the dataset.

2. Apart from traditional test phase of supervised learning, we need gradient
   computation also in test phase. Currently, pytorch-lightning does not allow
   you to enable gradient computation by settings, you have to add single line
   to your beginning of test and validation steps as following:
   ```python
    torch.set_grad_enabled(True)
   ```
3. In MAML algorithm, we have two different optimizers to train our model. Inner
   optimizer must be differentiable and outer optimizer should update model
   using updated weights inside inner iteration from support set and updates
   from query set. In Pytorch-lightning optimizer are handled and weight updates
   are done automatically. To disable this behaviour, we have to
   set `automatic_optimization=False` and add following lines to handle backward
   computations manually:
   ```python
   self.manual_backward(outer_loss, outer_optimizer)
   outer_optimizer.step()
   ```


## References

- [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400)
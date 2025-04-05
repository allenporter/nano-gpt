# Work Log

This is the work log for preparing for a legit training run.

## Pre-requisites

```
sudo snap install astral-uv --classic
uv venv --python=3.11
# Note: This should probably hold the systems current pytorch
uv pip install -r requirements_dev.txt
source .venv/bin/activate
```

## Prepare

```bash
$ DATASET=tinyshakespeare
$ nano-gpt prepare_dataset --dataset=${DATASET} --splits=train,validation
```

## Train

```bash
torchrun --standalone --nproc_per_node=4 `which nano-gpt` train --dataset=${DATASET}
```

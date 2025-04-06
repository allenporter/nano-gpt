# Work Log

This is the work log for preparing for a legit training run.

## Test Run

### Prepare

```bash
$ DATASET=tinyshakespeare
$ nano-gpt prepare_dataset --dataset=${DATASET} --splits=train,validation
```

### Train

```bash
torchrun --standalone --nproc_per_node=4 `which nano-gpt` train --dataset=${DATASET}
```

## Real Training Run

### Prepare

This will first download the 13 shared files of the dataset, which are each about 2.15GB,
totaling around 27GB of disk for the entire dataset. It will then tokenize the
dataset to prepare to feed it into training.

Next the script will start tokenizing the dataset into shard files with
100 million tokens per shard. This is about 100 files for 10B total tokens. Each
shard file is about 100MB, so the total token cache is about 10GB on disk. We
don't load the entire dataset into RAM, but re-read each shard as we iterate
through the training dataset.

```bash
$ DATASET=finewebedu
$ nano-gpt prepare_dataset --dataset=${DATASET} --splits=train,validation
```

The tokenization step takes about 15 seconds per shard on a lambda labs beefy
machine, so about 25 minutes in total.

### Train

Checkpoints will be saved in `checkpoints/` by default, every 5k steps.


```bash
torchrun --standalone --nproc_per_node=8 `which nano-gpt` train --dataset=${DATASET}
```

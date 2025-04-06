# Work Log

This is the work log for preparing for a legit training run.

## Prepare

```bash
$ DATASET=tinyshakespeare
$ nano-gpt prepare_dataset --dataset=${DATASET} --splits=train,validation
```

## Train

```bash
torchrun --standalone --nproc_per_node=4 `which nano-gpt` train --dataset=${DATASET}
```

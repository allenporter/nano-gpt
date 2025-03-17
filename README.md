# nano-gpt

This repo is an implementation of Andrej Karpathy's [GPT tutorial series](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ), packaged as a Python library. The core architecture and training approach follows Karpathy's excellent educational content, with some additional packaging and infrastructure work to make it more maintainable and reusable.

## Goals

This project takes Karpathy's tutorial code and adds some additional infrastructure to make it more maintainable and testable:

- Package the implementation as a proper Python library with CLI tools
- Add type hints and documentation for better code clarity
- Include unit tests to ensure reliability
- Support various hardware configurations (MPS, older GPUs, Colab, etc)
- Implement efficient data loading and preprocessing for larger datasets
- Maintain the educational value while making it a little easier to use.

The core GPT-2 implementation and training methodology remains true to Karpathy's original work.

## Architecture

This project implements a GPT-2 style transformer model, following Karpathy's tutorial. This
section contains a high level overview of the key components and any additions.

### Model Architecture
- Implements the standard GPT-2 architecture with transformer blocks containing:
  - Multi-head causal self-attention
  - Layer normalization
  - MLP blocks with GELU activation
- Configurable model sizes matching OpenAI's GPT-2 variants (from S 124M to XL 1.5B) and even smaller XSS variants for testing.
- Shared input/output embeddings for parameter efficiency
- Support for loading pretrained GPT-2 weights from HuggingFace

### Training Pipeline
- Efficient data loading with preprocessing and sharding:
  - Pre-tokenizes datasets using tiktoken (GPT-2 tokenizer)
  - Shards large datasets into manageable chunks
  - Supports streaming for large datasets
  - Implements gradient accumulation for effective larger batch sizes
- Learning rate scheduling with warmup
- Gradient clipping for stable training
- Support for different compute devices (CUDA, MPS, CPU)
- Model compilation for improved performance where available

### Datasets
- Built-in support for:
  - TinyShakespeare (for testing and quick experiments)
  - FineWebEdu (10B token educational dataset)
  - HellaSwag (for model evaluation)
- Extensible dataset loading system using HuggingFace datasets

### Evaluation & Inference
- Text generation with configurable parameters
- Model evaluation on benchmark datasets
- Support for different sampling strategies


## Environment Setup

Install pre-requisites

```bash
$ uv venv --python3.13
$ source .venv/bin/activate
$ uv pip install -r requirements_dev.txt
```

When using a jetson orin with the pytorch container `dustynv/pytorch:2.1-r36.2.0`
you can setup with these commands:

```bash
$ apt install python3.10-venv
$ python3.10 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements_dev.txt
$ pip install "numpy<2"
$ pip install /opt/torch-2.1.0-cp310-cp310-linux_aarch64.whl
```
That will take about 8 days to train, by the way.


Verify that you have the accelerator you expect:
```
$ python3
>>> import torch
>>> torch.cuda.is_available()
True
```

## Sample

This example will download the pretrained gpt2 and sample from it with the given prefix:

```bash
$ nano-gpt sample --pretrained=gpt2 --seed=42 --max-length=30 "Hello, I'm a language model,"
> Hello, I'm a language model, which means I'm familiar with it, but I'm not fluent in that. Well, with that said,
> Hello, I'm a language model, and the syntax, to make use of it, is pretty good. So why do you have that and not
> Hello, I'm a language model, I'm doing this work in Python, and then I'm writing code for Haskell.

So we can
> Hello, I'm a language model, and you're making assumptions about my use of them. I'm not a natural language learner. I'm
> Hello, I'm a language model, well, I'm from Java and have to write a programming language for it. I have my own vocabulary because
```

## Eval

This example will evaluate hellaswag against the pretrained gpt2:

```bash
$ nano-gpt eval --pretrained=gpt2
Accuracy: 0/1 = 0.0000
Accuracy: 0/2 = 0.0000
Accuracy: 1/3 = 0.3333
Accuracy: 1/4 = 0.2500
Accuracy: 1/5 = 0.2000
Accuracy: 1/6 = 0.1667
Accuracy: 1/7 = 0.1429
Accuracy: 1/8 = 0.1250
Accuracy: 1/9 = 0.1111
Accuracy: 1/10 = 0.1000
Accuracy: 2/11 = 0.1818
```

## Prepare training dataset

This will download the huggingface dataset into your `~/.cache` directory.

```bash
$ nano-gpt prepare_dataset --dataset=finewebedu --splits=train,validation
```

## Train

This will train a new gpt2 125M parameter model using 0.5M step sizes
(w/ gradient accumulation if needed) for 10B tokens.

```bash
nano-gpt train --dataset=finewebedu --device=cuda --sequence-length=1024 --micro-batch-size=16 
```

## Additional details

This project is managed with [scruft](https://github.com/allenporter/scruft)


## Work Plan

Additional features to add:
- [ ] hellaswag every N runs of training
- [ ] support checkpointing model
- [ ] log performance stats to disk
- [ ] ability to reset validation or not cycle
- [ ] validation set in eval command
- [ ] validation set in train command
- [ ] plot baseline of gpt2
- [ ] plot results from performance log

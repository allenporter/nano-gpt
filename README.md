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

## Released Model

My model trained with this library is available on huggingface at [allenporter/gp2](https://huggingface.co/allenporter/gpt2)

## Environment Setup

It is recommended to use virtual environments for installing python packages
such as `uv`. Here is an example for how to install uv and create an environment:

```bash
$ sudo snap install astral-uv --classic
$ uv venv --python3.13
$ source .venv/bin/activate
```

The following sections give examples for how to setup the package either for
using the command line or for local development.

### From pypi

This will install from the [nano-gpt](https://pypi.org/project/nano-gpt/)
project from pypi:

```bash
$ uv pip install nano-gpt
```

### Local development

These are the steps to use for local development. You'll first need to
clone the [git repo](https://github.com/allenporter/nano-gpt/) then you can
install the local project:

```bash
$ uv pip install -r requirements_dev.txt
```

### Special Envirments

You may come across machines that already have pytorch installed, e.g. such
as from a GPU hosting provider. You can create a virtual environment like this
to preserve the system packages:

```bash
$ python3 -m venv venv --system-site-packages
$ source venv/bin/activate
$ pip install -r requirements_dev.txt
```

Another example is when using a jetson orin with the pytorch
container `dustynv/pytorch:2.1-r36.2.0` you can setup with these commands:

```bash
$ apt install python3.10-venv
$ python3.10 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements_dev.txt
$ pip install "numpy<2"
$ pip install /opt/torch-2.1.0-cp310-cp310-linux_aarch64.whl
```

That will take about 8 days to train, by the way. I'd only use this for
inference or for local testing.

### Verify setup

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
$ nano-gpt --debug eval --pretrained=gpt2 --validation-steps=0
DEBUG:nano_gpt.hellaswag_eval:hellaswag: hellaswag: accuracy: 1.0000 | total: 1 | correct: 1
DEBUG:nano_gpt.hellaswag_eval:hellaswag: hellaswag: accuracy: 0.5000 | total: 2 | correct: 1
DEBUG:nano_gpt.hellaswag_eval:hellaswag: hellaswag: accuracy: 0.6667 | total: 3 | correct: 2
DEBUG:nano_gpt.hellaswag_eval:hellaswag: hellaswag: accuracy: 0.5000 | total: 4 | correct: 2
DEBUG:nano_gpt.hellaswag_eval:hellaswag: hellaswag: accuracy: 0.4000 | total: 5 | correct: 2
DEBUG:nano_gpt.hellaswag_eval:hellaswag: hellaswag: accuracy: 0.3333 | total: 6 | correct: 2
DEBUG:nano_gpt.hellaswag_eval:hellaswag: hellaswag: accuracy: 0.2857 | total: 7 | correct: 2
DEBUG:nano_gpt.hellaswag_eval:hellaswag: hellaswag: accuracy: 0.2500 | total: 8 | correct: 2
DEBUG:nano_gpt.hellaswag_eval:hellaswag: hellaswag: accuracy: 0.2222 | total: 9 | correct: 2
```

## Prepare training dataset

This will download the huggingface dataset into your `~/.cache` directory
which are 13 shard files about 2.15GB, totaling around 27GB of disk for the
entire dataset.

It will then tokenize the dataset to prepare to feed it into training and
store in `./dataset_cache`. This will create shard files with 100 million tokens
per shard. This is about 100 files for 10B total tokens. Each shard file is
about 100MB, so the total token cache is about 10GB on disk. We
don't load the entire dataset into RAM, but re-read each shard in each worker
as we iterate through the training dataset.

```bash
$ DATASET=finewebedu
$ nano-gpt prepare_dataset --dataset=${DATASET} --splits=train,validation
```

The tokenization step takes about 15 seconds per shard on a lambda labs beefy
machine, so about 25 minutes in total. The process currently appears to be I/O
bound reading/writing the shard, so possible future improvement there.


## Train

Checkpoints will be saved in `checkpoints/` by default, every 5k steps.

This will train a new gpt2 125M parameter model using 0.5M step sizes
(w/ gradient accumulation if needed) for 10B tokens.

```bash
nano-gpt train --dataset=finewebedu --device=cuda --sequence-length=1024 --micro-batch-size=16
```

This is the recipe used for a real training run with increased micro batch size
that can fit on our beefy machine across 8 GPUs:

```bash
$ torchrun --standalone --nproc_per_node=8 `which nano-gpt` train --dataset=${DATASET} --micro-batch-size=32 --hellaswag_samples=250
```

Run appears to take 390ms per step.
10B tokens / 500k tokens per step = 20k steps
20k * 390ms = 2.16 hours

## Training Progress

You can view the results of the training run using the notebook
`script/logs_analysis.ipynb` to examine the training log results.

![Screenshot Training Log](artifacts/train-log.png)

Note that in this training run, hellaswag was not evaluated on the entire dataset
and so the results are higher than expected. I recommend adjusting the hellaswag
evaluation steps to evaluate the entire dataset for future runs.

## Sampling from a checkpoint

This is an example of sampling from the trained model checkpoint after 10B tokens.

```bash
$ nano-gpt sample --checkpoint=./checkpoint_019072.bin --device=mps
> Hello, I'm a language model, you're doing your application, I've put your main program and you want to model. Here are some things
> Hello, I'm a language model, so let's have a look at a few very old and popular dialects with some basic information about some of
> Hello, I'm a language model, but I also use a number of core vocabulary from the Python language and some data structures from
the web to
> Hello, I'm a language model, so this is about building a language to help my students to express themselves in all possible situations when they are in
> Hello, I'm a language model, who wrote my first 'hello' and never used it, but my first 'hello' can't be in
```

## Export

You can export a safetensors model from the pytorch checkpoint. This will create
a `model.safetensors` file and `config.json` in the export directory:

```bash
$ nano-gpt export --checkpoint ./checkpoint_019072.bin --export-dir export --device=cpu
$ jq '.n_ctx' export/config.json
1024
```

This export is written in the same format as the OpenAI GPT2 export and can be
used with the `--pretrained` command line flag:

```bash
$ nano-gpt sample --pretrained=./export
> Hello, I'm a language model, you're doing your application, I've put your main program and you want to model. Here are some things
> Hello, I'm a language model, so let's have a look at a few very old and popular dialects with some basic information about some of
> Hello, I'm a language model, but I also use a number of core vocabulary from the Python language and some data structures from
the web to
> Hello, I'm a language model, so this is about building a language to help my students to express themselves in all possible situations when they are in
> Hello, I'm a language model, who wrote my first 'hello' and never used it, but my first 'hello' can't be in
```

## Upload

You can upload your exported model to huggingface and use it like any other
pretrained model.

```bash
$ huggingface-cli login
$ HUGGINGFACE_USER=`huggingface-cli whoami`
$ echo $HUGGINGFACE_USER
allenporter
$ huggingface-cli upload ${HUGGINGFACE_USER}/gpt2 export/ .
```

This will allow you to load the model from the huggingface repo:

```bash
$ nano-gpt sample --pretrained=allenporter/gpt2
> Hello, I'm a language model, you're doing your application, I've put your main program and you want to model. Here are some things
> Hello, I'm a language model, so let's have a look at a few very old and popular dialects with some basic information about some of
> Hello, I'm a language model, but I also use a number of core vocabulary from the Python language and some data structures from
the web to
> Hello, I'm a language model, so this is about building a language to help my students to express themselves in all possible situations when they are in
> Hello, I'm a language model, who wrote my first 'hello' and never used it, but my first 'hello' can't be in
```

## Additional details

This project is managed with [scruft](https://github.com/allenporter/scruft)

---
language:
- en
license: mit
library_name: pytorch
tags:
- nano-gpt
- pytorch
- safetensors
datasets:
- HuggingFaceFW/fineweb-edu
metrics:
- accuracy
---
# Model Card for gpt2

<!-- Provide a quick summary of what the model is/does. -->

This model is a reproduction of gpt2 following Andrej Karpathy's GPT [tutorial series](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
and the original [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

The model was trained using the [nano-gpt](https://github.com/allenporter/nano-gpt/) library
which follows the pattern from Karpathy's excellent content, with some additional
packaging and infrastructure work to make it more maintainable and reusable.

## Model Details

### Model Description

GPT-2 is a transformers model pretrained on a large corpus of english only text
with no labeling.  This is the smallest version of GPT-2, with 124M parameters.

The model follows the standard GPT-2 architecture with transformer blocks containing:
  - Multi-head causal self-attention
  - Layer normalization
  - MLP blocks with GELU activation

The model was trained using a sample of the FineWeb-EDU using 10B tokens. The
dataset contains educational web pages.

- **Developed by:** Allen Porter

### Model Sources

<!-- Provide the basic links for the model. -->

- **Repository:** https://github.com/allenporter/nano-gpt/

## How to Get Started with the Model

This model is stored in safetensors format and is in the same format used by
the gpt2 model released by OpenAI.

The easiest way to load this model is to use the `nano-gpt` command line
tool. You can install the package from pypi. Here is an example using
a virtual enviromnemt with `uv`:

```bash
$ uv venv --python=3.13
$ source .venv/bin/activate
$ uv pip install nano-gpt
```

You may then load this pretrained model:

```bash
$ nano-gpt sample --pretrained=allenporter/gpt2
> Hello, I'm a language model, you're doing your application, I've put your main program and you want to model. Here are some things
> Hello, I'm a language model, so let's have a look at a few very old and popular dialects with some basic information about some of
> Hello, I'm a language model, but I also use a number of core vocabulary from the Python language and some data structures from
the web to
> Hello, I'm a language model, so this is about building a language to help my students to express themselves in all possible situations when they are in
> Hello, I'm a language model, who wrote my first 'hello' and never used it, but my first 'hello' can't be in
```

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

This model was trained from https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
from the 10B token sample.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing

The model was pre-processed and sharded to make data loading efficient.  The
dataset was pre-tokenized using GPT-2 tokenizer using the `nano-gpt prepare_dataset`
command. The file is split into manageable chunks.

#### Training Hyperparameters

- **Training regime:** See `train_config` in `config.json` for the hyper parameters.

The main features of the training process are:
- Learning rate scheduling with warmup
- Gradient clipping for stable training
- Model compilation for improved performance where available

#### Speeds, Sizes, Times

The model was trained using 8 x A100s. The model was run for one full epoch of
the 10B token dataset, which is 19072 steps. The model was trained for about 2
hours.

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

The model was evaluated using hellaswag dataset. TBD results.

The `nano-gpt train` command has built in support for evaluating against
the val dataset as well as HellaSwag in between training steps. Every 500 steps
the model was evaluated against the val dataset and HellaSwag.

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** A100
- **Hours used:** 2 hours
- **Cloud Provider:** Lambda Labs
- **Compute Region:** Arizona
- **Power estimate:** 8 GPUs * 0.325 kW/GPU = 2.6 kW.  2.6 kW * 2 hours = 5.2 kWh
- **CO2 Estimate:** 2.6 pounds of CO2 equivalent assuming 500 lbs CO2e per MWh

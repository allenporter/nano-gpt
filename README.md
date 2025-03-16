# nano-gpt

This repo is for following along in Andrej Karpathy's [series](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) on ML for the section on gpt-2. This is
not meant to be unique for the community, more just for capturing my own
work environment.

## Goals

Some specific goals for this project:

- Pull out into a separate project from https://github.com/allenporter/karpathy-guides which is my colab environmnt for the other tutorials. The code is starting to get bigger than a single colab.
- Add additional documentation that is useful for my own reference
- Make routines for working on hardware I have available (mps, old GPUs, colab, etc)
- Additional type checking for safety/documentation
- Basic unit tests to ensure things continue working
- Add a little more abstraction, as a style preference, to facilitate typing and testing
- Allow experimenting on other python frameworks other than pytorch

This project is managed with [scruft](https://github.com/allenporter/scruft)

## Environment

Install pre-requisites

```bash
$ uv venv --python3.13
$ source .venv/bin/activate
$ uv pip install -r requirements_dev.txt
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

## Work Plan

Status: Everything works, however the GPU is not fully saturated given the I/O
  is not isolated out of the training loop.

- [x] Update `prepare_dataset` to pre-tokenize the dataset
- [x] Determine output storage location
- [x] Add support for training from pre-tokenized files
- [x] Add output write sharding
- [x] Add input read sharding
- [ ] Tokenize files in prepare dataset
- [ ] Update pre-tokenized `train` vs `validation` splits
- [ ] Loading of pre-tokenized datasets by slit

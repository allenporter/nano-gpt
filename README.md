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

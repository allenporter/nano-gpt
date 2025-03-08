# nano-gpt

This repo is for following along in Andrej Karpathy's [series](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) on ML for the section on gpt-2. This is
not meant to be unique for the community, more just for capturing my own
work environment.

Some specific goals for this project:

- Pull out colab code into reusable libraries
- Allow working on hardware I have available (mps, old GPUs, colab, etc)
- Basic python tests/type checking
- Allow experimenting on other python frameworks other than pytorch

## Environment

```bash
$ uv venv --python3.13
$ source .venv/bin/activate
$ uv pip install -r requirements.txt
```

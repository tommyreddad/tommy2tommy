# tommy2tommy

A small personal playground for deep learning. More to follow.

This repository is highly inspired by the excellent [https://github.com/tensorflow/tensor2tensor](tensor2tensor).

## Prerequisites

This requires the following dependencies:

- [`Python 3`](https://www.python.org/downloads/)
- [`TensorFlow 2`](https://www.tensorflow.org/install)
- [`NumPy`](https://numpy.org/install/)
- For testing: [`pytest`](https://docs.pytest.org/en/stable/getting-started.html#)

The latest supported version of Python is 3.8.

## Installation

One simply way to set up this packagage is to leverage the requirements file.

First, create a Python 3.8 virtual environment:

```shell
virtualenv -p python38 env
source ./env/bin/activate
```

Then, install the requirements:

```shell
pip install -r requirements.txt
```

## Testing

To run all unit tests, execute the following command from the top-level directory:

```shell
pytest --disable-warnings tommy2tommy/
```

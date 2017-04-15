# DLFractalSequences

TensorFlow implementation of a novel open-source Seq2SeqRegression API for performing a wide range of automatic feature extraction tasks outside of NLP. This general purpose Sequence-to-Sequence Regression model can predict a sequence of multidimensional vectors based on previous observations. The system of study being analyzed here is the Plouffe Graph, a graph by Canadian mathematician Simon Plouffe in 1974-1979. More information about the Plouffe Graph can be found here: [Times Tables, Mandelbrot and the Heart of Mathematics](https://www.youtube.com/watch?v=qhbuKbxJsk8).

## Table of Contents

* [Dataset](#dataset)
* [Installation](#installation)
* [Anaconda Installation](#anaconda)
* [Docker Installation](#docker)
* [Sharcnet: Train Multiple Jobs](#sharcnet)

* * *

### Dataset

The Plouffe dataset is already included. A dataset of multidimensional vectors that represent the Plouffe Graph gets constructed during training. The dataset can be configured easily in the `plouffe.yml` file inside the `configs` folder.

### Installation

The program requires the following dependencies (easy to install using pip, Anaconda or Docker):

* python 2.7
* tensorflow API (tested with r1.0.0)
* numpy
* scipy
* pandas
* matplotlib
* jupyter
* networkx
* tqdm
* pyyaml
* jupyterthemes
* seaborn

### Anaconda Installation

To install DLFractalSequences in an Anaconda environment:

```python
conda env create -f environment.yml
```

To activate Anaconda environment:

```python
source activate dlfractals-env
```

Train Seq2Seq Regression model on the local machine using the Plouffe dataset:

```python
python train.py -c configs/plouffe.yml
```

Note: The training inputs (i.e. dataset parameters, hyperparameters etc.) for training on a `local` machine can be modified in the `plouffe.yml` inside the `configs` folder.

### Docker Installation

**Prerequisites: Docker installed on your machine. If you don't have docker installed already, then go here to [Docker Setup](https://docs.docker.com/engine/getstarted/step_one/)**

To build Docker image:

```python
docker build -t dlfractals:latest .
```
To deploy and train on Docker container:
```python
docker run -it dlfractals:latest python train.py -c configs/plouffe.yml
```

### Sharcnet: Train Multiple Jobs

Activate Tensorflow Python2.7 environment:

```python
source /opt/sharcnet/testing/tensorflow/tensorflow-cp27-active
```

Note: If there is anything missing, then do:

```sh
pip install <missing_pkg> --user
```

Example: 

```sh
pip install /opt/sharcnet/testing/tensorflow/tensorflow-1.0.0-cp27-cp27m-linux_x86_64.whl --user
```

Train multiple jobs using the Seq2Seq Regression model on the Plouffe dataset:

```python
python train_manyjobs.py -c configs/plouffe_sharcnet.yml
```

Note: The training inputs (i.e. dataset parameters, hyperparameters etc.) for training on a `sharcnet` machine can be modified in the `plouffe.yml` inside the `configs` folder. You must specify `train` option inside the YAML config file to be either `copper` or `local` when training on sharcnet.

* * * 
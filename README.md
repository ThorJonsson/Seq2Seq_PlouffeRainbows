# DLFractalSequences

## Anaconda Installation
#### To install DLFractalSequences in an Anaconda environment:

```sh
conda env create -f environmental.yml
```

#### To activate Anaconda environment

```sh
source activate dlfractals-env
```

#### Train Seq2Seq Regression model on Plouffe dataset

```sh
python train.py -c configs/plouffe.yml
```

### To install 
## Docker Installation
#### To build Docker image:

```sh
docker build -t dlfractals:latest .
```
#### To deploy and train on Docker container:
```sh
docker run -it dlfractals:latest python train.py
```

## Train Multiple Jobs on Sharcnet (Copper cluster)
#### Activate Tensorflow Python2.7 environment

```sh
source /opt/sharcnet/testing/tensorflow/tensorflow-cp27-active
```

##### Note: If there is anything missing, then do:

```sh
pip install <missing_pkg> --user
i.e. pip install /opt/sharcnet/testing/tensorflow/tensorflow-1.0.0-cp27-cp27m-linux_x86_64.whl --user
```

#### Train multiple jobs using the Seq2Seq Regression model on the Plouffe dataset

```sh
python train_manyjobs.py -c configs/plouffe_sharcnet.yml
```
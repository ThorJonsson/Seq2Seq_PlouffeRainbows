# DLFractalSequences

## Anaconda Installation
### To install DLFractalSequences in an Anaconda environment:

```sh
conda env create -f environmental.yml
```

### To activate Anaconda environment

```sh
source activate dlfractals-env
```

### Train Seq2Seq Regression model on Plouffe dataset

```sh
python train.py -c configs/plouffe.yml
```

### To install 
## Docker Installation
### To build Docker image:

```sh
docker build -t dlfractals:latest .
```
### To deploy and train on Docker container:
```sh
docker run -it dlfractals:latest python train.py
```

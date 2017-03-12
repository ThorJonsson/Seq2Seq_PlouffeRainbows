FROM tensorflow/tensorflow
MAINTAINER vithursant

RUN \
	apt-get -qq -y update && apt-get -y install && \
	apt-get -y install ipython ipython-notebook

RUN \
	pip install -U numpy \
  	pandas \
  	networkx \
  	matplotlib \
  	scipy \
  	jupyter \
  	pdb \
  	tqdm

 COPY ./ /root/DLFractalSequences

 WORKDIR /root/DLFractalSequences/fibonacci/trainer

 CMD /bin/bash

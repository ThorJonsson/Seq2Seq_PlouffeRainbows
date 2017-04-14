FROM tensorflow/tensorflow
MAINTAINER vithursant

RUN \
	apt-get -qq -y update && apt-get -y install && \
	apt-get -y install ipython ipython-notebook python-tk

RUN \
	pip install -U numpy \
  	pandas \
  	networkx \
  	matplotlib \
  	scipy \
  	jupyter \
  	pdb \
  	tqdm \
        pyyaml

 COPY ./ /root/DLFractalSequences

 WORKDIR /root/DLFractalSequences

 CMD /bin/bash

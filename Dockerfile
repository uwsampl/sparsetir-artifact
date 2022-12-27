FROM nvidia/cuda:11.6.1-cudnn8-runtime-ubuntu20.04 as base

ENV DEBIAN_FRONTEND=noninteractive

# Base scripts
RUN apt clean
RUN apt update --fix-missing

# Ubuntu install core
# install libraries for building c++ core on ubuntu
COPY install/ubuntu_install_core.sh /install/ubuntu_install_core.sh
RUN bash /install/ubuntu_install_core.sh

# install miniconda
COPY install/install_conda.sh /install/install_conda.sh
RUN bash /install/install_conda.sh

# install pytorch

# install dgl and pytorch geometric

# compile & install dgSPARSE

# compile & install Sputnik

# compile & install taco

# compile & install torchsparse

# compile & install triton

# compile & install graphiler
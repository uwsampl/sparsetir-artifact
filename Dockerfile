FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04 as base

ENV DEBIAN_FRONTEND=noninteractive

# Base scripts
RUN apt clean
RUN apt update --fix-missing

# Ubuntu install core
# install libraries for building c++ core on ubuntu
COPY install/ubuntu_install_core.sh /install/ubuntu_install_core.sh
RUN bash /install/ubuntu_install_core.sh

# install python
COPY install/install_python.sh /install/install_python.sh
RUN bash /install/install_python.sh

# install python packages
COPY install/ubuntu_install_python_package.sh /install/ubuntu_install_python_package.sh
RUN bash /install/ubuntu_install_python_package.sh

# install pytorch
COPY install/install_pytorch.sh /install/install_pytorch.sh
RUN bash /install/install_pytorch.sh

# install dgl and pytorch geometric
COPY install/install_dgl.sh /install/install_dgl.sh
RUN bash /install/install_dgl.sh
COPY install/install_pyg.sh /install/install_pyg.sh
RUN bash /install/install_pyg.sh

# install llvm
COPY install/ubuntu2004_install_llvm.sh /install/ubuntu2004_install_llvm.sh
RUN bash /install/ubuntu2004_install_llvm.sh

# install SparseTIR
COPY 3rdparty/SparseTIR /tmp/SparseTIR
WORKDIR /tmp/SparseTIR
RUN bash docker/install/install_sparsetir_gpu.sh

# compile & install glog
COPY 3rdparty/glog /tmp/glog/
WORKDIR /tmp/glog
RUN mkdir build
RUN cd build/\
    && cmake ..\
    && make -j\
    && make install

# compile & install dgSPARSE

# compile & install Sputnik

# compile & install taco

# compile & install torchsparse

# compile & install triton

# compile & install graphiler
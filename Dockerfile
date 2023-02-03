FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04 as base

ENV DEBIAN_FRONTEND=noninteractive

# Base scripts
RUN apt clean
RUN apt update --fix-missing

# Ubuntu install core
# install libraries for building c++ core on ubuntu
COPY install/ubuntu_install_core.sh /install/ubuntu_install_core.sh
RUN bash /install/ubuntu_install_core.sh

# install cmake 3.24
COPY install/install_cmake_source.sh /install/install_cmake_source.sh
RUN bash /install/install_cmake_source.sh

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
RUN rm -rf build\
    && mkdir build\
    && cd build/\
    && cmake ..\
    && make -j\
    && make install

# compile & install torchsparse
RUN apt install libsparsehash-dev
COPY 3rdparty/torchsparse /tmp/torchsparse
WORKDIR /tmp/torchsparse
RUN pip3 install -e .

# compile & install kineto
COPY 3rdparty/kineto /tmp/kineto
WORKDIR /tmp/kineto
RUN cd libkineto\
    && rm -rf build/ && mkdir build\
    && cd build\
    && cmake .. && make -j\
    && make install

# compile & install graphiler
COPY 3rdparty/graphiler /tmp/graphiler
WORKDIR /tmp/graphiler
RUN rm -rf build/\
    && mkdir build\
    && cd build/\
    && cmake -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..\
    && make\
    && mkdir -p ~/.dgl\
    && mv libgraphiler.so ~/.dgl/
RUN pip3 install -e .

# compile & install Sputnik
COPY 3rdparty/sputnik /tmp/sputnik
WORKDIR /tmp/sputnik
RUN rm -rf build\
    && mkdir build\
    && cd build/\
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TEST=ON -DBUILD_BENCHMARK=ON -DCUDA_ARCHS="70;75;80;86"\
    && make -j\
    && make install

# compile & install dgSPARSE
COPY 3rdparty/dgsparse /tmp/dgsparse
WORKDIR /tmp/dgsparse
RUN rm -rf build\
    && mkdir build\
    && cd build/\
    && cmake ..\
    && make -j\
    && make install

# compile & install triton
COPY 3rdparty/triton /tmp/triton
WORKDIR /tmp/triton
RUN cd python\
    && pip3 install -e .

# compile & install taco
COPY 3rdparty/taco /tmp/taco
WORKDIR /tmp/taco
RUN rm -rf build/\
    && mkdir build\
    && cd build/\
    && cmake ..\
    && make

# install sparsetir_profiler
COPY python/ /tmp/sparsetir_profiler
WORKDIR /tmp/sparsetir_profiler
RUN pip3 install -e .

# download data
WORKDIR /root
COPY sparse-conv/download_data.sh download_sparse_conv.sh
COPY spmm/download_data.py download_gnn_data.py
COPY rgcn/download_data.py download_rgcn_data.py
COPY prunned-bert/download_model.py download_huggingface_model.py

RUN bash download_sparse_conv.sh
RUN echo "y\ny\n" | python3 download_gnn_data.py
RUN echo "y\n" | python3 download_rgcn_data.py
RUN python3 download_huggingface_model.py

# install plotting softwares
RUN apt-get install -y texlive-font-utils gnuplot

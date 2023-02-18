# SparseTIR Artifact

[![DOI](https://zenodo.org/badge/471216066.svg)](https://zenodo.org/badge/latestdoi/471216066)

This repository contains scripts for setting up environments and reproducing results presented in the ASPLOS 2023 paper entitled [SparseTIR: Composable Abstractions for Deep Learning](https://dl.acm.org/doi/10.1145/3582016.3582047). To access the core implementation of SparseTIR (i.e., the compiler), please visit the [SparseTIR repository](https://github.com/uwsampl/sparsetir). Additionally, we have written a post called [Retrospective on SparseTIR Artifact](https://gist.github.com/yzh119/c4425ee0375a98a36a8786f6ba9d4ee8) in which we discuss some issues we encountered when preparing this artifact. Please give it a read if you're interested.

## Prerequisite

We require NVIDIA Container Toolkit to setup environments, please follow instructions from [installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html),
below is the installation script for Debian/Ubuntu (extracted from official guide):
```
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker

distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

User can try the following command to test whether the installation was successful or not:
```
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
```

## Clone the Repository

```bash
git clone https://github.com/uwsampl/sparsetir-artifact.git --recursive
cd sparsetir-artifact
```
## Setup Docker Image

### Pull from Docker Hub

We provide a pre-built docker image available on Docker Hub which is compatible with Ampere architecture NVIDIA GPUs, user can pull it with:
```
docker image pull expye/sparsetir-ae:latest
docker tag expye/sparsetir-ae:latest sparsetir
```

### Build from source
Otherwise, user need to build the docker image from source code. Before building the artifact, user need to set docker's default runtime to `nvidia` to enable GPU access when buildling docker images ([reference](https://github.com/NVIDIA/nvidia-docker/wiki/Advanced-topics#default-runtime)),
by editing the file `/etc/docker/daemon.json` with content:
```json
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
         } 
    },
    "default-runtime": "nvidia" 
}
```
then restart docker daemon:
```bash
sudo systemctl restart docker
```

After these steps, user can run the following command to build docker container:
```bash
docker build -t sparsetir .
```

## Run experiments

Below is the script to reproduce experiments in SparseTIR paper, each script would emit logging files and figures in pdf format.

```bash
# Run SpMM experiments
docker run -it --gpus all -v $(pwd)/spmm/:/root/spmm sparsetir /bin/bash -c 'cd spmm && bash run.sh'
# Run SDDMM experiments
docker run -it --gpus all -v $(pwd)/sddmm/:/root/sddmm sparsetir /bin/bash -c 'cd sddmm && bash run.sh'
# Run GraphSAGE training experiments
docker run -it --gpus all -v $(pwd)/e2e/:/root/e2e sparsetir /bin/bash -c 'cd e2e && bash run.sh'
# Run RGCN inference experiments
docker run -it --gpus all -v $(pwd)/rgcn/:/root/rgcn sparsetir /bin/bash -c 'cd rgcn && bash run.sh'
# Run Sparse Attention experiments
docker run -it --gpus all -v $(pwd)/sparse-attention/:/root/sparse-attention sparsetir /bin/bash -c 'cd sparse-attention && bash run.sh'
# Run PrunedBERT experiments
docker run -it --gpus all -v $(pwd)/pruned-bert/:/root/pruned-bert sparsetir /bin/bash -c 'cd pruned-bert && bash run.sh'
# Run Sparse Convolution experiments
docker run -it --gpus all -v $(pwd)/sparse-conv/:/root/sparse-conv sparsetir /bin/bash -c 'cd sparse-conv && bash run.sh'
```

User can use `run-all.sh` script to run all experiments:
```bash
bash run-all.sh
```


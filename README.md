# SparseTIR Artifact

This repo stores scripts for setting up environments and reproducing results in paper [SparseTIR: Composable Abstractions for Deep Learning](https://arxiv.org/abs/2207.04606), please checkout [SparseTIR repo](https://github.com/uwsampl/sparsetir) for core implementation of SparseTIR.

We suggest using NVIDIA container to setup environments, please install these packages below by following their instructions:
- [Docker](https://docs.docker.com/engine/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## Clone the Repository

```bash
git clone git@github.com:uwsampl/sparsetir-artifact.git --recursive
cd sparsetir-artifact
```
## Setup Docker Image

### Pull from Docker Hub

We provide a pre-built docker image available on Docker Hub which is compatible with Ampere architecture NVIDIA GPUs, user can pull it with:
```
docker pull expye/sparestir-ae:latest
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
docker build -t sparsetir-artifact .
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

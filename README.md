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
## Build Docker Container

```bash
docker build -t sparsetir-artifact .
```
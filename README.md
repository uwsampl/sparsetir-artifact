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

## Run experiments

```bash
# Run SpMM experiments
docker run -it --gpus all -v $(pwd)/spmm/:/root/spmm sparsetir spmm/run.sh
# Run SDDMM experiments
docker run -it --gpus all -v $(pwd)/sddmm/:/root/sddmm sparsetir sddmm/run.sh
# Run GraphSAGE training experiments
docker run -it --gpus all -v $(pwd)/e2e/:/root/e2e sparsetir sddmm/run.sh
# Run RGCN inference experiments
docker run -it --gpus all -v $(pwd)/rgcn/:/root/rgcn sparsetir rgcn/run.sh
# Run Sparse Attention experiments
docker run -it --gpus all -v $(pwd)/sparse-attention/:/root/sparse-attention sparsetir sparse-attention/run.sh
# Run PrunedBERT experiments
docker run -it --gpus all -v $(pwd)/pruned-bert/:/root/pruned-bert sparsetir pruned-bert/run.sh
# Run Sparse Convolution experiments
docker run -it --gpus all -v $(pwd)/sparse-conv/:/root/sparse-conv sparsetir sparse-conv/run.sh
```
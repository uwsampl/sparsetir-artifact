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

Before building the artifact, user need to select nvidia runtime to enable GPU access when buildling docker containers ([reference](https://github.com/NVIDIA/nvidia-docker/wiki/Advanced-topics#default-runtime)),
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
then restart docker daemonn:
```bash
sudo systemctl restart docker
```


After these steps, user can run the following command to build docker container:
```bash
docker build -t sparsetir-artifact .
```

## Run experiments

```bash
# Run SpMM experiments
docker run -it --gpus all -v $(pwd)/spmm/:/root/spmm sparsetir cd spmm && bash run.sh
# Run SDDMM experiments
docker run -it --gpus all -v $(pwd)/sddmm/:/root/sddmm sparsetir cd sddmm && bash run.sh
# Run GraphSAGE training experiments
docker run -it --gpus all -v $(pwd)/e2e/:/root/e2e sparsetir cd e2e && bash run.sh
# Run RGCN inference experiments
docker run -it --gpus all -v $(pwd)/rgcn/:/root/rgcn sparsetir cd rgcn && bash rgcn/run.sh
# Run Sparse Attention experiments
docker run -it --gpus all -v $(pwd)/sparse-attention/:/root/sparse-attention cd sparse-attention && bash run.sh
# Run PrunedBERT experiments
docker run -it --gpus all -v $(pwd)/pruned-bert/:/root/pruned-bert sparsetir cd pruned-bert && bash run.sh
# Run Sparse Convolution experiments
docker run -it --gpus all -v $(pwd)/sparse-conv/:/root/sparse-conv sparsetir cd sparse-conv && bash run.sh
```

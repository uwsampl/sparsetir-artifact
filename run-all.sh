# Run SpMM experiments
docker run -it --gpus all -v $(pwd)/spmm/:/root/spmm sparsetir /bin/bash -c 'cd spmm && bash run.sh'
# # Run SDDMM experiments
docker run -it --gpus all -v $(pwd)/sddmm/:/root/sddmm sparsetir /bin/bash -c 'cd sddmm && bash run.sh'
# # Run GraphSAGE training experiments
docker run -it --gpus all -v $(pwd)/e2e/:/root/e2e sparsetir /bin/bash -c 'cd e2e && bash run.sh'
# # Run RGCN inference experiments
docker run -it --gpus all -v $(pwd)/rgcn/:/root/rgcn sparsetir /bin/bash -c 'cd rgcn && bash run.sh'
# # Run Sparse Attention experiments
docker run -it --gpus all -v $(pwd)/sparse-attention/:/root/sparse-attention sparsetir /bin/bash -c 'cd sparse-attention && bash run.sh'
# # Run PrunedBERT experiments
docker run -it --gpus all -v $(pwd)/pruned-bert/:/root/pruned-bert sparsetir /bin/bash -c 'cd pruned-bert && bash run.sh'
# # Run Sparse Convolution experiments
docker run -it --gpus all -v $(pwd)/sparse-conv/:/root/sparse-conv sparsetir /bin/bash -c 'cd sparse-conv && bash run.sh'

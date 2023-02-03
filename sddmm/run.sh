#!/bin/bash

if [ ! -d data/ ]
then
  python3 dump_nnz.py
fi

for dataset in cora citeseer pubmed ppi arxiv proteins reddit
do
  # sparsetir & dgl
  python3 bench_sddmm.py -d ${dataset} > sparsetir_${dataset}.log
  for feat_size in 32 64 128 256 512
  do
    # dgsparse
    dgsparse-sddmm data/${dataset}-sddmm.npz ${feat_size} >> dgsparse_${dataset}_${feat_size}.log
    # sputnik
    sputnik_sddmm_benchmark data/${dataset}-sddmm.npz ${feat_size} >> sputnik_${dataset}_${feat_size}.log
  done
done
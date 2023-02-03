#!/bin/bash

if [ ! -d data/ ]
then
  python3 dump_nnz.py
fi

for dataset in cora citeseer pubmed ppi arxiv proteins reddit
do
  # sparsetir & dgl
  python3 bench_spmm_hyb.py -d ${dataset} > sparsetir_${dataset}_hyb.log
  python3 bench_spmm_naive.py -d ${dataset} > sparsetir_${dataset}_naive.log
  for feat_size in 32 64 128 256 512
  do
    # dgsparse
    dgsparse-gespmm data/${dataset}.npz ${feat_size} >> dgsparse_${dataset}_${feat_size}.log
    # sputnik
    sputnik_spmm_benchmark data/${dataset}.npz ${feat_size} >> sputnik_${dataset}_${feat_size}.log
  done
done
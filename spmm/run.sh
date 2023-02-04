#!/bin/bash

if [ ! -d data/ ]
then
  python3 dump_nnz.py
fi

for dataset in cora citeseer pubmed ppi arxiv proteins reddit
do
  # sparsetir & dgl
  echo "Running SparseTIR SpMM w/ hybrid format on ${dataset}"
  python3 bench_spmm_hyb.py -d ${dataset} -i > sparsetir_${dataset}_hyb.log 2> sparsetir_${dataset}_hyb.err
  echo "Running SparseTIR SpMM w/o hybrid format on ${dataset}"
  python3 bench_spmm_naive.py -d ${dataset} > sparsetir_${dataset}_naive.log 2> sparsetir_${dataset}_naive.err
  for feat_size in 32 64 128 256 512
  do
    # dgsparse
    echo "Running dgsparse SpMM on ${dataset}"
    dgsparse-gespmm data/${dataset}.npz ${feat_size} > dgsparse_${dataset}_${feat_size}.log 2>> dgsparse_${dataset}_${feat_size}.log
    # sputnik
    echo "Running sputnik SpMM on ${dataset}"
    sputnik_spmm_benchmark data/${dataset}.npz ${feat_size} > sputnik_${dataset}_${feat_size}.log
    # taco
    echo "Running TACO SpMM on ${dataset}"
    taco_spmm data/${dataset}.npz ${feat_size} > taco_${dataset}_${feat_size}.log
  done
done
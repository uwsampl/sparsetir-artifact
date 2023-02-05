#!/bin/bash

if [ ! -d data/ ]
then
  python3 dump_npz.py
fi

for dataset in cora citeseer pubmed ppi arxiv proteins reddit
do
  # sparsetir & dgl
  echo "Running SparseTIR SDDMM on ${dataset}"
  python3 bench_sddmm.py -d ${dataset} > sparsetir_${dataset}.log
  for feat_size in 32 64 128 256 512
  do
    # dgsparse
    echo "Running dgsparse SDDMM on ${dataset}, feat_size = ${feat_size}"
    dgsparse-sddmm data/${dataset}-sddmm.npz ${feat_size} > dgsparse_${dataset}_${feat_size}.log 2> dgsparse_${dataset}_${feat_size}.err
    # sputnik
    echo "Running sputnik SDDMM on ${dataset}, feat_size = ${feat_size}"
    sputnik_sddmm_benchmark data/${dataset}-sddmm.npz ${feat_size} > sputnik_${dataset}_${feat_size}.log 2> sputnik_${dataset}_${feat_size}.err
    # taco
    echo "Running taco SDDMM on ${dataset}, feat_size = ${feat_size}"
    taco-sddmm data/${dataset}.npz ${feat_size} > taco_${dataset}_${feat_size}.log 2> taco_${dataset}_${feat_size}.err
  done
done

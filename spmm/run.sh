#!/bin/bash

# Benchmark
echo "Running SpMM benchmark..."
if [ ! -d data/ ]
then
  python3 dump_npz.py > dump_npz.log 2> dump_npz.err
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
    echo "Running dgsparse SpMM on ${dataset}, feat_size = ${feat_size}"
    dgsparse-gespmm data/${dataset}.npz ${feat_size} > dgsparse_${dataset}_${feat_size}.log 2> dgsparse_${dataset}_${feat_size}.log
    # sputnik
    echo "Running sputnik SpMM on ${dataset}, feat_size = ${feat_size}"
    sputnik_spmm_benchmark data/${dataset}.npz ${feat_size} > sputnik_${dataset}_${feat_size}.log 2> sputnik_${dataset}_${feat_size}.err
    # taco
    echo "Running TACO SpMM on ${dataset}, feat_size = ${feat_size}"
    taco-spmm data/${dataset}.npz ${feat_size} > taco_${dataset}_${feat_size}.log 2> taco_${dataset}_${feat_size}.err
  done
done

# Extract data
echo "Extracting data from log files..."
python3 extract_data.py

# Plot figures
echo "Plotting figures..."
python3 plot.py

echo "Evaluation finished, see spmm.pdf for results."

#!/bin/bash

# Benchmark
echo "Running GraphSAGE end-to-end training benchmark..."
for dataset in cora citeseer pubmed ppi arxiv proteins reddit
do
  echo "Running DGL GraphSAGE Training on ${dataset}"
  python3 sage_dgl.py -d ${dataset} > dgl_${dataset}.log 2> dgl_${dataset}.err
  echo "Running SparseTIR GraphSAGE Training on ${dataset}"
  python3 sage_sparse_tir.py -d ${dataset} > sparsetir_${dataset}.log 2> sparsetir_${dataset}.err
done

# Extract data
echo "Extracting data from log files..."
python3 extract_data.py

# Plot figures
echo "Plotting figures..."
python3 plot.py

echo "Evaluation finished, see `graphsage-e2e.pdf` for results."

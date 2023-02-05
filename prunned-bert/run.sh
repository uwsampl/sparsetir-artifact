#!/bin/bash

# Benchmark
echo "Running PrunedBERT benchmark..."
echo "Running structured prunning experiments..."
python3 structured-transposed-single-op.py -d 512 -c > structured.log 2> structured.err
echo "Running unstructured prunning experiments..."
python3 unstructured-transposed-single-op.py -d 512 -c > unstructured.log 2> unstructured.err

# Extract data
echo "Extracting data from log files..."
python3 extract_data.py

# Plot figures
echo "Plotting structured pruning figures..."
gnuplot structured.plt
epstopdf structured.ps
rm structured.ps

echo "Plotting unstructured pruning figures..."
gnuplot unstructured.plt
epstopdf unstructured.ps
rm unstructured.ps

echo "Finished evaluation, see structured.pdf and unstructured.pdf for the results."

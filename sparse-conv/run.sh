#!/bin/bash

# Benchmark
echo "Running sparse convolution benchmark..."
python3 benchmark.py > sparseconv.log 2> sparseconv.err

# Extract data
echo "Extracting data from log files..."
python3 extract_data.py

# Plot figures
echo "Plotting figures..."
gnuplot sparseconv.plt
epstopdf sparseconv.ps
rm sparseconv.ps

echo "Evaluation finished, see `sparseconv.pdf` for results."

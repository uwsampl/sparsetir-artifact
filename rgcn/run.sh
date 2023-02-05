#!/bin/bash

# Benchmark
echo "Running RGCN inference benchmark..."
python RGCN.py all 32 > rgcn.log 2> rgcn.err

# Extract data
echo "Extracting data from log files..."
python3 extract_data.py

# Plot figures
echo "Plotting figures..."
python3 plot.py

echo "Evaluation finished, see rgcn-e2e.pdf for results."

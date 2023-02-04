#!/bin/bash

SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
echo "Running sparse convolution benchmark..."
python3 ${SCRIPTPATH}/benchmark.py
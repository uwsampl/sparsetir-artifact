#!/bin/bash

echo "Running structured prunning experiments..."
python3 structured-transposed-single-op.py -d 512 -c > structured.log 2> structured.err
echo "Running unstructured prunning experiments..."
python3 unstructured-transposed-single-op.py -d 512 -c > unstructured.log 2> unstructured.err

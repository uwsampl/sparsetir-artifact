#!/bin/bash

for backend in csr sparse_tir triton_blocksparse
  do
  for pattern in pixelfly longformer
  do
    for op in spmm sddmm
    do
      echo "Running ${backend} + ${pattern} + ${op}..."
      python3 ${backend}_${op}.py -p ${pattern} > ${backend}_${op}_${pattern}.log 2> ${backend}_${op}_${pattern}.err
    done
  done
done
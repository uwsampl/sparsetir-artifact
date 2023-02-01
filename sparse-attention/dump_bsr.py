from utils import create_pixelfly
import os
import numpy as np

if not os.path.exists("data"):
    os.mkdir("data")

if __name__ == "__main__":
    seq_len = 4096
    block_size = 16
    mb = seq_len // block_size
    csr = create_pixelfly(1, mb, fmt="bsr", block_size=block_size)
    nnz_cols = csr.nnz / mb

    shp = np.array([block_size, seq_len, nnz_cols], dtype=np.int32)
    np.savez("data/butterfly.npz", shape=shp, indptr=csr.indptr, indices=csr.indices)

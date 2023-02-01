import torch
import numpy as np

# import matplotlib
# from matplotlib import pyplot as plt
from scipy.sparse import coo_matrix


def create_pixelfly(h, mb, fmt="mask", block_size=1):
    stride = mb // 2
    rows = []
    cols = []
    while stride >= 1:
        for blk_id in range(mb // stride // 2):
            for i in range(2):
                for j in range(2):
                    for k in range(stride):
                        rows.append(blk_id * (stride * 2) + i * stride + k)
                        cols.append(blk_id * (stride * 2) + j * stride + k)
        stride >>= 1
    rows = torch.tensor(rows)
    cols = torch.tensor(cols)

    if fmt == "mask":
        mask = torch.zeros(h, mb, mb, dtype=torch.int32)
        mask[:, rows, cols] = 1
        return mask
    elif fmt == "bsr":
        rows = rows.numpy()
        cols = cols.numpy()
        coo = coo_matrix((np.ones_like(rows), (rows, cols)), shape=(mb, mb))
        csr = coo.tocsr()
        return csr
    elif fmt == "csr":
        rows = (
            rows.view(-1, 1, 1) * block_size
            + torch.arange(block_size, dtype=torch.int32).view(1, block_size, 1)
            + torch.zeros(1, 1, block_size, dtype=torch.int32)
        )
        cols = (
            cols.view(-1, 1, 1) * block_size
            + torch.zeros(1, block_size, 1, dtype=torch.int32)
            + torch.arange(block_size, dtype=torch.int32).view(1, 1, block_size)
        )
        rows = rows.view(-1).numpy()
        cols = cols.view(-1).numpy()
        coo = coo_matrix(
            (np.ones_like(rows), (rows, cols)), shape=(mb * block_size, mb * block_size)
        )
        csr = coo.tocsr()
        return csr
    else:
        raise KeyError("Format {} not recognized.".format(fmt))


def create_longformer(h, mb, window, fmt="mask", block_size=1):
    rows = []
    cols = []
    for i in range(mb):
        for j in range(window):
            if i + j >= mb:
                continue
            rows.append(i)
            cols.append(i + j)
    rows = torch.tensor(rows, dtype=torch.long)
    cols = torch.tensor(cols, dtype=torch.long)

    if fmt == "mask":
        mask = torch.zeros(h, mb, mb, dtype=torch.int32)
        mask[:, rows, cols] = 1
        return mask
    elif fmt == "bsr":
        rows = rows.numpy()
        cols = cols.numpy()
        coo = coo_matrix((np.ones_like(rows), (rows, cols)), shape=(mb, mb))
        csr = coo.tocsr()
        return csr
    elif fmt == "csr":
        rows = (
            rows.view(-1, 1, 1) * block_size
            + torch.arange(block_size, dtype=torch.int32).view(1, block_size, 1)
            + torch.zeros(1, 1, block_size, dtype=torch.int32)
        )
        cols = (
            cols.view(-1, 1, 1) * block_size
            + torch.zeros(1, block_size, 1, dtype=torch.int32)
            + torch.arange(block_size, dtype=torch.int32).view(1, 1, block_size)
        )
        rows = rows.view(-1).numpy()
        cols = cols.view(-1).numpy()
        coo = coo_matrix(
            (np.ones_like(rows), (rows, cols)), shape=(mb * block_size, mb * block_size)
        )
        csr = coo.tocsr()
        return csr
    else:
        raise KeyError("Format {} not recognized.".format(fmt))


if __name__ == "__main__":
    create_pixelfly(1, 256, "csr")

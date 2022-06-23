import torch
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.sparse import coo_matrix

def create_pixelfly(h, m, fmt='mask'):
    stride = m // 2
    rows = []
    cols =[]
    while stride >= 1:
        for blk_id in range(m // stride // 2):
            for i in range(stride * 2):
                for j in range(2):
                    rows.append(blk_id * (stride * 2) + i)
                    cols.append(blk_id * (stride * 2) + i % stride + j * stride)
        stride >>= 1
    rows = torch.tensor(rows)
    cols = torch.tensor(cols)
 
    if fmt == 'mask':
        mask = torch.zeros(h, m, m, dtype=torch.int32)
        mask[:, rows, cols] = 1
        return mask
    elif fmt == 'csr':
        rows = rows.numpy()
        cols = cols.numpy()
        coo = coo_matrix((np.ones_like(rows), (rows, cols)), shape=(m, m))
        csr = coo.tocsr() 
        plt.spy(csr)
        plt.savefig("1.pdf")
        return csr
    else:
        raise KeyError("Format {} not recognized.".format(fmt))
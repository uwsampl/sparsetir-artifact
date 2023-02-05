import os
import numpy as np
import pandas


def extract_data():
    tbl = pandas.read_csv("sparse_conv.csv")
    feat_in = tbl['feat_in'].to_numpy()
    feat_out = tbl['feat_out'].to_numpy()
    dur_torchsparse = tbl["dur_torchsparse"].to_numpy()
    dur_sparsetir = tbl["dur_sparsetir"].to_numpy()
    sqrt_c = np.sqrt(feat_in * feat_out)
    speedup = dur_torchsparse / dur_sparsetir

    with open("sparseconv.dat", "w") as f:
        f.write("sqrt_channel SparseTIR(TC) TorchSparse\n")
        for i in range(len(sqrt_c)):
            f.write("{} {} {}\n".format(sqrt_c[i], speedup[i], 1))


if __name__ == "__main__":
    extract_data()
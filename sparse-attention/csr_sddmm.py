from utils import create_pixelfly, create_longformer

import dgl
import argparse
import torch
from sparsetir_profiler import profile_pytorch_ms


def test_csr_sddmm(pattern: str):
    if pattern == "pixelfly":
        csr = create_pixelfly(1, 4096 // 16, fmt="csr", block_size=16)
    elif pattern == "longformer":
        csr = create_longformer(1, 4096 // 16, 256 // 16, fmt="csr", block_size=16)
    else:
        raise KeyError("Pattern {} not supported.".format(pattern))
    g = dgl.from_scipy(csr).int()
    g = g.to(0)
    a_gpu = torch.rand(4096, 64).to(0)
    b_gpu = torch.rand(4096, 64).to(0)

    measure = profile_pytorch_ms(lambda: dgl.ops.u_dot_v(g, a_gpu, b_gpu))
    print("cusparse csrmm time: \t{:.5f}ms".format(measure))
    return measure


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CSR sddmm")
    parser.add_argument(
        "--pattern", "-p", type=str, help="Sparse pattern: longformer/pixelfly"
    )
    args = parser.parse_args()
    test_csr_sddmm(args.pattern)

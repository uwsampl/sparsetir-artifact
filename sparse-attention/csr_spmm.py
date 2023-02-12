from utils import create_pixelfly, create_longformer

import dgl
import torch
import argparse
from sparsetir_artifact import profile_pytorch_ms


def test_csr_spmm(pattern: str):
    num_heads = 12
    if pattern == "pixelfly":
        csr = create_pixelfly(1, 4096 // 16, fmt="csr", block_size=16)
    elif pattern == "longformer":
        csr = create_longformer(1, 4096 // 16, 256 // 16, fmt="csr", block_size=16)
    else:
        raise KeyError("Pattern {} not supported.".format(pattern))
    g = dgl.from_scipy(csr).int()
    g = g.to(0)
    w_gpu = torch.rand(num_heads, g.num_edges()).half().to(0)
    x_gpu = torch.rand(num_heads, 4096, 64).half().to(0)

    measure = profile_pytorch_ms(
        lambda: [
            dgl.ops.u_mul_e_sum(g, x_gpu[head], w_gpu[head])
            for head in range(num_heads)
        ]
    )
    print("cusparse csrmm time: \t{:.5f} ms".format(measure))
    return measure


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CSR spmm")
    parser.add_argument(
        "--pattern", "-p", type=str, help="Sparse pattern: longformer/pixelfly"
    )
    parser.add_argument(
        "--check", "-c", action="store_true", help="Whether to check result or not."
    )
    args = parser.parse_args()
    test_csr_spmm(args.pattern)

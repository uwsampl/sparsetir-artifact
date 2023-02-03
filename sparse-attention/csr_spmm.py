from utils import create_pixelfly, create_longformer

from torch.profiler import profile, ProfilerActivity, schedule
import dgl
import torch
import argparse


def test_csr_spmm(pattern: str):
    if pattern == "pixelfly":
        csr = create_pixelfly(1, 4096 // 16, fmt="csr", block_size=16)
    elif pattern == "longformer":
        csr = create_longformer(1, 4096 // 16, 256 // 16, fmt="csr", block_size=16)
    else:
        raise KeyError("Pattern {} not supported.".format(pattern))
    g = dgl.from_scipy(csr).int()
    g = g.to(0)
    x_gpu = torch.rand(4096, 64).to(0)
    wait = 1
    warmup = 10
    active = 100
    with profile(
        activities=[ProfilerActivity.CUDA],
        schedule=schedule(wait=wait, warmup=warmup, active=active),
    ) as prof:
        for _ in range(wait + warmup + active):
            y_gpu = dgl.ops.copy_u_sum(g, x_gpu)
            prof.step()

    measure = sum([e.cuda_time for e in prof.events()]) / 1000 / active
    print("cusparse csrmm time: \t{:.5f}ms".format(measure))
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
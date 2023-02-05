import dgl
import sys
import argparse
import torch as th
from sparsetir_artifact import profile_pytorch_ms
from utils import get_dataset


def bench_sddmm(g: dgl.DGLGraph, feat_size: int):
    m = g.num_src_nodes()
    n = g.num_dst_nodes()
    a_gpu = th.rand(m, feat_size).to(th.float32).to(0)
    b_gpu = th.rand(n, feat_size).to(th.float32).to(0)
    g = g.to(0)
    dur = profile_pytorch_ms(lambda: dgl.ops.u_dot_v(g, a_gpu, b_gpu))
    print("dgl time:\t{:.5f} ms".format(dur))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("sddmm in dgl")
    parser.add_argument(
        "--dataset", "-d", type=str, default="pubmed", help="dataset name"
    )
    args = parser.parse_args()
    name = args.dataset
    g = get_dataset(name)
    for feat_size in [32, 64, 128, 256, 512]:
        print("feat_size = ", feat_size)
        try:
            bench_sddmm(g, feat_size)
        except Exception as e:
            print("OOM")
            print(e, file=sys.stderr)

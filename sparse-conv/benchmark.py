import glob
import torch
import dgl
import numpy as np
import pandas as pd
import tvm
import os
from torchsparse.nn.functional.conv import ConvolutionFunction
from rgcn import rgcn_tensorcore
from sparsetir_artifact import profile_pytorch_ms


def get_type_pointers(g: dgl.DGLHeteroGraph):
    ntype_pointer = np.cumsum([0] + [g.number_of_nodes(ntype) for ntype in g.ntypes])

    etype_pointer = [0]
    for etype in g.canonical_etypes:
        g_sub = g[etype]
        etype_pointer.append(etype_pointer[-1] + g_sub.num_edges())

    return {
        "ntype_node_pointer": torch.IntTensor(ntype_pointer),
        "etype_edge_pointer": torch.IntTensor(etype_pointer),
    }


device = "cuda:0"
buffer = torch.zeros(
    400000 * 64, dtype=torch.float16, device=device, requires_grad=False
)

results = []

home = os.path.expanduser("~")
path = os.path.join(home, "layers")
if not os.path.exists(path):
    raise RuntimeError("Data not downloaded.")

for fpath in glob.glob(os.path.join(path, "*.pth")):
    data = torch.load(fpath)

    inputs = data["input"].half().to(device)
    outputs = data["output"].half().to(device)
    weights = data["weight"].half().to(device)
    nbmaps = data["nbmaps"].to(device)
    nbsizes = data["nbsizes"]
    input_mask = data["input_mask"].to(device)
    output_mask = data["output_mask"].to(device)
    transposed = data["transposed"]

    feat_in, feat_out = inputs.shape[1], outputs.shape[1]
    print(feat_in, feat_out, transposed)

    if feat_in < 32:
        torch.cuda.empty_cache()
        continue

    dur_torchsparse = profile_pytorch_ms(
        lambda: ConvolutionFunction.apply(
            inputs,
            weights,
            nbmaps,
            nbsizes,
            buffer,
            (inputs.shape[0], outputs.shape[0]),
            input_mask,
            output_mask,
            0.0,
            0,
            1,
            transposed,
        )
    )

    print("torchsparse time: \t{:5f}ms".format(dur_torchsparse))

    # use sparsetir op
    graph_data = {}
    offset = 0
    sizes = data["nbsizes"]
    maps = data["nbmaps"]
    for rel in range(weights.shape[0]):
        i = maps[offset : offset + sizes[rel], 0]
        o = maps[offset : offset + sizes[rel], 1]
        offset += sizes[rel]
        if transposed:
            i, o = o, i
        graph_data[("src", str(rel), "dst")] = (i, o)
    g = dgl.heterograph(graph_data)
    type_pointers = get_type_pointers(g)
    y1, dur_sparsetir = rgcn_tensorcore(
        g,
        type_pointers,
        feat_in,
        feat_out,
        inputs,
        weights.transpose(-1, -2).contiguous(),
        ty=8,
        num_workloads_per_thread=1,
        buckets=[1],
    )

    results.append(
        (
            feat_in,
            feat_out,
            fpath,
            g.num_dst_nodes(),
            g.num_src_nodes(),
            g.num_edges(),
            dur_torchsparse,
            dur_sparsetir,
        )
    )

results.sort(key=lambda x: x[-1])

pd = pd.DataFrame(
    data={
        "feat_in": [_[0] for _ in results],
        "feat_out": [_[1] for _ in results],
        "fpath": [_[2] for _ in results],
        "m": [_[3] for _ in results],
        "n": [_[4] for _ in results],
        "nnz": [_[5] for _ in results],
        "dur_torchsparse": [_[6] for _ in results],
        "dur_sparsetir": [_[7] for _ in results],
    }
)
pd.to_csv("sparse_conv.csv", index=False)

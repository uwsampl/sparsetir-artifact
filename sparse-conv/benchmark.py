import glob
import torch
import dgl
import numpy as np
import tvm
from torchsparse.nn.functional.conv import ConvolutionFunction
from torch.profiler import profile, ProfilerActivity, schedule
from rgcn import rgcn_tensorcore


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

buffer = torch.zeros(4000000 * 64,
                     dtype=torch.float16,
                     device=device,
                     requires_grad=False)

for fpath in glob.glob("layers/*.pth"):
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

    # s = 0
    # for k in range(weights.shape[0]):
    #     i = maps[s:s + sizes[k], 0].long()
    #     o = maps[s:s + sizes[k], 1].long()
    #     s += sizes[k]
    #     if transposed:
    #         i, o = o, i
    #     outputs[o] += torch.mm(inputs[i], weights[k])

    if feat_in < 32:
        continue

    with profile(activities=[ProfilerActivity.CUDA],
                schedule=schedule(wait=0, warmup=10, active=100)) as prof:
        for _ in range(100):
            y = ConvolutionFunction.apply(
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
                transposed
            )
            prof.step()

    measure = sum([e.cuda_time for e in prof.events()]) / 1000 / 90

    print("torchsparse time: \t{:5f}ms".format(measure))
    
    # use sparsetir op
    graph_data = {}
    offset = 0
    sizes = data["nbsizes"]
    maps = data["nbmaps"]
    for rel in range(weights.shape[0]):
        i = maps[offset: offset + sizes[rel], 0]
        o = maps[offset: offset + sizes[rel], 1]
        offset += sizes[rel]
        if transposed:
            i, o = o, i
        graph_data[('src', str(rel), 'dst')] = (i, o)
    g = dgl.heterograph(graph_data)
    type_pointers = get_type_pointers(g)
    y1 = rgcn_tensorcore(
        g,
        type_pointers,
        feat_in,
        feat_out,
        inputs,
        weights.transpose(-1, -2).contiguous(),
        ty=8,
        num_workloads_per_thread=1,
        buckets=[1]
    )

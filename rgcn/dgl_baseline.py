from dgl.heterograph import DGLHeteroGraph
import scipy.sparse as sp
import numpy as np
import dgl
import dgl.function as fn
from dgl.nn.pytorch.linear import TypedLinear
import torch as th
from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset


def get_dataset_by_name(name: str):
    if name == "aifb":
        return AIFBDataset()
    elif name == "mutag":
        return MUTAGDataset()
    elif name == "bgs":
        return BGSDataset()
    elif name == "am":
        return AMDataset()
    else:
        raise KeyError("Unknown dataset {}.".format(name))


class TorchOpTimer(object):
    def __enter__(self):
        self.start_event = th.cuda.Event(enable_timing=True)
        self.end_event = th.cuda.Event(enable_timing=True)
        self.start_event.record()
        return self

    def __exit__(self, type, value, traceback):
        self.end_event.record()
        th.cuda.synchronize()  # Wait for the events to be recorded!
        self.time = self.start_event.elapsed_time(self.end_event)


def prepare_hetero_graph_simplified(g: dgl.DGLHeteroGraph):
    ntype_pointer = np.cumsum([0] + [g.number_of_nodes(ntype) for ntype in g.ntypes])

    etype_pointer = [0]
    for etype in g.canonical_etypes:
        g_sub = g[etype]
        etype_pointer.append(etype_pointer[-1] + g_sub.num_edges())

    return {
        "ntype_node_pointer": th.IntTensor(ntype_pointer).cuda(),
        "etype_edge_pointer": th.IntTensor(etype_pointer).cuda(),
    }


def test_rgcn(g: DGLHeteroGraph, feat_size: int):
    g = g.to(0)
    feat = th.rand(g.num_src_nodes(), feat_size).to(0) / 100
    out = th.zeros(g.num_dst_nodes(), feat_size).to(0) / 100
    weight = th.rand(g.num_rels, feat_size, feat_size).to(0)
    indptr, indices, eid = g.adj_sparse(fmt="csc")
    etype = g.edata[dgl.ETYPE][eid]

    cold_start = 10
    total = 100
    accum = 0

    # dgl-lowmem
    try:
        g.srcdata["feat"] = feat.unsqueeze(-1)
        us, vs = g.edges()
        feat_transformed = feat[us]
        msg = th.zeros(g.num_edges(), feat_size).to(0)
        for epoch in range(total):
            with TorchOpTimer() as timer:
                with th.no_grad():
                    for i in range(1, len(g.etype_pointer)):
                        start = g.etype_pointer[i - 1]
                        end = g.etype_pointer[i]
                        msg[start:end] = feat_transformed[start:end] @ weight[i - 1]
                    y_dgl_lowmem = dgl.ops.copy_e_sum(g, msg)
            if epoch >= cold_start:
                accum += timer.time
        print("dgl-lowmem:\t\t {}ms".format(accum / (total - cold_start)))
        y_dgl_lowmem = None
    except RuntimeError as err:
        print("dgl-lowmem: OOM")
        y_dgl_lowmem = None
    except BaseException as err:
        print(err)
        raise

    # dgl-typed-linear
    try:
        g.srcdata["feat"] = feat.unsqueeze(-1)
        linear_r = TypedLinear(feat_size, feat_size, len(g.etype_pointer)).to(0)
        us, vs = g.edges()
        feat_transformed = feat[us]
        for epoch in range(total):
            with TorchOpTimer() as timer:
                with th.no_grad():
                    msg = linear_r(feat_transformed, g.edata[dgl.ETYPE], True)
                    y_dgl_typed_linear = dgl.ops.copy_e_sum(g, msg)
            if epoch >= cold_start:
                accum += timer.time
        print("dgl-typed-linear:\t {}ms".format(accum / (total - cold_start)))
    except RuntimeError as err:
        y_dgl_typed_linear = None
        print("dgl-typed-linear: OOM")
    except BaseException as err:
        print(err)
        raise

    # dgl-bmm

    def msg_func(edges):
        h = edges.src["feat"]
        W = weight[edges.data[dgl.ETYPE]]
        return {"msg": W @ h}

    try:
        g.srcdata["feat"] = feat.unsqueeze(-1)
        for epoch in range(total):
            with TorchOpTimer() as timer:
                with th.no_grad():
                    g.update_all(msg_func, fn.sum("msg", "y"))
                    y_dgl = g.dstdata["y"].squeeze(-1)
            if epoch >= cold_start:
                accum += timer.time
        print("dgl-bmm:\t\t {}ms".format(accum / (total - cold_start)))
    except RuntimeError as err:
        print("dgl-bmm: OOM")
        y_dgl = None
    except BaseException as err:
        print(err)
        raise


if __name__ == "__main__":
    for feat_size in [16, 32]:#[4, 8, 16, 32, 64]:
        for name in ['bgs']:#["aifb", "mutag", "bgs", "am"]:
            print("dataset {}, feat_size={}:".format(name, feat_size))
            dataset = get_dataset_by_name(name)
            g = dataset[0]
            type_pointers = prepare_hetero_graph_simplified(g)
            g = dgl.to_homogeneous(g)
            g.ntype_pointer = type_pointers["ntype_node_pointer"]
            g.etype_pointer = type_pointers["etype_edge_pointer"]
            g.num_ntypes = max(g.ndata[dgl.NTYPE]).item() + 1
            g.num_rels = max(g.edata[dgl.ETYPE]).item() + 1
            test_rgcn(g, feat_size)

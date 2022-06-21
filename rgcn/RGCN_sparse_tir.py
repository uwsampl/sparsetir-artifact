import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tvm
import tvm.tir as tir
from tvm.script import tir as T
from tvm.sparse import lower_sparse_iter, lower_sparse_buffer
from torch.utils.dlpack import to_dlpack as th_to_dlpack
from torch.utils.dlpack import from_dlpack as th_from_dlpack

cached_kernel = None

@T.prim_func
def rgcn_forward(
    etype: T.handle,
    w: T.handle,
    x: T.handle,
    y: T.handle,
    indptr: T.handle,
    indices: T.handle,
    n: T.int32,
    r: T.int32,
    in_feat: T.int32,
    out_feat: T.int32,
    nnz: T.int32,
):
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    I = T.dense_fixed(n)
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), "int32")
    J_detach = T.dense_fixed(n)
    R = T.dense_fixed(r)
    F_in = T.dense_fixed(in_feat)
    F_out = T.dense_fixed(out_feat)
    E = T.match_sparse_buffer(etype, (I, J), "int32")
    W = T.match_sparse_buffer(w, (R, F_out, F_in), "float32")
    X = T.match_sparse_buffer(x, (J_detach, F_in), "float32")
    Y = T.match_sparse_buffer(y, (I, F_out), "float32")
    with T.iter([I, F_out, J, F_in], "SSRR", "rgcn-forward") as [
        i,
        fo,
        j,
        fi,
    ]:
        with T.init():
            Y[i, fo] = 0.0
        Y[i, fo] = Y[i, fo] + W[E[i, j], fo, fi] * X[j, fi]


@T.prim_func
def rgcn_hetero_forward_2(
    w: T.handle,
    x: T.handle,
    y: T.handle,
    etypes: T.handle,
    indptr_i: T.handle,
    indices_i: T.handle,
    indptr_j: T.handle,
    indices_j: T.handle,
    n: T.int32,
    num_rels: T.int32,
    group: T.int32,
    feat_in: T.int32,
    feat_out: T.int32,
    nnz_i: T.int32,
    nnz_j: T.int32,
):
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    R = T.dense_fixed(num_rels)
    G = T.dense_fixed(group)
    I = T.sparse_variable(G, (n, nnz_i), (indptr_i, indices_i), "int32")
    J = T.sparse_variable(I, (n, nnz_j), (indptr_j, indices_j), "int32")
    I_detach = T.dense_fixed(n)
    J_detach = T.dense_fixed(n)
    F_in = T.dense_fixed(feat_in)
    F_out = T.dense_fixed(feat_out)
    W = T.match_sparse_buffer(w, (R, F_out, F_in), "float32")
    X = T.match_sparse_buffer(x, (J_detach, F_in), "float32")
    Y = T.match_sparse_buffer(y, (I_detach, F_out), "float32")
    E = T.match_sparse_buffer(etypes, (G,), "int32")
    with T.iter([F_out, G, I, J, F_in], "SSSRR", "rgcn-hetero-forward") as [fo, g, i, j, fi]:
        with T.init():
            Y[i, fo] = 0.
        Y[i, fo] = Y[i, fo] + W[E[g], fo, fi] * X[j, fi]


def create_homogeneous_kernel(g, W, etype, in_feat, out_feat):
    # tir
    N, R, IN_FEAT, OUT_FEAT, NNZ = rgcn_forward.params[-5:]
    mod = tvm.IRModule.from_expr(
        rgcn_forward.specialize(
            {N: g.number_of_nodes(), R: g.num_rels, IN_FEAT: in_feat, OUT_FEAT: out_feat, NNZ: g.number_of_edges()}
        )
    )
    mod = lower_sparse_iter(mod)
    sch = tir.Schedule(mod["main"])

    outer = sch.get_block("rgcn-forward0")
    inner = sch.get_block("rgcn-forward1")
    i, f_out = sch.get_loops(outer)
    j, f_in = sch.get_loops(inner)
    sch.bind(i, "blockIdx.x")
    sch.bind(f_out, "threadIdx.y")
    sch.bind(f_in, "threadIdx.x")
    mod = lower_sparse_buffer(sch.mod)
    f = tvm.build(mod, target="cuda")
    indptr, indices, eid = g.adj_sparse(fmt='csc')

    indptr_nd = tvm.nd.from_dlpack(th_to_dlpack(indptr.int().view(-1).contiguous()))
    indices_nd = tvm.nd.from_dlpack(th_to_dlpack(indices.int().view(-1).contiguous()))
    W_nd = tvm.nd.from_dlpack(th_to_dlpack(W.view(-1).contiguous()))
    etype = etype[eid].int()
    E_nd = tvm.nd.from_dlpack(th_to_dlpack(etype.view(-1).contiguous()))
    Y = torch.zeros((g.num_dst_nodes(), out_feat)).to(0)

    def foo(X):
        # Y = torch.zeros((g.num_dst_nodes(), out_feat)).to(0)
        X_nd = tvm.nd.from_dlpack(th_to_dlpack(X.view(-1).contiguous()))
        Y_nd = tvm.nd.from_dlpack(th_to_dlpack(Y.view(-1).contiguous()))
        f(E_nd, W_nd, X_nd, Y_nd, indptr_nd, indices_nd)
        return Y
    
    return foo

def prepare_hetero_graph_simplified(g):
    ntype_pointer = np.cumsum([0] + [g.number_of_nodes(ntype) for ntype in g.ntypes])

    etype_pointer = [0]
    for etype in g.canonical_etypes:
        g_sub = g[etype]
        etype_pointer.append(etype_pointer[-1] + g_sub.num_edges())

    return {
        "ntype_node_pointer": torch.IntTensor(ntype_pointer),
        "etype_edge_pointer": torch.IntTensor(etype_pointer),
    }

def create_heterogeneneous_kernel(g, W, in_feat, out_feat):
    g = g.cpu()
    type_pointers = prepare_hetero_graph_simplified(g)
    g.ntype_pointer = type_pointers["ntype_node_pointer"]
    g.etype_pointer = type_pointers["etype_edge_pointer"]
    bucket_size = 128
    split_factor_f = 2
    N, R, GROUP, FEAT_IN, FEAT_OUT, NNZ_I, NNZ_J = rgcn_hetero_forward_2.params[-7:]
    n = g.num_nodes()
    r = len(g.etypes)
    nnz_j = g.num_edges()

    indptr_i = [torch.LongTensor([0])]
    indices_i = []
    indptr_j = [torch.LongTensor([0])]
    indices_j = []
    etypes = []
    for etype in g.canonical_etypes:
        src_type, _, dst_type = etype
        etype_id = g.get_etype_id(etype)
        src_type_id = g.get_ntype_id(src_type)
        dst_type_id = g.get_ntype_id(dst_type)
        g_sub = g[etype]
        indptr, indices, _ = g_sub.adj_sparse(fmt="csc")

        unique_nodes = torch.nonzero(indptr[:-1] != indptr[1:]).squeeze(1)
        start = 0
        node_groups = []
        threshold = 0
        for end in range(0, len(unique_nodes)):
            indptr_val = indptr[unique_nodes[end]].item()
            if indptr_val >= threshold:
                node_groups.append(unique_nodes[start:end])
                start = end
                threshold += bucket_size
                etypes.append(torch.LongTensor([etype_id]))
        node_groups.append(unique_nodes[start:])
        etypes.append(torch.LongTensor([etype_id]))

        for node_group in node_groups:
            indptr_i.append(torch.LongTensor([len(node_group) + indptr_i[-1].item()]))
            indices_i.append(node_group + g.ntype_pointer[dst_type_id])
            indptr_j.append(indptr[node_group + 1] + g.etype_pointer[etype_id])

        indices_j.append(indices + g.ntype_pointer[src_type_id])

    group_size = len(indptr_i) - 1
    etypes = tvm.nd.array(torch.cat(etypes).numpy().astype("int32"), device=tvm.cuda(0))
    indptr_i = tvm.nd.array(torch.cat(indptr_i).numpy().astype("int32"), device=tvm.cuda(0))
    indices_i = tvm.nd.array(torch.cat(indices_i).numpy().astype("int32"), device=tvm.cuda(0))
    indptr_j = tvm.nd.array(torch.cat(indptr_j).numpy().astype("int32"), device=tvm.cuda(0))
    indices_j = tvm.nd.array(torch.cat(indices_j).numpy().astype("int32"), device=tvm.cuda(0))
    W_nd = tvm.nd.from_dlpack(th_to_dlpack(W.view(-1).contiguous()))
    Y = torch.zeros((g.num_dst_nodes(), out_feat)).to(0)

    nnz_i = indices_i.shape[0]
    mod = tvm.IRModule.from_expr(
        rgcn_hetero_forward_2.specialize(
            {N: n, R: r, GROUP: group_size, FEAT_IN: in_feat, FEAT_OUT: out_feat, NNZ_I: nnz_i, NNZ_J: nnz_j}
        )
    )
    mod = lower_sparse_iter(mod)
    sch = tir.Schedule(mod)
    blk0 = sch.get_block("rgcn-hetero-forward0")
    blk1 = sch.get_block("rgcn-hetero-forward1")
    blk2 = sch.get_block("rgcn-hetero-forward2")
    read_blk = sch.cache_read(blk1, 2, "local")
    write_blk = sch.cache_write(blk2, 0, "local")
    sch.annotate(write_blk, "atomic", True)
    f_out, g = sch.get_loops(blk0)
    f_out_o, f_out_i = sch.split(f_out, [split_factor_f, None])
    (i,) = sch.get_loops(blk1)
    j, f_in = sch.get_loops(blk2)
    sch.bind(g, "blockIdx.y")
    sch.bind(f_out_o, "blockIdx.x")
    sch.bind(f_in, "threadIdx.x")
    sch.bind(f_out_i, "threadIdx.y")
    _, _, ax2 = sch.get_loops(read_blk)
    sch.bind(ax2, "threadIdx.x")
    mod = lower_sparse_buffer(sch.mod)
    f = tvm.build(mod["main"], target="cuda")

    def foo(X):
        X_nd = tvm.nd.from_dlpack(th_to_dlpack(X.view(-1).contiguous()))
        Y_nd = tvm.nd.from_dlpack(th_to_dlpack(Y.view(-1).contiguous()))
        f(W_nd, X_nd, Y_nd, etypes, indptr_i, indices_i, indptr_j, indices_j)
        return Y

    return foo

class RelGraphConvHomo(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels) -> None:
        super().__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat

        self.W = nn.Parameter(torch.Tensor(num_rels, out_feat, in_feat))
        self.cached_kernel = None

    def forward(self, g, feat, etypes, norm=None):
        if self.cached_kernel is None:
            self.cached_kernel = create_homogeneous_kernel(g, self.W.data, etypes, self.in_feat, self.out_feat)
        h = self.cached_kernel(feat)
        return h


class RelGraphConvHetero(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels) -> None:
        super().__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        dropout = 0.

        self.dropout = nn.Dropout(dropout)
        self.W = nn.Parameter(torch.Tensor(num_rels, out_feat, in_feat))
        self.cached_kernel = None

    def forward(self, g, feat):
        if self.cached_kernel is None:
            self.cached_kernel = create_heterogeneneous_kernel(g, self.W.data, self.in_feat, self.out_feat)
        h = self.cached_kernel(feat)
        return h


class RGCNSparseTIRHomo(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_rels):
        super(RGCNSparseTIRHomo, self).__init__()
        self.layer1 = RelGraphConvHomo(in_dim, hidden_dim, num_rels)
        self.layer2 = RelGraphConvHomo(hidden_dim, out_dim, num_rels)

    def forward(self, g, features, etypes, norm):
        x = F.relu(self.layer1(g, features, etypes, norm))
        x = self.layer2(g, x, etypes, norm)
        return x


class RGCNSparseTIRHetero(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_rels):
        super(RGCNSparseTIRHetero, self).__init__()
        self.layer1 = RelGraphConvHetero(in_dim, hidden_dim, num_rels)
        self.layer2 = RelGraphConvHetero(hidden_dim, out_dim, num_rels)

    def forward(self, g, features):
        x = F.relu(self.layer1(g, features))
        x = self.layer2(g, x)
        return x

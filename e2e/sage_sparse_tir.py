import argparse
from turtle import backward
import dgl
import dgl.function as fn
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.dlpack import to_dlpack as th_to_dlpack
from torch.utils.dlpack import from_dlpack as th_from_dlpack

from utils import Logger, get_dataset
import tvm
import tvm.testing
import tvm.tir as tir
from tvm.script import tir as T
from tvm.sparse import FormatRewriteRule, lower_sparse_buffer, lower_sparse_iter


@T.prim_func
def csrmm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    num_tiles: T.int32,
    nnz: T.int32,
    cwm: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    I = T.dense_fixed(m)
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), "int32")
    J_detach = T.dense_fixed(n)
    K1 = T.dense_fixed(num_tiles)
    K2 = T.dense_fixed(cwm)
    K3 = T.dense_fixed(32)
    A = T.match_sparse_buffer(a, (I, J), "float32")
    B = T.match_sparse_buffer(b, (J_detach, K1, K2, K3), "float32")
    C = T.match_sparse_buffer(c, (I, K1, K2, K3), "float32")
    with T.iter([I, J, K1, K2, K3], "SRSSS", "csrmm") as [i, j, k1, k2, k3]:
        with T.init():
            C[i, k1, k2, k3] = 0.0
        C[i, k1, k2, k3] = C[i, k1, k2, k3] + A[i, j] * B[j, k1, k2, k3]


@T.prim_func
def ell(
    a: T.handle,
    indptr_i: T.handle,
    indices_i: T.handle,
    indices_j: T.handle,
    m: T.int32,
    n: T.int32,
    num_rows: T.int32,
    nnz_cols: T.int32,
) -> None:
    O = T.dense_fixed(1)
    I = T.sparse_variable(O, (m, num_rows), (indptr_i, indices_i))
    J = T.sparse_fixed(I, (n, nnz_cols), indices_j)
    A = T.match_sparse_buffer(a, (O, I, J), "float32")
    T.evaluate(0)


def csr2ell_inv_index_map(o, i, j):
    return i, j


def csr2ell_index_map(i, j):
    return 0, i, j


kernels = {}
kernel_args = {}


class SpMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        X_nd = tvm.nd.from_dlpack(th_to_dlpack(X.view(-1).contiguous()))
        Y = torch.zeros_like(X)
        Y_nd = tvm.nd.from_dlpack(th_to_dlpack(Y.view(-1).contiguous()))
        f = kernels[(X.shape[-1], True)]
        args = [X_nd, Y_nd]
        args += kernel_args[True]
        f(*args)
        return Y

    @staticmethod
    def backward(ctx, dY):
        dY_nd = tvm.nd.from_dlpack(th_to_dlpack(dY.view(-1).contiguous()))
        dX = torch.zeros_like(dY)
        dX_nd = tvm.nd.from_dlpack(th_to_dlpack(dX.view(-1).contiguous()))
        # The graph we profiled are undirected.
        # f = kernels[(dY.shape[-1], False)]
        f = kernels[(dY.shape[-1], True)]
        args = [dY_nd, dX_nd]
        # args += kernel_args[False]
        args += kernel_args[True]
        f(*args)
        return dX


class SAGEConv(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(SAGEConv, self).__init__()

        self._in_src_feats, self._in_dst_feats = in_feats, in_feats
        self._out_feats = out_feats
        self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=False)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def forward(self, feat):
        r"""Compute GraphSAGE layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        rst = self.fc_self(feat) + self.fc_neigh(SpMM.apply(feat))

        return rst


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers, dropout):
        super(GraphSAGE, self).__init__()

        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        # input layer
        self.layers.append(SAGEConv(in_feats, hidden_feats))
        self.bns.append(nn.BatchNorm1d(hidden_feats))
        # hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_feats, hidden_feats))
            self.bns.append(nn.BatchNorm1d(hidden_feats))
        # output layer
        self.layers.append(SAGEConv(hidden_feats, out_feats))
        self.dropout = nn.Dropout(p=dropout)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.layers[-1](x)

        return x.log_softmax(dim=-1)


def train(dataset, model, feats, y_true, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(feats)[train_idx]
    if dataset == "ppi":
        loss = F.binary_cross_entropy_with_logits(out, y_true[train_idx])
    else:
        loss = F.nll_loss(out, y_true[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


def create_kernels(g, feat_sizes, bucket_sizes=[], column_part=1):
    print(feat_sizes)
    global kernels
    global kernel_args
    # The graph we profiled are undirected.
    for forward in [True]:  # [True, False]:
        mat = g.adj(transpose=forward, scipy_fmt="csr")
        buckets = bucket_sizes * column_part
        m = mat.shape[0]
        n = mat.shape[1]
        nnz = mat.nnz
        per_column_part_size = (n + column_part - 1) // column_part
        sub_mats = [
            mat[:, i * per_column_part_size : (i + 1) * per_column_part_size]
            for i in range(column_part)
        ]

        num_buckets = len(buckets)
        ell_n = []
        is_bucket_atomic = []

        for partition in range(column_part):
            sub_mat = sub_mats[partition]
            in_degrees = sub_mat.indptr[1:] - sub_mat.indptr[:-1]
            for i, bucket_size in enumerate(bucket_sizes[:-1]):
                last_bucket_size = 0 if i == 0 else bucket_sizes[i - 1]
                ell_n.append(
                    int(
                        (
                            (in_degrees > last_bucket_size)
                            & (in_degrees <= bucket_size)
                        ).sum()
                    )
                )
                if column_part == 1:
                    is_bucket_atomic.append(False)
                else:
                    is_bucket_atomic.append(True)
            sub_indegrees = in_degrees[in_degrees > bucket_sizes[-2]]
            ell_n.append(
                int(((sub_indegrees + bucket_sizes[-1] - 1) // bucket_sizes[-1]).sum())
            )
            is_bucket_atomic.append(True)
        print(ell_n)
        print(nnz / (sum([b * sub_nnz for b, sub_nnz in zip(buckets, ell_n)])))

        ell_rows = []
        ell_indices = []
        ell_a = []

        for partition in range(column_part):
            sub_mat = sub_mats[partition]
            in_degrees = sub_mat.indptr[1:] - sub_mat.indptr[:-1]

            for i, bucket_size in enumerate(bucket_sizes[:-1]):
                last_bucket_size = 0 if i == 0 else bucket_sizes[i - 1]
                ell_rows.append(
                    (
                        (in_degrees > last_bucket_size) & (in_degrees <= bucket_size)
                    ).nonzero()[0]
                )
            ell_rows.append((in_degrees > bucket_sizes[-2]).nonzero()[0])

            for i, bucket_size in enumerate(bucket_sizes[:-1]):
                indices = np.zeros(
                    (ell_n[partition * len(bucket_sizes) + i], bucket_size),
                    dtype=np.int32,
                )
                a = np.zeros(
                    (ell_n[partition * len(bucket_sizes) + i], bucket_size),
                    dtype=np.float32,
                )
                for j, row_id in enumerate(ell_rows[partition * len(bucket_sizes) + i]):
                    row = sub_mat[row_id]
                    indices[j, : row.nnz] = (
                        row.indices + partition * per_column_part_size
                    )
                    a[j, : row.nnz] = row.data
                ell_indices.append(indices)
                ell_a.append(a)

            # split rows for the last bucket
            indices = np.zeros(
                (ell_n[(partition + 1) * len(bucket_sizes) - 1], bucket_sizes[-1]),
                dtype=np.int32,
            )
            a = np.zeros(
                (ell_n[(partition + 1) * len(bucket_sizes) - 1], bucket_sizes[-1]),
                dtype=np.float32,
            )
            new_rows = np.zeros(
                (ell_n[(partition + 1) * len(bucket_sizes) - 1],), dtype=np.int32
            )
            bucket_size = bucket_sizes[-1]
            i = 0
            for row_id in ell_rows[-1]:
                row = sub_mat[row_id]
                for start_offset in range(0, row.nnz, bucket_size):
                    if start_offset + bucket_size >= row.nnz:
                        # last bucket
                        indices[i, : row.nnz - start_offset] = (
                            row.indices[start_offset:]
                            + partition * per_column_part_size
                        )
                        a[i, : row.nnz - start_offset] = row.data[start_offset:]
                    else:
                        indices[i] = (
                            row.indices[start_offset : start_offset + bucket_size]
                            + partition * per_column_part_size
                        )
                        a[i].fill(1)
                    new_rows[i] = row_id
                    i += 1

            ell_indices.append(indices)
            ell_a.append(a)
            ell_rows[-1] = new_rows

        # prepare nd array
        ell_indices_i_nd = []
        ell_a_nd = []
        ell_indices_j_nd = []
        for i in range(num_buckets):
            ell_indices_i_nd.append(
                tvm.nd.array(ell_rows[i].astype("int32"), device=tvm.cuda(0))
            )
            ell_a_nd.append(
                tvm.nd.array(ell_a[i].reshape(-1).astype("float32"), device=tvm.cuda(0))
            )
            ell_indices_j_nd.append(
                tvm.nd.array(
                    ell_indices[i].reshape(-1).astype("int32"), device=tvm.cuda(0)
                )
            )

        args = []
        for i in range(num_buckets):
            args += [ell_a_nd[i], ell_indices_i_nd[i], ell_indices_j_nd[i]]

        kernel_args[forward] = args

        for feat_size in feat_sizes:
            if feat_size <= 32:
                cwm = 1
            elif feat_size <= 128:
                cwm = 2
            else:
                cwm = 4

            # rewrite csrmm
            nnz_cols_symbol = ell.params[-1]
            rewrites = []
            for i, bucket_size in enumerate(buckets):
                rewrites.append(
                    FormatRewriteRule(
                        str(i),
                        ell.specialize({nnz_cols_symbol: bucket_size}),
                        ["A"],
                        ["I", "J"],
                        ["O", "I", "J"],
                        {"I": ["O", "I"], "J": ["J"]},
                        csr2ell_index_map,
                        csr2ell_inv_index_map,
                    )
                )
            mod = tvm.IRModule.from_expr(csrmm)
            mod = tvm.tir.transform.SparseFormatRewrite(rewrites)(mod)
            mod = tvm.tir.transform.RemovePreprocess()(mod)

            # specialize
            params = mod["main"].params
            param_map = {
                params[5]: m,  # m
                params[6]: n,  # n
                params[7]: feat_size // cwm // 32,  # num_tiles,
                params[8]: nnz,  # nnz
                params[9]: cwm,  # cwm
            }
            for i, bucket_size in enumerate(buckets):
                param_map[params[10 + 7 * i + 4]] = m
                param_map[params[10 + 7 * i + 5]] = n
                param_map[params[10 + 7 * i + 6]] = ell_n[i]

            mod["main"] = (
                mod["main"].specialize(param_map).with_attr("horizontal_fuse", True)
            )

            # schedule
            sch = tvm.tir.Schedule(mod)
            for sp_iter_name in ["csrmm_{}".format(i) for i in range(num_buckets)]:
                sp_iteration = sch.get_sparse_iteration(sp_iter_name)
                o, i, j, k1, k2, k3 = sch.get_sp_iters(sp_iteration)
                sch.sparse_fuse(sp_iteration, [o, i])

            mod = sch.mod
            mod = lower_sparse_iter(mod)
            sch = tvm.tir.Schedule(mod)
            for i, bucket_size in enumerate(buckets):
                is_atomic = is_bucket_atomic[i]
                bucket_size = buckets[i]
                blk = sch.get_block("csrmm_{}0".format(i))
                i, j, foo, foi, fi = sch.get_loops(blk)
                sch.reorder(foo, fi, j, foi)
                if is_atomic:
                    sch.annotate(blk, "atomic", True)
                    write_blk = sch.cache_write(blk, 0, "local")
                    sch.reverse_compute_at(write_blk, fi, True)
                    # sch.unroll(sch.get_loops(write_blk)[-2])
                sch.bind(fi, "threadIdx.x")
                sch.bind(foo, "blockIdx.y")
                sch.unroll(foi)
                sch.unroll(j)
                io, ioi, ii = sch.split(i, [None, bucket_sizes[-1] // bucket_size, 8])
                sch.bind(io, "blockIdx.x")
                sch.bind(ii, "threadIdx.y")
                init_blk = sch.decompose_reduction(blk, fi)
                ax0, ax1 = sch.get_loops(init_blk)[-2:]
                sch.bind(ax0, "threadIdx.x")
                sch.unroll(ax1)

            mod = lower_sparse_buffer(sch.mod)
            mod = tvm.tir.transform.RemoveUnusedArgs()(mod)
            f = tvm.build(mod, target="cuda")

            kernels[(feat_size, forward)] = f


def pad_length(x: int):
    if x <= 32:
        return 32
    if x <= 64:
        return 64
    ret = 128
    while ret < x:
        ret = ret + 128
    return ret


bucketing_config = {
    "arxiv": [1, 2, 4, 8, 16, 32],
    "proteins": [1, 2, 4, 8, 16, 32, 64, 128, 256],
    "pubmed": [1, 2, 4, 8, 16, 32],
    "ppi": [1, 2, 4, 8, 16, 32],
    "cora": [1, 2, 4],
    "citeseer": [1, 2, 4],
    "reddit": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
}

col_part = {
    "arxiv": 1,
    "proteins": 8,
    "pubmed": 1,
    "cora": 1,
    "citeseer": 1,
    "ppi": 8,
    "reddit": 8,
}


def main():
    parser = argparse.ArgumentParser(description="OGBN-Arxiv (GraphSAGE Full-Batch)")
    parser.add_argument("--dataset", type=str, default="arxiv")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    g, feats, labels, split_idx, num_classes = get_dataset(args.dataset)
    # pad
    feats_ = torch.zeros([feats.shape[0], pad_length(feats.shape[1])])
    feats_[:, : feats.shape[1]] = feats
    feats = feats_
    if args.dataset == "ppi":
        labels_ = torch.zeros([labels.shape[0], pad_length(num_classes)]).to(labels)
        labels_[:, : labels.shape[1]] = labels
        labels = labels_
    g = dgl.to_bidirected(g)
    g = g.int().to(device)
    feats, labels = feats.to(device), labels.to(device)
    train_idx = split_idx["train"].to(device)
    num_classes = pad_length(num_classes)

    create_kernels(
        g,
        [feats.shape[-1], args.hidden_channels, num_classes],
        bucketing_config[args.dataset],
        col_part[args.dataset],
    )

    model = GraphSAGE(
        in_feats=feats.size(-1),
        hidden_feats=args.hidden_channels,
        out_feats=num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    dur = []
    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            t0 = time.time()
            loss = train(args.dataset, model, feats, labels, train_idx, optimizer)
            if epoch >= 3:
                dur.append(time.time() - t0)
                print("Training time/epoch {}".format(np.mean(dur)))


if __name__ == "__main__":
    main()

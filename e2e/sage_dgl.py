import argparse
import dgl
import dgl.function as fn
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.utils import expand_as_pair
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

from utils import get_dataset


class SAGEConv(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(SAGEConv, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=False)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def forward(self, graph, feat):
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
        graph = graph.local_var()

        if isinstance(feat, tuple):
            feat_src, feat_dst = feat
        else:
            feat_src = feat_dst = feat

        h_self = feat_dst

        graph.srcdata["h"] = feat_src
        graph.update_all(fn.copy_src("h", "m"), fn.mean("m", "neigh"))
        h_neigh = graph.dstdata["neigh"]
        rst = self.fc_self(h_self) + self.fc_neigh(h_neigh)

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

    def forward(self, g, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(g, x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.layers[-1](g, x)

        return x.log_softmax(dim=-1)


def train(dataset, model, g, feats, y_true, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(g, feats)[train_idx]
    if dataset == "ppi":
        loss = F.binary_cross_entropy_with_logits(out, y_true[train_idx])
    else:
        loss = F.nll_loss(out, y_true[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


def main():
    parser = argparse.ArgumentParser(description="GraphSAGE Full-Batch")
    parser.add_argument("-d", "--dataset", type=str, default="arxiv")
    args = parser.parse_args()

    device = torch.device(0)

    g, feats, labels, split_idx, num_classes = get_dataset(args.dataset)
    g = dgl.to_bidirected(g)
    g = g.int().to(device)
    feats, labels = feats.to(device), labels.to(device)
    train_idx = split_idx["train"].to(device)

    model = GraphSAGE(
        in_feats=feats.size(-1),
        hidden_feats=128,
        out_feats=num_classes,
        num_layers=3,
        dropout=0.5,
    ).to(device)

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    warmup = 20
    active = 200
   
    for _ in range(warmup):
        loss = train(args.dataset, model, g, feats, labels, train_idx, optimizer)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(active):
        loss = train(args.dataset, model, g, feats, labels, train_idx, optimizer)
    end_event.record()
    torch.cuda.synchronize()
    dur = start_event.elapsed_time(end_event) / active
    print("Training time: {} ms/epoch".format(dur))


if __name__ == "__main__":
    main()

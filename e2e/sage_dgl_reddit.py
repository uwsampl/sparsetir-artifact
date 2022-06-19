"""
Inductive Representation Learning on Large Graphs
Paper: http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf
Code: https://github.com/williamleif/graphsage-simple
Simple reference implementation of GraphSAGE.
"""
import argparse
import dgl
import dgl.function as fn
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import load_data
from dgl.utils import expand_as_pair

from utils import Logger

class SAGEConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 aggr,
                 feat_drop=0.,
                 activation=None):
        super(SAGEConv, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggr = aggr
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=False)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
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
            feat_src = self.feat_drop(feat[0])
            feat_dst = self.feat_drop(feat[1])
        else:
            feat_src = feat_dst = self.feat_drop(feat)

        h_self = feat_dst

        graph.srcdata['h'] = feat_src
        if self._aggr == 'sum':
            graph.update_all(fn.copy_src('h', 'm'), fn.sum('m', 'neigh'))
        elif self._aggr == 'mean':
            graph.update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neigh'))
        else:
            return ValueError("Expect aggregation to be 'sum' or 'mean', got {}".format(self._aggr))
        # h_neigh = graph.dstdata['neigh']
        # rst = self.fc_self(h_self) + self.fc_neigh(h_neigh)
        rst = self.fc_self(h_self)

        # activation
        if self.activation is not None:
            rst = self.activation(rst)
        return rst

class GraphSAGE(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 aggr,
                 activation=F.relu,
                 dropout=0.):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.g = g
        self.layers.append(SAGEConv(in_feats, n_hidden, aggr, activation=activation))
        self.layers.append(SAGEConv(n_hidden, n_classes, aggr, feat_drop=dropout, activation=None))

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, features):
        h = features
        for layer in self.layers:
            h = layer(self.g, h)
        return h

def calc_acc(logits, labels, mask):
    logits = logits[mask]
    labels = labels[mask]
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

def evaluate(model, features, labels, train_mask, val_mask, test_mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        train_acc = calc_acc(logits, labels, train_mask)
        val_acc = calc_acc(logits, labels, val_mask)
        test_acc = calc_acc(logits, labels, test_mask)
        return train_acc, val_acc, test_acc

def main():    
    parser = argparse.ArgumentParser(description='GraphSAGE')
    parser.add_argument("--dataset", type=str, default='reddit')
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--aggr", type=str, choices=['sum', 'mean'], default='mean',
                        help='Aggregation for messages')
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--eval", action='store_true',
                        help='If not set, we will only do the training part.')
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    print(args)
    # load and preprocess dataset
    data = load_data(args)
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    if hasattr(torch, 'BoolTensor'):
        train_mask = torch.BoolTensor(data.train_mask)
        val_mask = torch.BoolTensor(data.val_mask)
        test_mask = torch.BoolTensor(data.test_mask)
    else:
        train_mask = torch.ByteTensor(data.train_mask)
        val_mask = torch.ByteTensor(data.val_mask)
        test_mask = torch.ByteTensor(data.test_mask)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    # Remove duplicate edges
    # In PyG, this is a default pre-processing step for Reddit, see
    # https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/datasets/reddit.py#L58
    #g = data.graph
    g = data[0]
    g = g.int().to(device)

    # create GraphSAGE model
    model = GraphSAGE(g,
                      in_feats,
                      args.n_hidden,
                      n_classes,
                      args.aggr,
                      F.relu,
                      args.dropout).to(device)

    loss_fcn = nn.CrossEntropyLoss()

    logger = Logger(args.runs, args)
    dur = []
    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for epoch in range(args.epochs):
            model.train()
            if epoch >= 3:
                t0 = time.time()
            # forward
            logits = model(features)
            loss = loss_fcn(logits[train_mask], labels[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)
                print('Training time/epoch {}'.format(np.mean(dur)))

            if not args.eval:
                continue

            train_acc, val_acc, test_acc = evaluate(model, features, labels, train_mask, val_mask, test_mask)
            logger.add_result(run, (train_acc, val_acc, test_acc))

            print("Run {:02d} | Epoch {:05d} | Loss {:.4f} | Train {:.4f} | Val {:.4f} | Test {:.4f}".format(run, epoch, loss.item(), train_acc, val_acc, test_acc))

        if args.eval:
            logger.print_statistics(run)

    if args.eval:
        logger.print_statistics()

if __name__ == '__main__':
    main()
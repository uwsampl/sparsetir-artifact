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

from utils import Logger

class SAGEConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats):
        super(SAGEConv, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=False)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
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
            feat_src, feat_dst = feat
        else:
            feat_src = feat_dst = feat

        h_self = feat_dst

        graph.srcdata['h'] = feat_src
        graph.update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neigh'))
        h_neigh = graph.dstdata['neigh']
        rst = self.fc_self(h_self) + self.fc_neigh(h_neigh)

        return rst

class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 hidden_feats,
                 out_feats,
                 num_layers,
                 dropout):
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

def train(model, g, feats, y_true, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(g, feats)[train_idx]
    loss = F.nll_loss(out, y_true.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def test(model, g, feats, y_true, split_idx, evaluator):
    model.eval()

    out = model(g, feats)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc

def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GraphSAGE Full-Batch)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument("--eval", action='store_true',
                        help='If not set, we will only do the training part.')
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = DglNodePropPredDataset(name='ogbn-arxiv')
    split_idx = dataset.get_idx_split()

    g, labels = dataset[0]
    feats = g.ndata['feat']
    g = dgl.to_bidirected(g)
    g = g.int().to(device)
    feats, labels = feats.to(device), labels.to(device)
    train_idx = split_idx['train'].to(device)
    print(feats.shape[-1], dataset.num_classes)

    model = GraphSAGE(in_feats=feats.size(-1),
                      hidden_feats=args.hidden_channels,
                      out_feats=dataset.num_classes,
                      num_layers=args.num_layers,
                      dropout=args.dropout).to(device)

    evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(args.runs, args)

    dur = []
    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            t0 = time.time()
            loss = train(model, g, feats, labels, train_idx, optimizer)
            if epoch >= 3:
                dur.append(time.time() - t0)
                print('Training time/epoch {}'.format(np.mean(dur)))
            if not args.eval:
                continue

            result = test(model, g, feats, labels, split_idx, evaluator)
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')

        if args.eval:
            logger.print_statistics(run)
    if args.eval:
        logger.print_statistics()


if __name__ == '__main__':
    main()
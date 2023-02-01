from numpy import product
import torch
import dgl
from ogb.nodeproppred import DglNodePropPredDataset


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')


def get_dataset(dataset_name: str):
    """ Return DGLGraph object from dataset name.

    Parameters
    ----------
    dataset_name: str
        The dataset name.

    Returns
    -------
    DGLGraph
        The graph object.
    """
    home = os.path.expanduser("~")
    ogb_path = os.path.join(home, "ogb")
    if not os.path.exists(ogb_path):
        os.makedirs(ogb_path)
    if dataset_name == 'pubmed':
        pubmed = dgl.data.PubmedGraphDataset()
        g = pubmed[0].int()
        split_idx = {
            "train": g.ndata['train_mask'],
            "valid": g.ndata['val_mask'],
            "test": g.ndata['test_mask']
        }
        return g, g.ndata['feat'], g.ndata['label'], split_idx, pubmed.num_labels
    elif dataset_name == 'cora':
        cora = dgl.data.CoraGraphDataset()
        g = cora[0].int()
        split_idx = {
            "train": g.ndata['train_mask'],
            "valid": g.ndata['val_mask'],
            "test": g.ndata['test_mask']
        }
        return g, g.ndata['feat'], g.ndata['label'], split_idx, cora.num_labels
    elif dataset_name == 'citeseer':
        citeseer = dgl.data.CiteseerGraphDataset()
        g = citeseer[0].int()
        split_idx = {
            "train": g.ndata['train_mask'],
            "valid": g.ndata['val_mask'],
            "test": g.ndata['test_mask']
        }
        return g, g.ndata['feat'], g.ndata['label'], split_idx, citeseer.num_labels
    elif dataset_name == 'arxiv':
        arxiv = DglNodePropPredDataset(name='ogbn-arxiv', root=ogb_path)
        g = arxiv[0][0].int()
        return g, g.ndata['feat'], arxiv.labels.squeeze(-1), arxiv.get_idx_split(), arxiv.num_classes
    elif dataset_name == 'proteins':
        proteins = DglNodePropPredDataset(name='ogbn-proteins', root=ogb_path)
        g = proteins[0][0].int()
        return g, g.ndata['feat'], proteins.labels.squeeze(-1), proteins.get_idx_split(), proteins.num_classes
    elif dataset_name == 'products':
        products = DglNodePropPredDataset(name='ogbn-products', root=ogb_path)
        g = products[0][0].int()
        return g, g.ndata['feat'], products.labels.squeeze(-1), products.get_idx_split(), products.num_classes
    elif dataset_name == 'ppi':
        ppi = dgl.data.PPIDataset()
        g = dgl.batch(ppi).int()
        split_idx = {
            "train": torch.ones(g.num_nodes()).bool(),
            "valid": torch.zeros(g.num_nodes()).bool(),
            "test": torch.zeros(g.num_nodes()).bool() 
        }
        return g, g.ndata['feat'], g.ndata['label'], split_idx, ppi.num_labels
    elif dataset_name == 'reddit':
        reddit = dgl.data.RedditDataset()
        g = reddit[0].int()
        split_idx = {
            "train": g.ndata['train_mask'],
            "valid": g.ndata['val_mask'],
            "test": g.ndata['test_mask']
        }
        return g, g.ndata['feat'], g.ndata['label'], split_idx, reddit.num_labels
    else:
        pass
    

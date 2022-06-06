import dgl
from ogb.nodeproppred import DglNodePropPredDataset

__all__ = ['get_graph', 'th_time_op']


def get_graph(dataset_name: str):
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
    if dataset_name == 'pubmed':
        pubmed = dgl.data.PubmedGraphDataset()
        return pubmed[0].int()
    elif dataset_name == 'cora':
        cora = dgl.data.CoraGraphDataset()
        return cora[0].int()
    elif dataset_name == 'citeseer':
        citeseer = dgl.data.CiteseerGraphDataset()
        return citeseer[0].int()
    elif dataset_name == 'arxiv':
        arxiv = DglNodePropPredDataset(name='ogbn-arxiv')
        return arxiv[0][0].int()
    elif dataset_name == 'proteins':
        arxiv = DglNodePropPredDataset(name='ogbn-proteins')
        return arxiv[0][0].int()
    elif dataset_name == 'products':
        products = DglNodePropPredDataset(name='ogbn-products')
        return products[0][0].int()
    elif dataset_name == 'ppi':
        ppi = dgl.data.PPIDataset()
        g = dgl.batch(ppi).int()
        return g
    elif dataset_name == 'reddit':
        reddit = dgl.data.RedditDataset()
        return reddit[0].int()
    else:
        pass
    

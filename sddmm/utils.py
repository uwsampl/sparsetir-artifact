import dgl
import os
from ogb.nodeproppred import DglNodePropPredDataset

__all__ = ["get_dataset"]


def get_dataset(dataset_name: str):
    """Return DGLGraph object from dataset name.

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
    if dataset_name == "pubmed":
        pubmed = dgl.data.PubmedGraphDataset()
        g = pubmed[0]
    elif dataset_name == "cora":
        cora = dgl.data.CoraGraphDataset()
        g = cora[0]
    elif dataset_name == "citeseer":
        citeseer = dgl.data.CiteseerGraphDataset()
        g = citeseer[0]
    elif dataset_name == "arxiv":
        arxiv = DglNodePropPredDataset(name="ogbn-arxiv", root=ogb_path)
        g = arxiv[0][0]
    elif dataset_name == "proteins":
        proteins = DglNodePropPredDataset(name="ogbn-proteins", root=ogb_path)
        g = proteins[0][0]
    elif dataset_name == "products":
        products = DglNodePropPredDataset(name="ogbn-products", root=ogb_path)
        g = products[0][0]
    elif dataset_name == "ppi":
        ppi = dgl.data.PPIDataset()
        g = dgl.batch(ppi)
    elif dataset_name == "reddit":
        reddit = dgl.data.RedditDataset()
        g = reddit[0]
    else:
        raise KeyError("Unknown dataset {}.".format(dataset_name))
    g = dgl.graph(g.edges("uv", "srcdst"), num_nodes=g.num_nodes())
    return g.int()


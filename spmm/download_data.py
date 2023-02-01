import dgl
import os
from ogb.nodeproppred import DglNodePropPredDataset


def download_data():
    home = os.path.expanduser("~")
    ogb_path = os.path.join(home, "ogb")
    if not os.path.exists(ogb_path):
        os.makedirs(ogb_path)
    pubmed = dgl.data.PubmedGraphDataset()
    cora = dgl.data.CoraGraphDataset()
    citeseer = dgl.data.CiteseerGraphDataset()
    arxiv = DglNodePropPredDataset(name="ogbn-arxiv", root=ogb_path)
    proteins = DglNodePropPredDataset(name="ogbn-proteins", root=ogb_path)
    products = DglNodePropPredDataset(name="ogbn-products", root=ogb_path)
    ppi = dgl.data.PPIDataset()
    reddit = dgl.data.RedditDataset()


if __name__ == "__main__":
    download_data()

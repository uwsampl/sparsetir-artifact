import dgl
import os
from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset
from ogb.linkproppred import DglLinkPropPredDataset


def download_data():
    home = os.path.expanduser("~")
    ogb_path = os.path.join(home, "ogb")
    if not os.path.exists(ogb_path):
        os.makedirs(ogb_path)
    aifb = AIFBDataset()
    mutag = MUTAGDataset()
    bgs = BGSDataset()
    am = AMDataset()
    biokg = DglLinkPropPredDataset(name="ogbl-biokg", root=ogb_path)

if __name__ == "__main__":
    download_data()


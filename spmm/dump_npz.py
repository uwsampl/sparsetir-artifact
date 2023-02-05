from ogb.nodeproppred import DglNodePropPredDataset
import dgl
import numpy as np
import os
from utils import get_dataset

if not os.path.exists("data"):
    os.mkdir("data")

for dataset_name in ["cora", "citeseer", "arxiv", "pubmed", "ppi", "reddit", "proteins"]:
    print("dumping dataset {}".format(dataset_name))
    g = get_dataset(dataset_name)
    indptr, indices, _ = g.adj_sparse("csc")
    shp = np.array(
        [g.num_dst_nodes(), g.num_src_nodes(), g.num_edges()], dtype=np.int32
    )
    np.savez(
        "data/{}.npz".format(dataset_name),
        shape=shp,
        indptr=indptr.int().numpy(),
        indices=indices.int().numpy(),
    )

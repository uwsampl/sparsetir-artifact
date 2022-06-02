from ogb.nodeproppred import DglNodePropPredDataset
import dgl
import numpy as np
from utils import get_graph

arxiv = DglNodePropPredDataset(name="ogbn-arxiv")
g = arxiv[0][0].int() # [1, 2, 4, 8, 16, 32]
mat = g.adj(transpose=True, scipy_fmt='csr')

for dataset_name in ["arxiv", "pubmed", "ppi", "reddit", "proteins"]:
    print("dumping dataset {}".format(dataset_name))
    g = get_graph(dataset_name)
    indptr, indices, _ = g.adj_sparse('csc')
    shp = np.array([g.num_dst_nodes(), g.num_src_nodes(), g.num_edges()], dtype=np.int32)
    np.savez("../{}.npz".format(dataset_name),
            shape=shp,
            indptr=indptr.int().numpy(),
            indices=indices.int().numpy())

from ogb.nodeproppred import DglNodePropPredDataset
import dgl
from scipy.io import mmwrite

#arxiv = DglNodePropPredDataset(name="ogbn-arxiv")
#g = arxiv[0][0].int() # [1, 2, 4, 8, 16, 32]
#mat = g.adj(transpose=True, scipy_fmt='csr')
#mmwrite('../arxiv.mtx', mat)
proteins = DglNodePropPredDataset(name="ogbn-proteins")
g = proteins[0][0].int()
mat = g.adj(transpose=True, scipy_fmt='csr')
mmwrite('../proteins.mtx', mat)
#pubmed = dgl.data.PubmedGraphDataset()
#g = pubmed[0] # [1, 8, 16]
#mat = g.adj(transpose=True, scipy_fmt='csr')
#mmwrite('../pubmed.mtx', mat)
#ppi = dgl.data.PPIDataset()
#g = dgl.batch(ppi) # [4, 8, 16, 32]
#mat = g.adj(transpose=True, scipy_fmt='csr')
#mmwrite('../ppi.mtx', mat)
#reddit = dgl.data.RedditDataset()
#g = reddit[0] # [64, 128, 256, 512]
#mat = g.adj(transpose=True, scipy_fmt='csr')
#mmwrite('../reddit.mtx', mat)

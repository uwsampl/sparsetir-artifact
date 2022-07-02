from utils import create_pixelfly, create_longformer

from torch.profiler import profile, ProfilerActivity, schedule 
import dgl
import torch

def test_csr_sddmm():
    #csr = create_pixelfly(1, 4096 // 16, fmt='csr', block_size=16)
    csr = create_longformer(1, 4096 // 16, 256 // 16, fmt='csr', block_size=16)
    g = dgl.from_scipy(csr).int()
    g = g.to(0)
    a_gpu = torch.rand(4096, 64).to(0)
    b_gpu = torch.rand(4096, 64).to(0)
    with profile(activities=[ProfilerActivity.CUDA], schedule=schedule(wait=0, warmup=10, active=100)) as prof:
        for _ in range(100):
            c_gpu = dgl.ops.u_dot_v(g, a_gpu, b_gpu)
            prof.step()
 
    print(prof.key_averages())


if __name__ == "__main__":
    test_csr_sddmm()


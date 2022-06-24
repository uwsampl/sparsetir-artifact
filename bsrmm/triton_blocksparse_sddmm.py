import pytest
import torch

import triton
from torch.profiler import profile, ProfilerActivity, schedule 
from utils import create_pixelfly


class TorchOpTimer(object):
    def __enter__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()
        return self

    def __exit__(self, type, value, traceback):
        self.end_event.record()
        torch.cuda.synchronize()  # Wait for the events to be recorded!
        self.time = self.start_event.elapsed_time(self.end_event)


# "sdd" for block-sddmm, and "dsd" for block-spmm
# @pytest.mark.parametrize("MODE", ["sdd", "dds", "dsd"])
# @pytest.mark.parametrize("TRANS_A", [False, True])
# @pytest.mark.parametrize("TRANS_B", [False, True])
# @pytest.mark.parametrize("BLOCK", [16, 32, 64])
# @pytest.mark.parametrize("DTYPE", [torch.float32, torch.float16])
def test_matmul(MODE, TRANS_A, TRANS_B, BLOCK, DTYPE, Z=1, H=1, M=512, N=384, K=256):
    seed = 0
    torch.manual_seed(seed)
    is_sdd = MODE == "sdd"
    is_dsd = MODE == "dsd"
    is_dds = MODE == "dds"
    do_sparsify = lambda x: triton.testing.sparsify_tensor(x, layout, BLOCK)
    do_mask = lambda x: triton.testing.mask_tensor(x, layout, BLOCK)
    # create inputs
    # create op
    a_shape = (Z, H, K, M) if TRANS_A else (Z, H, M, K)
    b_shape = (Z, H, N, K) if TRANS_B else (Z, H, K, N)
    shape = {
        "sdd": (M, N),
        "dsd": (a_shape[2], a_shape[3]),
        "dds": (b_shape[2], b_shape[3]),
    }[MODE]
    layout = create_pixelfly(H, M // BLOCK, fmt='mask')

    # create data
    a_ref, a_tri = triton.testing.make_pair(a_shape, dtype=DTYPE, alpha=.1)
    b_ref, b_tri = triton.testing.make_pair(b_shape, dtype=DTYPE, alpha=.1)
    # compute [torch]
    a_ref = do_mask(a_ref) if is_dsd else a_ref
    b_ref = do_mask(b_ref) if is_dds else b_ref
    c_ref = torch.matmul(a_ref.transpose(2, 3) if TRANS_A else a_ref,
                         b_ref.transpose(2, 3) if TRANS_B else b_ref)
    c_ref = do_sparsify(c_ref) if is_sdd else c_ref
    # triton result
    a_tri = do_sparsify(a_tri) if is_dsd else a_tri
    b_tri = do_sparsify(b_tri) if is_dds else b_tri
    # print(a_tri)
    op = triton.ops.blocksparse.matmul(layout, BLOCK, MODE, trans_a=TRANS_A, trans_b=TRANS_B, device="cuda")
    c_tri = triton.testing.catch_oor(lambda: op(a_tri, b_tri), pytest)
    # compare
    triton.testing.assert_almost_equal(c_ref, c_tri)

    print("triton-block-sddmm\tM\t{}\tN\t{}\tK\t{}\tblock\t{}\tdtype\t{}\ttrans_A\t{}\ttrans_B\t{}".format(M, N, K, BLOCK, DTYPE, TRANS_A, TRANS_B))

    with profile(activities=[ProfilerActivity.CUDA], schedule=schedule(wait=0, warmup=10, active=100)) as prof:
        for _ in range(100):
            op(a_tri, b_tri)
            prof.step()
    print(prof.key_averages())


if __name__ == "__main__":
    test_matmul("sdd", False, True, 16, torch.float16, Z=1, H=1, M=4096, N=4096, K=768)




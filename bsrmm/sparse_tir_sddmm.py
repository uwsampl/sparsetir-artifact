import tvm
import tvm.testing
from tvm import tir
from tvm.script import tir as T
from tvm.ir import IRModule
from tvm.sparse import lower_sparse_iter, lower_sparse_buffer
import numpy as np
import scipy.sparse as sp
from utils import create_pixelfly, create_longformer


@T.prim_func
def bsrsddmm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    nb: T.int32,
    mb: T.int32,
    nnzb: T.int32,
    blk: T.int32,
    feat_size: T.int32,
    num_heads: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    I = T.dense_fixed(nb)
    J = T.sparse_variable(I, (mb, nnzb), (indptr, indices), "int32")
    J_detach = T.dense_fixed(mb)
    BI = T.dense_fixed(blk)
    BJ = T.dense_fixed(blk)
    H = T.dense_fixed(num_heads)
    F = T.dense_fixed(feat_size)
    A = T.match_sparse_buffer(a, (H, I, BI, F), "float16")
    B = T.match_sparse_buffer(b, (H, J_detach, F, BJ), "float16")
    C = T.match_sparse_buffer(c, (H, I, J, BI, BJ), "float16")

    with T.iter([H, I, J, BI, BJ, F], "SSSSSR", "sddmm") as [
        h,
        i,
        j,
        bi,
        bj,
        f,
    ]:
        with T.init():
            C[h, i, j, bi, bj] = T.float16(0)
        C[h, i, j, bi, bj] = C[h, i, j, bi, bj] + A[h, i, bi, f] * B[h, j, f, bj]


@T.prim_func
def wmma_sync_desc(a_frag: T.handle, b_frag: T.handle, c_frag: T.handle) -> None:
    A_frag = T.match_buffer(
        a_frag, (16, 16), "float16", align=128, offset_factor=1, scope="wmma.matrix_a"
    )
    B_frag = T.match_buffer(
        b_frag, (16, 16), "float16", align=128, offset_factor=1, scope="wmma.matrix_b"
    )
    C_frag = T.match_buffer(
        c_frag, (16, 16), "float16", align=128, offset_factor=1, scope="wmma.accumulator"
    )

    with T.block("root"):
        for i, j, k in T.grid(16, 16, 16):
            with T.block("update"):
                vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
                T.block_attr({"sparse": True})
                C_frag[vii, vjj] = C_frag[vii, vjj] + A_frag[vii, vkk] * B_frag[vkk, vjj]
                


@T.prim_func
def wmma_sync_impl(a_frag: T.handle, b_frag: T.handle, c_frag: T.handle) -> None:
    A_frag = T.match_buffer(
        a_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_a"
    )
    B_frag = T.match_buffer(
        b_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_b"
    )
    C_frag = T.match_buffer(
        c_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.accumulator"
    )

    with T.block("root"):
        T.reads(
            [
                C_frag[0:16, 0:16],
                A_frag[0:16, 0:16],
                B_frag[0:16, 0:16],
            ]
        )
        T.writes(C_frag[0:16, 0:16])
        for tx in T.thread_binding(0, 32, "threadIdx.x"):
            T.evaluate(
                T.tvm_mma_sync(
                    C_frag.data,
                    C_frag.elem_offset // 256 + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                    A_frag.data,
                    A_frag.elem_offset // 256 + T.floordiv(T.floormod(A_frag.elem_offset, 256), 16),
                    B_frag.data,
                    B_frag.elem_offset // 256 + T.floordiv(T.floormod(B_frag.elem_offset, 256), 16),
                    C_frag.data,
                    C_frag.elem_offset // 256 + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                    dtype="handle",
                )
            )


@T.prim_func
def wmma_load_a_desc(a: T.handle, a_frag: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float16", align=128, offset_factor=16, scope="global")
    A_frag = T.match_buffer(
        a_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_a"
    )

    with T.block("root"):
        T.reads(A[0:16, 0:16])
        T.writes(A_frag[0:16, 0:16])
        for i, j in T.grid(16, 16):
            with T.block("load"):
                vii, vjj = T.axis.remap("SS", [i, j])
                A_frag[vii, vjj] = A[vii, vjj]


@T.prim_func
def wmma_load_a_impl(a: T.handle, a_frag: T.handle) -> None:
    s0 = T.var("int32")
    s1 = T.var("int32")
    A = T.match_buffer(
        a, (16, 16), "float16", align=128, offset_factor=16, scope="global", strides=[s0, s1]
    )
    A_frag = T.match_buffer(
        a_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_a"
    )

    with T.block("root"):
        T.reads(A[0:16, 0:16])
        T.writes(A_frag[0:16, 0:16])
        for tx in T.thread_binding(0, 32, "threadIdx.x"):
            T.evaluate(
                T.tvm_load_matrix_sync(
                    A_frag.data,
                    16,
                    16,
                    16,
                    A_frag.elem_offset // 256 + T.floordiv(T.floormod(A_frag.elem_offset, 256), 16),
                    A.access_ptr("r"),
                    A.strides[0],
                    "row_major",
                    dtype="handle",
                )
            )


@T.prim_func
def wmma_load_b_desc(b: T.handle, b_frag: T.handle) -> None:
    B = T.match_buffer(b, (16, 16), "float16", align=128, offset_factor=16, scope="global")
    B_frag = T.match_buffer(
        b_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_b"
    )
    with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("load"):
                vii, vjj = T.axis.remap("SS", [i, j])
                B_frag[vii, vjj] = B[vii, vjj]


@T.prim_func
def wmma_load_b_impl(b: T.handle, b_frag: T.handle) -> None:
    s0 = T.var("int32")
    s1 = T.var("int32")
    B = T.match_buffer(
        b, (16, 16), "float16", align=128, offset_factor=16, scope="global", strides=[s0, s1]
    )
    B_frag = T.match_buffer(
        b_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_b"
    )
    with T.block("root"):
        T.reads(B[0:16, 0:16])
        T.writes(B_frag[0:16, 0:16])
        for tx in T.thread_binding(0, 32, "threadIdx.x"):
            T.evaluate(
                T.tvm_load_matrix_sync(
                    B_frag.data,
                    16,
                    16,
                    16,
                    B_frag.elem_offset // 256 + T.floordiv(T.floormod(B_frag.elem_offset, 256), 16),
                    B.access_ptr("r"),
                    B.strides[0],
                    "row_major",
                    dtype="handle",
                )
            )


@T.prim_func
def wmma_fill_desc(c_frag: T.handle) -> None:
    C_frag = T.match_buffer(
        c_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("init"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C_frag[vii, vjj] = T.float16(0)


@T.prim_func
def wmma_fill_impl(c_frag: T.handle) -> None:
    C_frag = T.match_buffer(
        c_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    with T.block("root"):
        T.reads([])
        T.writes(C_frag[0:16, 0:16])
        for tx in T.thread_binding(0, 32, "threadIdx.x"):
            T.evaluate(
                T.tvm_fill_fragment(
                    C_frag.data,
                    16,
                    16,
                    16,
                    C_frag.elem_offset // 256 + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                    T.float16(0),
                    dtype="handle",
                )
            )


@T.prim_func
def wmma_store_desc(c_frag: T.handle, c: T.handle) -> None:
    C_frag = T.match_buffer(
        c_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    C = T.match_buffer(c, (16, 16), "float16", align=128, offset_factor=16, scope="global")
    with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("store"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = C_frag[vii, vjj]


@T.prim_func
def wmma_store_impl(c_frag: T.handle, c: T.handle) -> None:
    s0 = T.var("int32")
    s1 = T.var("int32")
    C_frag = T.match_buffer(
        c_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    C = T.match_buffer(
        c, (16, 16), "float16", align=128, offset_factor=16, scope="global", strides=[s0, s1]
    )
    with T.block("root"):
        T.reads(C_frag[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        for tx in T.thread_binding(0, 32, "threadIdx.x"):
            T.evaluate(
                T.tvm_store_matrix_sync(
                    C_frag.data,
                    16,
                    16,
                    16,
                    C_frag.elem_offset // 256 + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                    C.access_ptr("w"),
                    C.strides[0],
                    "row_major",
                    dtype="handle",
                )
            )


WMMA_SYNC = tir.TensorIntrin.register(
    "wmma_sync",
    wmma_sync_desc,
    wmma_sync_impl,
)

WMMA_LOAD_A = tir.TensorIntrin.register(
    "wmma_load_a",
    wmma_load_a_desc,
    wmma_load_a_impl,
)

WMMA_LOAD_B = tir.TensorIntrin.register(
    "wmma_load_b",
    wmma_load_b_desc,
    wmma_load_b_impl,
)

WMMA_FILL = tir.TensorIntrin.register(
    "wmma_fill",
    wmma_fill_desc,
    wmma_fill_impl,
)


WMMA_STORE = tir.TensorIntrin.register(
    "wmma_store",
    wmma_store_desc,
    wmma_store_impl,
)


block_size = 16
nb = 256
mb = 256
feat_size = 64
num_heads = 12
n = nb * block_size
m = mb * block_size

#C_block = create_pixelfly(1, mb, fmt="bsr")
C_block = create_longformer(1, mb, 256 // block_size, fmt='bsr')

indptr = C_block.indptr
indices = C_block.indices
nnzb = C_block.nnz
np.random.seed(0)
data = np.random.rand(num_heads, nnzb, block_size, block_size)
A = np.random.rand(num_heads, m, feat_size).astype("float16")
B = np.random.rand(num_heads, n, feat_size).astype("float16")
# C = np.matmul(A, B.T).astype("float16")

v_nb, v_mb, v_nnzb, v_blk, v_feat_size, v_num_heads = bsrsddmm.params[-6:]
bsrmm = bsrsddmm.specialize(
    {v_nb: nb, v_mb: mb, v_nnzb: nnzb, v_blk: block_size, v_feat_size: feat_size, v_num_heads: num_heads}
)
sch = tvm.tir.Schedule(bsrmm)
sp_iteration = sch.get_sparse_iteration("sddmm")
h, i, j, bi, bj, f = sch.get_sp_iters(sp_iteration)
sch.sparse_fuse(sp_iteration, [i, j])
mod = lower_sparse_iter(sch.mod)

# split preprocess and compute
mod_preprocess = tvm.tir.transform.ExtractPreprocess()(mod)
mod_sddmm = tvm.tir.transform.RemovePreprocess()(mod)

# schedule preprocess
sch = tir.Schedule(mod_preprocess)
blk = sch.get_block("binary_search_block_0_0")
i, = sch.get_loops(blk)
io, ii = sch.split(i, [None, 32])
sch.bind(ii, "threadIdx.x")
sch.bind(io, "blockIdx.x")
mod = tvm.sparse.lower_sparse_buffer(sch.mod)
mod = tvm.tir.transform.RemoveUnusedArgs()(mod)
preproc = tvm.build(mod["main"], target="cuda")

# compute mid
indptr_nd = tvm.nd.array(indptr, tvm.cuda())
mid_nd = tvm.nd.array(np.zeros((nnzb,), np.int32), tvm.cuda())

preproc(indptr_nd, mid_nd)

# schedule sddmm
sch = tir.Schedule(mod_sddmm)
blk = sch.get_block("sddmm0")
h, j, bi, bj, f = sch.get_loops(blk)
fo, fi = sch.split(f, [None, 16])
sch.bind(h, "blockIdx.y")
sch.reorder(j, fo, bi, fi, bj)
jo, ji = sch.split(j, [None, 4])
sch.bind(jo, "blockIdx.x")
sch.bind(ji, "threadIdx.y")
C_local = sch.cache_write(blk, 0, "wmma.accumulator")
sch.reverse_compute_at(C_local, ji)
new_blk = sch.blockize(bi)
sch.decompose_reduction(new_blk, fo)
A_local = sch.cache_read(blk, 1, "wmma.matrix_a")
B_local = sch.cache_read(blk, 3, "wmma.matrix_b")
sch.hide_buffer_access(blk, "read", [2, 4])
sch.tensorize(sch.get_loops(A_local)[-2], "wmma_load_a")
sch.tensorize(sch.get_loops(B_local)[-2], "wmma_load_b")
sch.tensorize(sch.get_loops(C_local)[-2], "wmma_store")
ax0, ax1, ax2 = sch.get_loops(blk)[-3:]
sch.reorder(ax2, ax1)
sch.tensorize(ax0, "wmma_sync")
sch.tensorize(sch.get_loops(sch.get_block("sddmm0_init"))[-2], "wmma_fill")
mod = lower_sparse_buffer(sch.mod)
f = tvm.build(mod["main"], target="cuda")
# print(f.imported_modules[0].get_source())

ctx = tvm.cuda(0)
C_indptr = tvm.nd.array(np.copy(indptr).astype("int32"), device=ctx)
C_indices = tvm.nd.array(np.copy(indices).astype("int32"), device=ctx)
A_nd = tvm.nd.array(np.copy(A.reshape(-1)).astype("float16"), device=ctx)
B_nd = tvm.nd.array(np.copy(B.reshape(-1)).astype("float16"), device=ctx)
C_nd = tvm.nd.array(np.zeros((num_heads * nnzb * block_size * block_size,), dtype="float16"), device=ctx)
args = [A_nd, B_nd, C_nd, C_indptr, C_indices, mid_nd]
f(*args)
# tvm.testing.assert_allclose(
#     y_ground_truth.reshape(-1),
#     Y_nd.numpy(),
#     rtol=1e-2,
# )

evaluator = f.time_evaluator(f.entry_name, ctx, number=100)
print("avg time: {} ms".format(evaluator(*args).mean * 1000))

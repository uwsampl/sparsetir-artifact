import tvm
from tvm import tir
from tvm.script import tir as T
import tvm.testing
import numpy as np
import scipy.sparse as sp
import argparse
from tvm.ir import IRModule
from tqdm import tqdm
from tvm.sparse import lower_sparse_iter, lower_sparse_buffer
from utils import create_pixelfly, create_longformer
from sparsetir_artifact import profile_tvm_ms


def bsrmm(mb: int, nb: int, nnzb: int, blk: int, feat_size: int, num_heads: int):
    @T.prim_func
    def func(
        a: T.handle,
        b: T.handle,
        c: T.handle,
        indptr: T.handle,
        indices: T.handle,
    ) -> None:
        T.func_attr(
            {"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2}
        )
        I = T.dense_fixed(mb)
        J = T.sparse_variable(I, (nb, nnzb), (indptr, indices), "int32")
        J_detach = T.dense_fixed(nb)
        BI = T.dense_fixed(blk)
        BJ = T.dense_fixed(blk)
        F = T.dense_fixed(feat_size)
        H = T.dense_fixed(num_heads)
        A = T.match_sparse_buffer(a, (H, I, J, BI, BJ), "float16")
        B = T.match_sparse_buffer(b, (H, J_detach, BJ, F), "float16")
        C = T.match_sparse_buffer(c, (H, I, BI, F), "float16")

        with T.iter([H, I, BI, BJ, F, J], "SSSRSR", "bsrmm") as [
            h,
            i,
            bi,
            bj,
            f,
            j,
        ]:
            with T.init():
                C[h, i, bi, f] = T.float16(0.0)
            C[h, i, bi, f] = C[h, i, bi, f] + A[h, i, j, bi, bj] * B[h, j, bj, f]

    return func


@T.prim_func
def wmma_sync_desc(a_frag: T.handle, b_frag: T.handle, c_frag: T.handle) -> None:
    A_frag = T.match_buffer(
        a_frag, (16, 16), "float16", align=128, offset_factor=1, scope="wmma.matrix_a"
    )
    B_frag = T.match_buffer(
        b_frag, (16, 16), "float16", align=128, offset_factor=1, scope="wmma.matrix_b"
    )
    C_frag = T.match_buffer(
        c_frag,
        (16, 16),
        "float16",
        align=128,
        offset_factor=1,
        scope="wmma.accumulator",
    )

    with T.block("root"):
        for i, j, k in T.grid(16, 16, 16):
            with T.block("update"):
                vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
                T.block_attr({"sparse": True})
                C_frag[vii, vjj] = (
                    C_frag[vii, vjj] + A_frag[vii, vkk] * B_frag[vkk, vjj]
                )


@T.prim_func
def wmma_sync_impl(a_frag: T.handle, b_frag: T.handle, c_frag: T.handle) -> None:
    A_frag = T.match_buffer(
        a_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_a"
    )
    B_frag = T.match_buffer(
        b_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_b"
    )
    C_frag = T.match_buffer(
        c_frag,
        (16, 16),
        "float16",
        align=128,
        offset_factor=16,
        scope="wmma.accumulator",
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
                    C_frag.elem_offset // 256
                    + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                    A_frag.data,
                    A_frag.elem_offset // 256
                    + T.floordiv(T.floormod(A_frag.elem_offset, 256), 16),
                    B_frag.data,
                    B_frag.elem_offset // 256
                    + T.floordiv(T.floormod(B_frag.elem_offset, 256), 16),
                    C_frag.data,
                    C_frag.elem_offset // 256
                    + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                    dtype="handle",
                )
            )


@T.prim_func
def wmma_load_a_desc(a: T.handle, a_frag: T.handle) -> None:
    A = T.match_buffer(
        a, (16, 16), "float16", align=128, offset_factor=16, scope="global"
    )
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
        a,
        (16, 16),
        "float16",
        align=128,
        offset_factor=16,
        scope="global",
        strides=[s0, s1],
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
                    A_frag.elem_offset // 256
                    + T.floordiv(T.floormod(A_frag.elem_offset, 256), 16),
                    A.access_ptr("r"),
                    A.strides[0],
                    "row_major",
                    dtype="handle",
                )
            )


@T.prim_func
def wmma_load_b_desc(b: T.handle, b_frag: T.handle) -> None:
    B = T.match_buffer(
        b, (16, 16), "float16", align=128, offset_factor=16, scope="global"
    )
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
        b,
        (16, 16),
        "float16",
        align=128,
        offset_factor=16,
        scope="global",
        strides=[s0, s1],
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
                    B_frag.elem_offset // 256
                    + T.floordiv(T.floormod(B_frag.elem_offset, 256), 16),
                    B.access_ptr("r"),
                    B.strides[0],
                    "row_major",
                    dtype="handle",
                )
            )


@T.prim_func
def wmma_fill_desc(c_frag: T.handle) -> None:
    C_frag = T.match_buffer(
        c_frag,
        (16, 16),
        "float16",
        align=128,
        offset_factor=16,
        scope="wmma.accumulator",
    )
    with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("init"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C_frag[vii, vjj] = T.float16(0)


@T.prim_func
def wmma_fill_impl(c_frag: T.handle) -> None:
    C_frag = T.match_buffer(
        c_frag,
        (16, 16),
        "float16",
        align=128,
        offset_factor=16,
        scope="wmma.accumulator",
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
                    C_frag.elem_offset // 256
                    + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                    T.float16(0),
                    dtype="handle",
                )
            )


@T.prim_func
def wmma_store_desc(c_frag: T.handle, c: T.handle) -> None:
    C_frag = T.match_buffer(
        c_frag,
        (16, 16),
        "float16",
        align=128,
        offset_factor=16,
        scope="wmma.accumulator",
    )
    C = T.match_buffer(
        c, (16, 16), "float16", align=128, offset_factor=16, scope="global"
    )
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
        c_frag,
        (16, 16),
        "float16",
        align=128,
        offset_factor=16,
        scope="wmma.accumulator",
    )
    C = T.match_buffer(
        c,
        (16, 16),
        "float16",
        align=128,
        offset_factor=16,
        scope="global",
        strides=[s0, s1],
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
                    C_frag.elem_offset // 256
                    + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser("SparseTIR sparse attention spmm")
    parser.add_argument(
        "--pattern", "-p", type=str, help="Sparse pattern: longformer/pixelfly"
    )
    parser.add_argument(
        "--check", "-c", action="store_true", help="Whether to check result or not."
    )
    block_size = 16
    mb = 256
    nb = 256
    feat_size = 64
    num_heads = 12
    m = mb * block_size
    n = nb * block_size
    args = parser.parse_args()

    if args.pattern == "pixelfly":
        A_block = create_pixelfly(1, mb, fmt="bsr")
    elif args.pattern == "longformer":
        A_block = create_longformer(1, mb, 256 // block_size, fmt="bsr")
    else:
        raise KeyError("Sparse pattern {} not recongized.".format(args.pattern))
    indptr = A_block.indptr
    indices = A_block.indices
    nnzb = A_block.nnz
    np.random.seed(0)
    data = np.random.rand(num_heads, nnzb, block_size, block_size)
    x = np.random.rand(num_heads, n, feat_size).astype("float16")
    if args.check:
        A = sp.bsr_matrix((data.astype("float16"), indices, indptr), shape=(m, n))
        y_ground_truth = (A * x).astype("float16")

    best_dur = 1e9
    for coarsening_factor in [1, 2, 4]:
        for ty in [2, 4, 8, 16]:
            sch = tvm.tir.Schedule(
                bsrmm(mb, nb, nnzb, block_size, feat_size, num_heads)
            )
            sp_iteration = sch.get_sparse_iteration("bsrmm")
            h, i, bi, bj, f, j = sch.get_sp_iters(sp_iteration)
            sch.sparse_reorder(sp_iteration, [h, i, j, bi, f, bj])
            mod = lower_sparse_iter(sch.mod)
            sch = tir.Schedule(mod)
            blk_inner = sch.get_block("bsrmm1")
            blk_outer = sch.get_block("bsrmm0")
            j, bi, f, bj = sch.get_loops(blk_inner)
            foo, foi, fi = sch.split(f, [None, coarsening_factor, 16])
            sch.reorder(foo, j, foi, bi, fi, bj)
            (
                h,
                i,
            ) = sch.get_loops(blk_outer)
            io, ii = sch.split(i, [None, ty])
            sch.bind(h, "blockIdx.z")
            sch.bind(io, "blockIdx.x")
            sch.bind(ii, "threadIdx.y")
            sch.bind(foo, "blockIdx.y")
            sch.unroll(foi)
            new_blk = sch.blockize(bi)
            C_local = sch.cache_write(new_blk, 0, "wmma.accumulator")
            sch.reverse_compute_at(C_local, foo, True)
            ax0, ax1, ax2 = sch.get_loops(C_local)[-3:]
            ax3, ax2 = sch.split(ax2, [None, 16])
            sch.reorder(ax0, ax3, ax1, ax2)
            sch.unroll(ax3)
            sch.decompose_reduction(new_blk, j)
            A_local = sch.cache_read(blk_inner, 1, "wmma.matrix_a")
            B_local = sch.cache_read(blk_inner, 2, "wmma.matrix_b")
            sch.hide_buffer_access(blk_inner, "read", [3])
            sch.tensorize(sch.get_loops(blk_inner)[-3], "wmma_sync")
            sch.tensorize(sch.get_loops(B_local)[-2], "wmma_load_b")
            sch.tensorize(sch.get_loops(A_local)[-2], "wmma_load_a")
            sch.tensorize(sch.get_loops(C_local)[-2], "wmma_store")
            sch.tensorize(sch.get_loops(sch.get_block("bsrmm1_init"))[-2], "wmma_fill")
            mod = lower_sparse_buffer(sch.mod)
            f = tvm.build(mod["main"], target="cuda")

            ctx = tvm.cuda(0)
            A_indptr = tvm.nd.array(np.copy(indptr).astype("int32"), device=ctx)
            A_indices = tvm.nd.array(np.copy(indices).astype("int32"), device=ctx)
            A_data = tvm.nd.array(
                np.copy(data).reshape(-1).astype("float16"), device=ctx
            )
            X_nd = tvm.nd.array(np.copy(x.reshape(-1)).astype("float16"), device=ctx)
            Y_nd = tvm.nd.array(
                np.zeros((num_heads * mb * block_size * feat_size), dtype="float16"),
                device=ctx,
            )
            fargs = [A_data, X_nd, Y_nd, A_indptr, A_indices]
            f(*fargs)
            if args.check:
                tvm.testing.assert_allclose(
                    y_ground_truth.reshape(-1),
                    Y_nd.numpy(),
                    rtol=1e-2,
                )

            dur = profile_tvm_ms(f, fargs)
            best_dur = min(dur, best_dur)
    print("avg time: {} ms".format(best_dur))

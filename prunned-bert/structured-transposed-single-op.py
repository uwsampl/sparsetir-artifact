import argparse
from typing import Any
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import BertModel
from scipy import sparse as sp
import numpy as np
import torch as th
import pandas as pd
import tvm
from tvm.sparse.format import condense
import tvm.testing
import tvm.tir as tir
from tvm.script import tir as T
from tvm.sparse import lower_sparse_buffer, lower_sparse_iter
import torch.utils.benchmark as benchmark


@T.prim_func
def bsrmm(
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
) -> None:
    T.func_attr({
        "global_symbol": "main",
        "tir.noalias": True,
        "sparse_tir_level": 2
    })
    I = T.dense_fixed(nb)
    J = T.sparse_variable(I, (mb, nnzb), (indptr, indices), "int32")
    J_detach = T.dense_fixed(mb)
    BI = T.dense_fixed(blk)
    BJ = T.dense_fixed(blk)
    F = T.dense_fixed(feat_size)
    A = T.match_sparse_buffer(a, (I, J, BI, BJ), "float16")
    B = T.match_sparse_buffer(b, (J_detach, BJ, F), "float16")
    C = T.match_sparse_buffer(c, (I, BI, F), "float16")

    with T.iter([I, BI, BJ, F, J], "SSRSR", "bsrmm") as [
            i,
            bi,
            bj,
            f,
            j,
    ]:
        with T.init():
            C[i, bi, f] = T.float16(0.0)
        C[i, bi, f] = C[i, bi, f] + A[i, j, bi, bj] * B[j, bj, f]


@T.prim_func
def wmma_sync_desc(a_frag: T.handle, b_frag: T.handle,
                   c_frag: T.handle) -> None:
    A_frag = T.match_buffer(a_frag, (16, 16),
                            "float16",
                            align=128,
                            offset_factor=1,
                            scope="wmma.matrix_a")
    B_frag = T.match_buffer(b_frag, (16, 16),
                            "float16",
                            align=128,
                            offset_factor=1,
                            scope="wmma.matrix_b")
    C_frag = T.match_buffer(c_frag, (16, 16),
                            "float16",
                            align=128,
                            offset_factor=1,
                            scope="wmma.accumulator")

    with T.block("root"):
        for i, j, k in T.grid(16, 16, 16):
            with T.block("update"):
                vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
                T.block_attr({"sparse": True})
                C_frag[vii,
                       vjj] = C_frag[vii,
                                     vjj] + A_frag[vii, vkk] * B_frag[vkk, vjj]


@T.prim_func
def wmma_sync_impl(a_frag: T.handle, b_frag: T.handle,
                   c_frag: T.handle) -> None:
    A_frag = T.match_buffer(a_frag, (16, 16),
                            "float16",
                            align=128,
                            offset_factor=16,
                            scope="wmma.matrix_a")
    B_frag = T.match_buffer(b_frag, (16, 16),
                            "float16",
                            align=128,
                            offset_factor=16,
                            scope="wmma.matrix_b")
    C_frag = T.match_buffer(c_frag, (16, 16),
                            "float16",
                            align=128,
                            offset_factor=16,
                            scope="wmma.accumulator")

    with T.block("root"):
        T.reads([
            C_frag[0:16, 0:16],
            A_frag[0:16, 0:16],
            B_frag[0:16, 0:16],
        ])
        T.writes(C_frag[0:16, 0:16])
        for tx in T.thread_binding(0, 32, "threadIdx.x"):
            T.evaluate(
                T.tvm_mma_sync(
                    C_frag.data,
                    C_frag.elem_offset // 256 +
                    T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                    A_frag.data,
                    A_frag.elem_offset // 256 +
                    T.floordiv(T.floormod(A_frag.elem_offset, 256), 16),
                    B_frag.data,
                    B_frag.elem_offset // 256 +
                    T.floordiv(T.floormod(B_frag.elem_offset, 256), 16),
                    C_frag.data,
                    C_frag.elem_offset // 256 +
                    T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                    dtype="handle",
                ))


@T.prim_func
def wmma_load_a_desc(a: T.handle, a_frag: T.handle) -> None:
    A = T.match_buffer(a, (16, 16),
                       "float16",
                       align=128,
                       offset_factor=16,
                       scope="global")
    A_frag = T.match_buffer(a_frag, (16, 16),
                            "float16",
                            align=128,
                            offset_factor=16,
                            scope="wmma.matrix_a")

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
    A = T.match_buffer(a, (16, 16),
                       "float16",
                       align=128,
                       offset_factor=16,
                       scope="global",
                       strides=[s0, s1])
    A_frag = T.match_buffer(a_frag, (16, 16),
                            "float16",
                            align=128,
                            offset_factor=16,
                            scope="wmma.matrix_a")

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
                    A_frag.elem_offset // 256 +
                    T.floordiv(T.floormod(A_frag.elem_offset, 256), 16),
                    A.access_ptr("r"),
                    A.strides[0],
                    "row_major",
                    dtype="handle",
                ))


@T.prim_func
def wmma_load_b_desc(b: T.handle, b_frag: T.handle) -> None:
    B = T.match_buffer(b, (16, 16),
                       "float16",
                       align=128,
                       offset_factor=16,
                       scope="global")
    B_frag = T.match_buffer(b_frag, (16, 16),
                            "float16",
                            align=128,
                            offset_factor=16,
                            scope="wmma.matrix_b")
    with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("load"):
                vii, vjj = T.axis.remap("SS", [i, j])
                B_frag[vii, vjj] = B[vii, vjj]


@T.prim_func
def wmma_load_b_impl(b: T.handle, b_frag: T.handle) -> None:
    s0 = T.var("int32")
    s1 = T.var("int32")
    B = T.match_buffer(b, (16, 16),
                       "float16",
                       align=128,
                       offset_factor=16,
                       scope="global",
                       strides=[s0, s1])
    B_frag = T.match_buffer(b_frag, (16, 16),
                            "float16",
                            align=128,
                            offset_factor=16,
                            scope="wmma.matrix_b")
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
                    B_frag.elem_offset // 256 +
                    T.floordiv(T.floormod(B_frag.elem_offset, 256), 16),
                    B.access_ptr("r"),
                    B.strides[0],
                    "row_major",
                    dtype="handle",
                ))


@T.prim_func
def wmma_fill_desc(c_frag: T.handle) -> None:
    C_frag = T.match_buffer(c_frag, (16, 16),
                            "float16",
                            align=128,
                            offset_factor=16,
                            scope="wmma.accumulator")
    with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("init"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C_frag[vii, vjj] = T.float16(0)


@T.prim_func
def wmma_fill_impl(c_frag: T.handle) -> None:
    C_frag = T.match_buffer(c_frag, (16, 16),
                            "float16",
                            align=128,
                            offset_factor=16,
                            scope="wmma.accumulator")
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
                    C_frag.elem_offset // 256 +
                    T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                    T.float16(0),
                    dtype="handle",
                ))


@T.prim_func
def wmma_store_desc(c_frag: T.handle, c: T.handle) -> None:
    C_frag = T.match_buffer(c_frag, (16, 16),
                            "float16",
                            align=128,
                            offset_factor=16,
                            scope="wmma.accumulator")
    C = T.match_buffer(c, (16, 16),
                       "float16",
                       align=128,
                       offset_factor=16,
                       scope="global")
    with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("store"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = C_frag[vii, vjj]


@T.prim_func
def wmma_store_impl(c_frag: T.handle, c: T.handle) -> None:
    s0 = T.var("int32")
    s1 = T.var("int32")
    C_frag = T.match_buffer(c_frag, (16, 16),
                            "float16",
                            align=128,
                            offset_factor=16,
                            scope="wmma.accumulator")
    C = T.match_buffer(c, (16, 16),
                       "float16",
                       align=128,
                       offset_factor=16,
                       scope="global",
                       strides=[s0, s1])
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
                    C_frag.elem_offset // 256 +
                    T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                    C.access_ptr("w"),
                    C.strides[0],
                    "row_major",
                    dtype="handle",
                ))


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


def bench_bsrmm(bsr_mat: Any, x: th.Tensor):
    global bsrmm
    mb = bsr_mat.shape[0] // bsr_mat.blocksize[0]
    nb = bsr_mat.shape[1] // bsr_mat.blocksize[1]
    block_size = 32
    nnzb = bsr_mat.nnz // (block_size**2)
    feat_size = x.shape[1]

    v_mb, v_nb, v_nnzb, v_blk, v_feat_size = bsrmm.params[-5:]
    mod = tvm.IRModule.from_expr(
        bsrmm.specialize({
            v_mb: mb,
            v_nb: nb,
            v_nnzb: nnzb,
            v_blk: block_size,
            v_feat_size: feat_size
        }))
    sch = tvm.tir.Schedule(mod)
    sp_iteration = sch.get_sparse_iteration("bsrmm")
    i, bi, bj, f, j = sch.get_sp_iters(sp_iteration)
    sch.sparse_reorder(sp_iteration, [i, j, bi, f, bj])
    mod = lower_sparse_iter(sch.mod)
    sch = tir.Schedule(mod)
    blk_inner = sch.get_block("bsrmm1")
    blk_outer = sch.get_block("bsrmm0")
    j, bi, f, bj = sch.get_loops(blk_inner)
    bio, bii = sch.split(bi, [2, 16])
    bjo, bji = sch.split(bj, [2, 16])
    foo, foi, fi = sch.split(f, [None, 2, 16])
    sch.reorder(foo, j, bio, bjo, foi, bii, fi, bji)
    sch.lift_loop(bio)
    new_blk = sch.blockize(bii)
    (i, bio) = sch.get_loops(blk_outer)
    io, ii = sch.split(i, [None, 8])
    sch.reorder(io, bio, ii)
    fused_i = sch.fuse(io, bio)
    sch.bind(fused_i, "blockIdx.x")
    sch.bind(ii, "threadIdx.y")
    sch.bind(foo, "blockIdx.y")
    sch.unroll(foi)
    C_local = sch.cache_write(new_blk, 0, "wmma.accumulator")
    sch.reverse_compute_at(C_local, foi, True)
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
    A_indptr = tvm.nd.array(np.copy(bsr_mat.indptr).astype("int32"),
                            device=ctx)
    A_indices = tvm.nd.array(np.copy(bsr_mat.indices).astype("int32"),
                             device=ctx)
    A_data = tvm.nd.array(np.copy(bsr_mat.data).reshape(-1).astype("float16"),
                          device=ctx)
    X_nd = tvm.nd.array(np.copy(x.reshape(-1)).astype("float16"), device=ctx)
    Y_nd = tvm.nd.array(np.zeros((mb * block_size * feat_size),
                                 dtype="float16"),
                        device=ctx)
    args = [A_data, X_nd, Y_nd, A_indptr, A_indices]
    f(*args)

    evaluator = f.time_evaluator(f.entry_name, tvm.cuda(0), number=100)
    avg_time = evaluator(*args).mean
    print("bsrmm time: \t{:.5f}ms".format(avg_time * 1000))
    return avg_time


def bench_cublas(W: th.Tensor, X: th.Tensor):
    with th.no_grad():
        W = W.half().to(0)
        X = X.to(0)
        timer = benchmark.Timer(stmt="a @ b", globals={'a': W, 'b': X})
        measure = timer.timeit(100)

        print("cublas time: \t{:.5f}ms".format(measure.mean * 1000))
        return measure.mean


def bench_cusparse(csr: Any, X: th.Tensor):
    W = th.sparse_csr_tensor(csr.indptr,
                             csr.indices,
                             csr.data,
                             size=csr.shape,
                             dtype=th.float16).to(0)
    with th.no_grad():
        W = W.half().to(0)
        X = X.to(0)
        timer = benchmark.Timer(stmt="a @ b", globals={'a': W, 'b': X})
        measure = timer.timeit(100)

        print("cusparse time: \t{:.5f}ms".format(measure.mean * 1000))
        return measure.mean


if __name__ == "__main__":
    parser = argparse.ArgumentParser("structured prunned bert")
    parser.add_argument("--dim",
                        "-d",
                        type=int,
                        default=128,
                        help="feature size")
    parser.add_argument("--csv",
                        "-c",
                        action="store_true",
                        help="whether to dump csv file or not")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        "vuiseng9/bert-base-squadv1-block-pruning-hybrid")

    model = AutoModelForQuestionAnswering.from_pretrained(
        "vuiseng9/bert-base-squadv1-block-pruning-hybrid")

    sparsetir_durs = []
    cublas_durs = []
    cusparse_durs = []
    densities = []
    for name, param in model.named_parameters():
        if name.endswith("key.weight") or name.endswith(
                "value.weight") or name.endswith("query.weight"):

            bsr_weight = sp.bsr_matrix(param.detach().numpy(),
                                       shape=param.shape,
                                       blocksize=(32, 32))
            csr_weight = sp.csr_matrix(param.detach().numpy())
            x = th.rand(csr_weight.shape[1], args.dim).half()
            densities.append(csr_weight.nnz / param.numel())
            print('--------------------')
            print('density:\t{:.3f}'.format(densities[-1]))
            sparsetir_durs.append(bench_bsrmm(bsr_weight, x))
            cublas_durs.append(bench_cublas(param.data, x))
            cusparse_durs.append(bench_cusparse(csr_weight, x))

    print(sum(sparsetir_durs), sum(cublas_durs), sum(cusparse_durs))
    if args.csv:
        pd = pd.DataFrame(
            data={
                "density": densities,
                "sparsetir_dur": sparsetir_durs,
                "cublas_dur": cublas_durs,
                "cusparse_dur": cusparse_durs
            })
        pd.to_csv("single_op.csv", index=False)
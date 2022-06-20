#include <cuda_runtime.h>

#include <cassert>
#include <vector>

#include "cuda_common.h"

#ifndef BLOCKSPARSE_SPMM_H_
#define BLOCKSPARSE_SPMM_H_

namespace blocksparse {

namespace {

template <typename DType>
cusparseStatus_t bsrmm(cusparseHandle_t handle, cusparseDirection_t dirA,
                       cusparseOperation_t transA, cusparseOperation_t transB, int mb, int n,
                       int kb, int nnzb, const DType* alpha, const cusparseMatDescr_t descrA,
                       const DType* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA,
                       int blockDim, const DType* B, int ldb, const DType* beta, DType* C,
                       int ldc) {
  return CUSPARSE_STATUS_NOT_SUPPORTED;
}

template <>
cusparseStatus_t bsrmm<float>(cusparseHandle_t handle, cusparseDirection_t dirA,
                              cusparseOperation_t transA, cusparseOperation_t transB, int mb, int n,
                              int kb, int nnzb, const float* alpha, const cusparseMatDescr_t descrA,
                              const float* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA,
                              int blockDim, const float* B, int ldb, const float* beta, float* C,
                              int ldc) {
  return cusparseSbsrmm(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrValA,
                        bsrRowPtrA, bsrColIndA, blockDim, B, ldb, beta, C, ldc);
}

template <>
cusparseStatus_t bsrmm<double>(cusparseHandle_t handle, cusparseDirection_t dirA,
                               cusparseOperation_t transA, cusparseOperation_t transB, int mb,
                               int n, int kb, int nnzb, const double* alpha,
                               const cusparseMatDescr_t descrA, const double* bsrValA,
                               const int* bsrRowPtrA, const int* bsrColIndA, int blockDim,
                               const double* B, int ldb, const double* beta, double* C, int ldc) {
  return cusparseDbsrmm(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrValA,
                        bsrRowPtrA, bsrColIndA, blockDim, B, ldb, beta, C, ldc);
}

}  // namespace

/*
 * \param m number of rows
 * \param n number of feature dimension
 * \param k number of columns
 * \param indptr The index pointer array.
 * \parma indices The column indices array.
 * \param x the feature matrix array.
 * \param weight the edge weight array.
 */
template <typename IdType, typename DType>
void SpMMCsr(IdType m, IdType n, IdType k, IdType nnz, IdType* indptr, IdType* indices, DType* x,
             DType* weight, DType* out) {
  DType alpha = 1., beta = 0.;
  auto thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  if (!thr_entry->cusparse_handle) {
    CUSPARSE_CALL(cusparseCreate(&(thr_entry->cusparse_handle)));
  }
  auto handle = thr_entry->cusparse_handle;
  CUSPARSE_CALL(cusparseSetStream(handle, thr_entry->stream));
  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB, matC;

  auto dtype = typecast::CUDA<DType>::Get();
  auto idtype = typecast::CuSparse<IdType>::Get();

  CUSPARSE_CALL(cusparseCreateCsr(&matA, m, k, nnz, indptr, indices, const_cast<DType*>(weight),
                                  idtype, idtype, CUSPARSE_INDEX_BASE_ZERO, dtype));
  CUSPARSE_CALL(
      cusparseCreateDnMat(&matB, k, n, n, const_cast<DType*>(x), dtype, CUSPARSE_ORDER_ROW));
  CUSPARSE_CALL(cusparseCreateDnMat(&matC, m, n, n, out, dtype, CUSPARSE_ORDER_ROW));

  size_t workspace_size;
  auto transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  auto transB = CUSPARSE_OPERATION_NON_TRANSPOSE;
  CUSPARSE_CALL(cusparseSpMM_bufferSize(handle, transA, transB, &alpha, matA, matB, &beta, matC,
                                        dtype, CUSPARSE_SPMM_CSR_ALG2, &workspace_size));
  void* workspace;
  cudaMalloc(&workspace, workspace_size);
  CUSPARSE_CALL(cusparseSpMM(handle, transA, transB, &alpha, matA, matB, &beta, matC, dtype,
                             CUSPARSE_SPMM_CSR_ALG2, workspace));
  cudaFree(workspace);
  CUSPARSE_CALL(cusparseDestroyDnMat(matC));
  CUSPARSE_CALL(cusparseDestroyDnMat(matB));
  CUSPARSE_CALL(cusparseDestroySpMat(matA));
}

/*
 * \param mb number of blocks in rows
 * \param n number of feature dimension
 * \param kb number of blocks in columns
 * \param nnzb number of non zero blocks of sparse matrix.
 * \param block_dim the block size
 * \param indptr the block index pointer array.
 * \param indices the block column indices array.
 * \param x the feature matrix array.
 * \param weight the edge weight array (nnz * block_dim * block_dim).
 */
template <typename IdType, typename DType>
void SpMMBsr(IdType mb, IdType n, IdType kb, IdType nnzb, IdType block_size, IdType* indptr,
             IdType* indices, DType* x, DType* weight, DType* out) {
  DType alpha = 1., beta = 0.;
  auto thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  if (!thr_entry->cusparse_handle) {
    CUSPARSE_CALL(cusparseCreate(&(thr_entry->cusparse_handle)));
  }
  auto handle = thr_entry->cusparse_handle;
  CUSPARSE_CALL(cusparseSetStream(handle, thr_entry->stream));

  // create matrix descriptor for sparse matrix.
  cusparseMatDescr_t descrA = 0;
  CUSPARSE_CALL(cusparseCreateMatDescr(&descrA));
  CUSPARSE_CALL(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
  CUSPARSE_CALL(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

  const IdType m = mb * block_size;
  const IdType k = kb * block_size;
  // matrix B: k * n, matrix C: m * n
  const IdType ldb = n;
  const IdType ldc = n;

  CUSPARSE_CALL(bsrmm<DType>(handle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
                             CUSPARSE_OPERATION_NON_TRANSPOSE, mb, n, kb, nnzb, &alpha, descrA,
                             weight, indptr, indices, block_size, x, ldb, &beta, out, ldc));

  CUSPARSE_CALL(cusparseDestroyMatDescr(descrA));
}

/*
 * \param rows number of rows
 * \param n number of feature size
 * \param cols number of columns
 * \param block_size the block size
 * \param ell_cols number of block columns per block row.
 * \param col_ind the column block indices.
 * \parma x the feature matrix array.
 * \param weight the edge weight array (row * ell_cols)
 */
template <typename IdType, typename DType>
void SpMMEll(IdType rows, IdType n, IdType cols, IdType block_size, IdType ell_cols,
             IdType* col_ind, DType* x, DType* weight, DType* out) {
  assert(rows % block_size == 0);
  assert(cols % block_size == 0);
  IdType m = rows, k = cols;
  // TODO(zihao): fix

  DType alpha = 1., beta = 0.;
  auto thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  if (!thr_entry->cusparse_handle) {
    CUSPARSE_CALL(cusparseCreate(&(thr_entry->cusparse_handle)));
  }
  auto handle = thr_entry->cusparse_handle;
  CUSPARSE_CALL(cusparseSetStream(handle, thr_entry->stream));
  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB, matC;

  auto dtype = typecast::CUDA<DType>::Get();
  auto idtype = typecast::CuSparse<IdType>::Get();

  CUSPARSE_CALL(cusparseCreateBlockedEll(&matA, rows, cols, block_size, ell_cols, col_ind, weight,
                                         idtype, CUSPARSE_INDEX_BASE_ZERO, dtype));

  CUSPARSE_CALL(cusparseCreateDnMat(&matB, k, n, n, x, dtype, CUSPARSE_ORDER_ROW));

  CUSPARSE_CALL(cusparseCreateDnMat(&matC, m, n, n, out, dtype, CUSPARSE_ORDER_ROW));

  size_t workspace_size;
  auto transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  auto transB = CUSPARSE_OPERATION_NON_TRANSPOSE;
  CUSPARSE_CALL(cusparseSpMM_bufferSize(handle, transA, transB, &alpha, matA, matB, &beta, matC,
                                        dtype, CUSPARSE_SPMM_ALG_DEFAULT, &workspace_size));
  void* workspace;
  cudaMalloc(&workspace, workspace_size);
  CUSPARSE_CALL(cusparseSpMM(handle, transA, transB, &alpha, matA, matB, &beta, matC, dtype,
                             CUSPARSE_SPMM_ALG_DEFAULT, workspace));
  cudaFree(workspace);
  CUSPARSE_CALL(cusparseDestroyDnMat(matC));
  CUSPARSE_CALL(cusparseDestroyDnMat(matB));
  CUSPARSE_CALL(cusparseDestroySpMat(matA));
}

}  // namespace blocksparse

#endif  // BLOCKSPARSE_SPMM_H_
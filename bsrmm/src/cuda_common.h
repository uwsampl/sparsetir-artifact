/*
 * Common utilities for CUDA.
 */
#ifndef BLOCKSPARSE_SRC_CUDA_COMMON_H_
#define BLOCKSPARSE_SRC_CUDA_COMMON_H_

#include <cublas.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusparse.h>

namespace blocksparse {
namespace runtime {

#define CUSPARSE_CALL(func)                                                                    \
  {                                                                                            \
    cusparseStatus_t e = (func);                                                               \
    if (e != CUSPARSE_STATUS_SUCCESS) perror(cusparseGetErrorString(e)); \
  }

#define CUBLAS_CALL(func)                                          \
  {                                                                \
    cublasStatus_t e = (func);                                     \
    if (e != CUBLAS_STATUS_SUCCESS) perror("CUBLAS ERROR"); \
  }

class CUDAThreadEntry {
 public:
  /*! \brief The cuda stream */
  cudaStream_t stream{nullptr};
  /*! \brief The cusparse handler */
  cusparseHandle_t cusparse_handle{nullptr};
  /*! \brief The cublas handler */
  cublasHandle_t cublas_handle{nullptr};
  /*! \brief The curand generator */
  curandGenerator_t curand_gen{nullptr};
  /*! \brief constructor */
  CUDAThreadEntry();
  // get the threadlocal workspace
  static CUDAThreadEntry* ThreadLocal();
};

}  // namespace runtime

namespace typecast {

template <typename DType>
struct CUDA {
  static cudaDataType_t Get() { return CUDA_R_32F; }
};

template <>
struct CUDA<float> {
  static cudaDataType_t Get() { return CUDA_R_32F; }
};

template <>
struct CUDA<double> {
  static cudaDataType_t Get() { return CUDA_R_64F; }
};

template <>
struct CUDA<half> {
  static cudaDataType_t Get() { return CUDA_R_16F; }
};

template <>
struct CUDA<nv_bfloat16> {
  static cudaDataType_t Get() { return CUDA_R_16BF; }
};

template <typename IdType>
struct CuSparse {
  static cusparseIndexType_t Get() { return CUSPARSE_INDEX_32I; }
};

template <>
struct CuSparse<int32_t> {
  static cusparseIndexType_t Get() { return CUSPARSE_INDEX_32I; }
};

template <>
struct CuSparse<int64_t> {
  static cusparseIndexType_t Get() { return CUSPARSE_INDEX_64I; }
};

}  // namespace typecast

}  // namespace blocksparse

#endif  // BLOCKSPARSE_SRC_CUDA_COMMON_H_
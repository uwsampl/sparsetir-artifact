#include <cnpy.h>
#include <cuda_runtime_api.h>
#include "bsr_spmm.h"

#define CUDA_CHECK(func)                                                       \
  {                                                                            \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
      printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__,     \
             cudaGetErrorString(status), status);                              \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  }

void read_npz_file(const std::string& filename, int &M, int &NNZ_COLS, int& B, std::vector<int>& indptr, std::vector<int>& indices) {
  cnpy::npz_t data = cnpy::npz_load(filename);
  cnpy::NpyArray shape = data["shape"];
  int* shape_data = shape.data<int>();
  B = shape_data[0];
  M = shape_data[1];
  NNZ_COLS = shape_data[2];
  indptr = std::move(data["indptr"].as_vec<int>());
  indices = std::move(data["indices"].as_vec<int>());
}

struct GpuTimer {
  cudaEvent_t startEvent;
  cudaEvent_t stopEvent;

  GpuTimer() {
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
  }

  ~GpuTimer() {
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
  }

  void start() { cudaEventRecord(startEvent, 0); }

  void stop() {
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
  }

  float elapsed_msecs() {
    float elapsed;
    cudaEventElapsedTime(&elapsed, startEvent, stopEvent);
    return elapsed;
  }
};

int main(int argc, char* argv[]) {
  if (argc != 4) {
    perror("Input format: bsrmm FORMAT_PATH BATCH_SIZE FEAT_SIZE");
    abort();
  }
  std::vector<int> indptr;
  std::vector<int> indices;
  int M, NNZ_COLS, B;
  read_npz_file(argv[1], M, NNZ_COLS, B, indptr, indices);
  int MB = M / B;
  int NNZB = NNZ_COLS * MB;
  int batch_size = std::stoi(argv[2]);
  int feat_size = std::stoi(argv[3]);
  
  int *indptr_d = NULL, *indices_d = NULL;
  half *A_d = NULL, *B_d = NULL, *C_d = NULL;

  CUDA_CHECK(cudaMalloc((void **)&indptr_d, sizeof(int) * (MB + 1)));
  CUDA_CHECK(cudaMalloc((void **)&indices_d, sizeof(int) * NNZB));
  CUDA_CHECK(cudaMalloc((void **)&A_d, sizeof(half) * NNZB * B * B));
  CUDA_CHECK(cudaMalloc((void **)&B_d, sizeof(half) * M * feat_size));
  CUDA_CHECK(cudaMalloc((void **)&C_d, sizeof(half) * M * feat_size));

  assert(indptr.size() == MB + 1);
  assert(indices.size() == NNZB);
  CUDA_CHECK(cudaMemcpy(indptr_d, indptr.data(), sizeof(int) * (MB + 1), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(indices_d, indices.data(), sizeof(int) * NNZB, cudaMemcpyHostToDevice));

  // benchmark cusparseELLMM
  {
    GpuTimer gpu_timer;
    int warmup_iter = 10;
    int repeat_iter = 100;
    for (int iter = 0; iter < warmup_iter + repeat_iter; iter++) {
      if (iter == warmup_iter) {
        gpu_timer.start();
      }
      blocksparse::SpMMEll(M, feat_size, M, B, NNZ_COLS * B, indices_d, B_d, A_d, C_d);
    }
    gpu_timer.stop();
    float kernel_dur_msecs = gpu_timer.elapsed_msecs() / repeat_iter;
    printf("[cusparse ellmm]: #batch=( %d ) #feat=( %d ), Time %f (ms)\n", batch_size, feat_size, kernel_dur_msecs);
  }

  CUDA_CHECK(cudaFree(indptr_d));
  CUDA_CHECK(cudaFree(indices_d));
  CUDA_CHECK(cudaFree(A_d));
  CUDA_CHECK(cudaFree(B_d));
  CUDA_CHECK(cudaFree(C_d));

  return 0;
}
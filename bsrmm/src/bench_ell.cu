#include <cnpy.h>
#include <cuda_runtime_api.h>
#include "bsr_spmm.h"

void read_npz_file(const std::string& filename, int &M, int &K, int &NNZ, std::vector<int>& indptr, std::vector<int>& indices) {
  cnpy::npz_t data = cnpy::npz_load(filename);
  cnpy::NpyArray shape = data["shape"];
  int* shape_data = shape.data<int>();
  M = shape_data[0];
  K = shape_data[1];
  NNZ = shape_data[2];
  indptr = std::move(data["indptr"].as_vec<int>());
  indices = std::move(data["indices"].as_vec<int>());
}

int main(int argc, char* argv[]) {
  return 0;
}
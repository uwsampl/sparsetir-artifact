cmake_minimum_required(VERSION 3.16)

add_subdirectory(cnpy)

project(bsrmm C CXX CUDA)
find_package( CUDAToolkit )
set(CMAKE_CXX_FLAGS "-O2 -Wall -std=c++14")

file(GLOB BSRMM_SRC
  bench_ell.cu
)

add_executable(bsrmm ${BSRMM_SRC})
target_include_directories(bsrmm PRIVATE cnpy)
target_include_directories(bsrmm PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(bsrmm PRIVATE CUDA::cusparse CUDA::cublas CUDA::curand CUDA::cudart cnpy)

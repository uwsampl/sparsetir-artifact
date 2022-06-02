cmake_minimum_required(VERSION 3.16)

project(sddmm C CXX CUDA)
find_package( CUDAToolkit )
set(CMAKE_CXX_FLAGS "-O2 -Wall -std=c++14")

file(GLOB SDDMMSRC
  *.cc
)

add_executable(sddmm ${SDDMMSRC})
target_include_directories(sddmm PRIVATE ../../cnpy)
target_include_directories(sddmm PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(sddmm PRIVATE CUDA::cusparse CUDA::cublas CUDA::curand CUDA::cudart cnpy dgsparse)

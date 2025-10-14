cmake_minimum_required(VERSION 3.17)
include(CheckLanguage)
check_language(CUDA)
project(nvshmem LANGUAGES CXX C CUDA)
set(CMAKE_CXX_STANDARD 20)
include_directories(
  /opt/amazon/efa/include
  /usr/local/cuda/include
)
link_directories(
  /opt/amazon/efa/lib
  /usr/local/cuda/lib64
  /opt/nccl/build/lib
)

add_subdirectory(third_party/multi-gpu-programming-models/single_gpu)
add_subdirectory(third_party/multi-gpu-programming-models/single_threaded_copy)
add_subdirectory(third_party/multi-gpu-programming-models/nccl)
add_subdirectory(third_party/multi-gpu-programming-models/nccl_graphs)
add_subdirectory(third_party/multi-gpu-programming-models/nccl_overlap)
add_subdirectory(third_party/multi-gpu-programming-models/nvshmem)

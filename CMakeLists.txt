cmake_minimum_required(VERSION 3.20)
project(CUDA_MEMOPT LANGUAGES CXX CUDA)

# Set CUDA architectures (can be overridden with -DCMAKE_CUDA_ARCHITECTURES=...)
set(CMAKE_CUDA_ARCHITECTURES "61" CACHE STRING "Target CUDA architectures (e.g. 61;75;86)")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Suppress CMP0104 policy warnings
if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif()

# List all CUDA kernel variants
set(KERNEL_VARIANTS
  naive
  tiled
  tiled_vec
  tiled_vec_unroll
  tuned
)

# Add executables for each variant
foreach(kernel IN LISTS KERNEL_VARIANTS)
  add_executable(matmul_${kernel} src/matmul_${kernel}.cu)

  # Use host compiler optimization flags in a cross-platform way:
  # - MSVC:     /O2
  # - GCC/Clang: -O3
  target_compile_options(matmul_${kernel} PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/O2>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Xcompiler=-O3>
  )

  set_target_properties(matmul_${kernel} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
endforeach()

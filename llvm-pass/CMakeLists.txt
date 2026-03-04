cmake_minimum_required(VERSION 3.20)  # LLVM requires CMake ≥3.20
project(LoopTilingPass LANGUAGES CXX)

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

# Include LLVM headers and your pass headers
include_directories(
  ${LLVM_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/include
)

# Define the pass module (source lives at llvm-pass/LoopTiling.cpp)
add_llvm_loadable_module(LoopTilingPass
  LoopTiling.cpp
  include/LoopTiling.h
)

# Link against required LLVM libraries
llvm_map_components_to_libnames(REQ_LLVM_LIBS
  Core Support Analysis TransformUtils
)
target_link_libraries(LoopTilingPass PRIVATE ${REQ_LLVM_LIBS})

# Set compiler flags and C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

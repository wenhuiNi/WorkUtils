cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(cuda_abs)
SET(CMAKE_BUILD_TYPE "Release") 
find_package(Torch REQUIRED)
 
add_executable(cuda_abs testForNaiveDilatedConv.cpp)
target_link_libraries(cuda_abs "${TORCH_LIBRARIES}")
set_property(TARGET cuda_abs PROPERTY CXX_STANDARD 14)


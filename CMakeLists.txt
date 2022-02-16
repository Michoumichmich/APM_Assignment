cmake_minimum_required(VERSION 3.10)
project(Project_APM_MIGDAL CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif (NOT CMAKE_BUILD_TYPE)

#  Compile Flags
if (${CMAKE_CUDA_COMPILER_ID} STREQUAL "NVIDIA")
    set(NVCC_GENERAL_FLAGS --extended-lambda -ftz=true --relocatable-device-code=true --save-temps)
    set(NVCC_OPT_FLAGS -prec-sqrt=false -prec-div=false --use_fast_math -O3 --ptxas-options -v,-warn-lmem-usage,-warn-spills,-Werror)
else () # To use CLANG++ as the cuda compiler pass CUDACXX=clang++ to cmake.
    set(NVCC_GENERAL_FLAGS -stdlib=libc++ --cuda-path=/usr/local/cuda -Xcuda-ptxas -v -Wno-unknown-cuda-version --save-temps)
    set(NVCC_OPT_FLAGS -Ofast -fgpu-inline-threshold=10000) # Secret flag to allow great perf.
    set(NVCC_LINK_FLAGS -stdlib=libc++ --cuda-path=/usr/local/cuda -L/usr/local/cuda/lib64 -lcudart_static -ldl -lrt -lcuda)
endif ()

# Find FreeImage
set(FreeImage_ROOT /usr)
FIND_PATH(FreeImage_INCLUDE_DIR FreeImage.h HINTS ${FreeImage_ROOT} ${FreeImage_HINT})
FIND_LIBRARY(FreeImage_LIBRARY NAMES freeimage HINTS ${FreeImage_ROOT} ${FreeImage_HINT})
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FreeImage DEFAULT_MSG FreeImage_INCLUDE_DIR FreeImage_LIBRARY)

include_directories(include)

# Main executable
add_executable(main main.cu src/cuda_filters.cu src/free_image_wrapper.cu)
if (${CMAKE_CUDA_COMPILER_ID} STREQUAL "NVIDIA")
    set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif ()

# Compile options for main
target_link_options(main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_LINK_FLAGS}>)
target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_GENERAL_FLAGS}>)
target_compile_options(main PRIVATE $<$<AND:$<CONFIG:RELEASE>,$<COMPILE_LANGUAGE:CUDA>>:${NVCC_OPT_FLAGS}>)

# FreeImage
target_link_libraries(main ${FreeImage_LIBRARY})
target_include_directories(main PRIVATE ${FreeImage_INCLUDE_DIR})

#if (USE_HPC_KIT)
#    find_package(EnvModules REQUIRED)
#    set(ENV{MODULESPATH} /opt/nvidia/hpc_sdk/modulefiles)
#    env_module(purge)
#    env_module(load modules)
#    env_module(load nvhpc)
#    set(CUDACXX nvc++)t'
#endif ()
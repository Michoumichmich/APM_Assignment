#pragma once

#include "cuda_err_check.cuh"
#include "lambda_type_checker.hpp"
#include <cstdint>

static constexpr int threads_per_block_1D = 1024;
static constexpr int pixels_per_thread_1D = 32;

static constexpr int threads_per_block_2D_x = 128;
static constexpr int threads_per_block_2D_y = 4;
static constexpr int pixels_per_thread_2D_x = 4;
static constexpr int pixels_per_thread_2D_y = 1;


template<typename func> __global__ static void run_1D_kernel(size_t n, func kernel) {
    type_checker::enforce_reqd_lambda_traits<func, 1>();
    uint32_t tid = threadIdx.x + pixels_per_thread_1D * blockIdx.x * blockDim.x;
#pragma unroll
    for (int i = 0; i < pixels_per_thread_1D; ++i) {
        if (tid + i * blockDim.x >= n) { return; }
        kernel(tid + i * blockDim.x);
    }
}

template<typename func> static inline void dispatch_1D_kernel(size_t n, cudaStream_t stream, func&& kernel) {
    constexpr int pixels_per_block = threads_per_block_1D * pixels_per_thread_1D;
    int n_blocks = (n + pixels_per_block - 1) / (pixels_per_block);
    run_1D_kernel<<<n_blocks, threads_per_block_1D, 0, stream>>>(n, kernel);
    CHECK_LAST_ERROR_ASYNC
}

template<typename func> static inline void dispatch_1D_kernel(size_t n, func&& kernel) { return dispatch_1D_kernel(n, nullptr, std::move(kernel)); }


/**
 *
 * @param src
 * @param dst
 */
template<typename func> __global__ void static run_2D_stencil_kernel(int32_t width, int32_t height, func kernel) {
    type_checker::enforce_reqd_lambda_traits<func, 2>();
    int horizontal_stride = blockDim.x;
    int vertical_stride = blockDim.y;
    int horizontal_offset = threadIdx.x + blockIdx.x * horizontal_stride * pixels_per_thread_2D_x;
    int vertical_offset = threadIdx.y + blockIdx.y * vertical_stride * pixels_per_thread_2D_y;
#pragma unroll
    for (int i = 0; i < pixels_per_thread_2D_x; ++i) {
#pragma unroll
        for (int j = 0; j < pixels_per_thread_2D_y; ++j) {
            int x = horizontal_offset + i * horizontal_stride;
            int y = vertical_offset + j * vertical_stride;
            if (x >= width || y >= height) { continue; }
            kernel(x, y);
        }
    }
}

template<typename func> void static inline dispatch_2D_stencil_kernel(int32_t width, int32_t height, cudaStream_t stream, func&& kernel) {
    constexpr int pixels_per_block_x = threads_per_block_2D_x * pixels_per_thread_2D_x;
    constexpr int pixels_per_block_y = threads_per_block_2D_y * pixels_per_thread_2D_y;
    dim3 block(threads_per_block_2D_x, threads_per_block_2D_y);
    dim3 grid((width + pixels_per_block_x - 1) / pixels_per_block_x, (height + pixels_per_block_y - 1) / pixels_per_block_y);
    run_2D_stencil_kernel<<<grid, block, 0, stream>>>(width, height, kernel);
    CHECK_LAST_ERROR_ASYNC
}

template<typename func> void static inline dispatch_2D_stencil_kernel(int32_t width, int32_t height, func&& kernel) { return dispatch_2D_stencil_kernel(width, height, nullptr, std::move(kernel)); }
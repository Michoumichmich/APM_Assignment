#pragma once

#include <cstdio>

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = false) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define CUDA_CHECK(ans)                                                                                                                                                                                \
    { gpuAssert((ans), __FILE__, __LINE__); }

#define CHECK_LAST_ERROR_ASYNC CUDA_CHECK(cudaGetLastError())

#pragma once

#include <cuda_runtime.h>
#include "common.h"
#include "hdcommon.h"

#include <iostream>

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t _CudaSetup();

#ifdef __cplusplus
} // extern "C"
#endif

cudaError_t _CudaPoolReport();

void _print_stacktrace();

template <typename... Args>
cudaError_t _CudaCheck(cudaError_t err, const Args&... args) {
    if (err != cudaSuccess) {
        (std::cerr << ... << args) << " : " << cudaGetErrorName(err) << " : " << cudaGetErrorString(err) << std::endl;
        _CudaPoolReport();
        _print_stacktrace();
    }
    return err;
}

template <typename... Args>
cudaError_t _CudaPrint(cudaError_t err, const Args&... args) {
    (std::cerr << ... << args) << " : " << cudaGetErrorName(err) << " : " << cudaGetErrorString(err) << std::endl;
    return err;
}

// #define _CUDA_DEBUG

#ifdef _CUDA_DEBUG

#define _CUDA_CHECK(cudaCall) _CudaCheck(cudaCall, __FILE__, ":", __LINE__, " : ", #cudaCall)
#define _CUDA_PRINT(cudaCall) _CudaPrint(cudaCall, __FILE__, ":", __LINE__, " : ", #cudaCall)

#else /* _CUDA_DEBUG */

#define _CUDA_CHECK(cudaCall) cudaCall
#define _CUDA_PRINT(cudaCall) cudaCall

#endif /* _CUDA_DEBUG */

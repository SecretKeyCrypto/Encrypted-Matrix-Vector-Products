#include "cudacommon.h"
#include <cstdint>
#include <execinfo.h>

// #define USE_COMMON_DEBUG

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t _CudaSetup() {
    int device;
    cudaError_t err;

#ifdef USE_COMMON_DEBUG
    std::cerr << "Cuda setup: setting pool release threshold" << std::endl;
#endif /* USE_COMMON_DEBUG */

    if (cudaSuccess != (err = _CUDA_CHECK(cudaGetDevice(&device)))) {
        return err;
    }

    cudaMemPool_t pool;
    if (cudaSuccess != (err = _CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool, device)))) {
        return err;
    }

    size_t threshold = SIZE_MAX; // or any desired value in bytes
    if (cudaSuccess != (err = _CUDA_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold)))) {
        return err;
    }

#ifdef USE_COMMON_DEBUG
    std::cerr << "Cuda setup: pre-growing pool" << std::endl;
#endif /* USE_COMMON_DEBUG */

    cudaStream_t stream;
    if (cudaSuccess != (err = _CUDA_CHECK(cudaStreamCreate(&stream)))) {
        return err;
    }

    void* tmp;
    if (cudaSuccess != (err = _CUDA_CHECK(cudaMallocAsync(&tmp, 1024*1024*1024, stream)))) {
        return err;
    }
    if (cudaSuccess != (err = _CUDA_CHECK(cudaFreeAsync(tmp, stream)))) {
        return err;
    }

    if (cudaSuccess != (err = _CUDA_CHECK(cudaStreamSynchronize(stream)))) {
        return err;
    }
    if (cudaSuccess != (err = _CUDA_CHECK(cudaStreamDestroy(stream)))) {
        return err;
    }

#ifdef USE_COMMON_DEBUG
    std::cerr << "Cuda setup: complete" << std::endl;
#endif /* USE_COMMON_DEBUG */

    return cudaSuccess;
}

#ifdef __cplusplus
} // extern "C"
#endif

cudaError_t _CudaPoolReport() {
    int device;
    cudaError_t err;
    if (cudaSuccess != (err = cudaGetDevice(&device))) {
        return err;
    }
    cudaMemPool_t memPool;
    if (cudaSuccess != (err = cudaDeviceGetDefaultMemPool(&memPool, device))) {
        return err;
    }
    size_t reserved, used;
    if (cudaSuccess != (err = cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrReservedMemCurrent, &reserved))) {
        return err;
    }
    if (cudaSuccess != (err = cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrUsedMemCurrent, &used))) {
        return err;
    }
    std::cerr << "Memory pool: reserved=" << reserved << " used=" << used << std::endl;


    size_t freeMem, totalMem;
    if (cudaSuccess != (err = cudaMemGetInfo(&freeMem, &totalMem))) {
        return err;
    }

    std::cerr << "GPU memory status:" << std::endl;
    std::cerr << "  Free  = " << freeMem  << " bytes ("
              << static_cast<double>(freeMem) / (1024.0 * 1024.0 * 1024.0)
              << " GB)" << std::endl;
    std::cerr << "  Total = " << totalMem << " bytes ("
              << static_cast<double>(totalMem) / (1024.0 * 1024.0 * 1024.0)
              << " GB)" << std::endl;

    return cudaSuccess;
}

void _print_stacktrace() {
    void *array[50];
    size_t size = backtrace(array, 50);
    char **strings = backtrace_symbols(array, size);
    fprintf(stderr, "Stack trace:\n");
    for (size_t i = 0; i < size; i++) {
        fprintf(stderr, "%s\n", strings[i]);
    }
    free(strings);
}

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <stdexcept>

#include "common.h"
#include "cudafields.h"

// #define _FIELDS_VALIDATE

#ifdef _FIELDS_VALIDATE
static void _error() {
    std::cerr << "Cuda error" << std::endl;
    throw std::runtime_error("Cuda error");
}
#endif

__global__ void CudaRangeVectorKernel(uint32_t* r, uint32_t start, uint64_t length) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        r[idx] = start + idx;
    }
}

__global__ void CudaCopyVectorKernel(uint32_t* r, const uint32_t* a, uint64_t length) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        r[idx] = a[idx];
    }
}

__global__ void CudaSetVectorKernel(uint32_t* r, uint64_t length, uint32_t v) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        r[idx] = v;
    }
}

__global__ void CudaAddToVectorKernel(uint32_t* r, uint32_t v, uint64_t length) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        r[idx] += v;
    }
}

__global__ void CudaAddVectorsKernel(uint32_t* r, const uint32_t* a, const uint32_t* b, uint64_t length, uint32_t p) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        uint32_t sum = a[idx] + b[idx];
        r[idx] = sum >= p ? sum - p : sum;
    }
}

__global__ void CudaMulVectorKernel(uint32_t* r, const uint32_t* a, uint32_t b, uint64_t length, uint32_t p) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        uint64_t product = static_cast<uint64_t>(a[idx]) * b;
        r[idx] = static_cast<uint32_t>(product % p);
    }
}

__global__ void CudaMulVectorsKernel(uint32_t* r, const uint32_t* a, const uint32_t* b, uint64_t length, uint32_t p) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        uint64_t product = static_cast<uint64_t>(a[idx]) * static_cast<uint64_t>(b[idx]);
        r[idx] = static_cast<uint32_t>(product % p);
    }
}

__global__ void CudaSubVectorsKernel(uint32_t* r, const uint32_t* a, const uint32_t* b, uint64_t length, uint32_t p) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        uint32_t av = a[idx];
        uint32_t bv = b[idx];
        r[idx] = (av >= bv) ? (av - bv) : (p - (bv - av));
    }
}

__global__ void CudaNegVectorsKernel(uint32_t* r, uint64_t length, uint32_t p) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        uint32_t val = r[idx];
        r[idx] = (val == 0) ? 0 : (p - val);
    }
}

__global__ void CudaIsNonZeroVectorKernel(bool* t, const uint32_t* e, uint64_t length) {
    __shared__ bool has_non_zero;
    if (threadIdx.x == 0) {
        has_non_zero = false;
    }
    __syncthreads();

    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length && e[idx] != 0) {
        atomicExch((int*)&has_non_zero, true);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        *t = has_non_zero;
    }
}

__global__ void CudaAddVectorIfNonZeroKernel(bool* t, uint32_t* r, const uint32_t* e, uint64_t length, uint32_t p) {
    __shared__ bool has_non_zero;
    if (threadIdx.x == 0) {
        has_non_zero = false;
    }
    __syncthreads();

    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length && e[idx] != 0) {
        atomicExch((int*)&has_non_zero, true);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        *t = has_non_zero;

        if (has_non_zero) {
            uint64_t blocksPerGrid = blocksPerGridFor(length);
            CudaAddVectorsKernel<<<blocksPerGrid, threadsPerBlock>>>(r, r, e, length, p);
            //FIXME cudaDeviceSynchronize(); // wait for nested kernel invocation
        }
    }
}

// Host wrappers
void CudaRangeVector(uint32_t* r, uint32_t start, uint64_t length) {
    uint64_t blocksPerGrid = blocksPerGridFor(length);
    CudaRangeVectorKernel<<<blocksPerGrid, threadsPerBlock>>>(r, start, length);
}

void CudaCopyVector(uint32_t* r, const uint32_t* a, uint64_t length) {
    uint64_t blocksPerGrid = blocksPerGridFor(length);
    CudaCopyVectorKernel<<<blocksPerGrid, threadsPerBlock>>>(r, a, length);
}

void CudaSetVector(uint32_t* r, uint64_t length, uint32_t v) {
    uint64_t blocksPerGrid = blocksPerGridFor(length);
    CudaSetVectorKernel<<<blocksPerGrid, threadsPerBlock>>>(r, length, v);
}

void CudaAddToVector(uint32_t* r, uint32_t v, uint64_t length) {
    uint64_t blocksPerGrid = blocksPerGridFor(length);
    CudaAddToVectorKernel<<<blocksPerGrid, threadsPerBlock>>>(r, v, length);
}

void CudaAddVectors(uint32_t* r, const uint32_t* a, const uint32_t* b, uint64_t length, uint32_t p) {
    uint64_t blocksPerGrid = blocksPerGridFor(length);
    CudaAddVectorsKernel<<<blocksPerGrid, threadsPerBlock>>>(r, a, b, length, p);
#ifdef _FIELDS_VALIDATE
    cudaDeviceSynchronize();
    for (uint64_t i = 0; i < length; i++) {
        uint32_t t = a[i] + b[i];
        if (r[i] != (t > p ? t - p : t)) {
            _error();
        }
    }
#endif
}

void CudaMulVector(uint32_t* r, const uint32_t* a, uint32_t b, uint64_t length, uint32_t p) {
    uint64_t blocksPerGrid = blocksPerGridFor(length);
    CudaMulVectorKernel<<<blocksPerGrid, threadsPerBlock>>>(r, a, b, length, p);
#ifdef _FIELDS_VALIDATE
    cudaDeviceSynchronize();
    for (uint64_t i = 0; i < length; i++) {
        if (r[i] != (uint64_t(a[i]) * uint64_t(b)) % p) {
            _error();
        }
    }
#endif
}

void CudaMulVectors(uint32_t* r, const uint32_t* a, const uint32_t* b, uint64_t length, uint32_t p) {
    uint64_t blocksPerGrid = blocksPerGridFor(length);
    CudaMulVectorsKernel<<<blocksPerGrid, threadsPerBlock>>>(r, a, b, length, p);
#ifdef _FIELDS_VALIDATE
    cudaDeviceSynchronize();
    for (uint64_t i = 0; i < length; i++) {
        if (r[i] != (uint64_t(a[i]) * uint64_t(b)) % p) {
            _error();
        }
    }
#endif
}

void CudaSubVectors(uint32_t* r, const uint32_t* a, const uint32_t* b, uint64_t length, uint32_t p) {
    uint64_t blocksPerGrid = blocksPerGridFor(length);
    CudaSubVectorsKernel<<<blocksPerGrid, threadsPerBlock>>>(r, a, b, length, p);
#ifdef _FIELDS_VALIDATE
    cudaDeviceSynchronize();
    for (uint64_t i = 0; i < length; i++) {
        if (r[i] != ((a[i] >= b[i]) ? (a[i] - b[i]) : (p - (b[i] - a[i])))) {
            _error();
        }
    }
#endif
}

void CudaNegVector(uint32_t* r, uint64_t length, uint32_t p) {
#ifdef _FIELDS_VALIDATE
    cudaDeviceSynchronize();
    uint32_t r0[length];
    for (uint64_t i = 0; i < length; i++) {
        r0[i] = r[i];
    }
#endif
    uint64_t blocksPerGrid = blocksPerGridFor(length);
    CudaNegVectorsKernel<<<blocksPerGrid, threadsPerBlock>>>(r, length, p);
#ifdef _FIELDS_VALIDATE
    cudaDeviceSynchronize();
    for (uint64_t i = 0; i < length; i++) {
        if (r[i] != (r0[i] == 0) ? 0 : (p - r0[i])) {
            _error();
        }
    }
#endif
}

void FieldIsNonZeroVector(bool* t, const uint32_t* e, uint64_t eo, uint64_t length) {
    uint64_t blocksPerGrid = blocksPerGridFor(length);
    CudaIsNonZeroVectorKernel<<<blocksPerGrid, threadsPerBlock>>>(t, e + eo, length);
}

void FieldAddVectorIfNonZero(bool* t, uint32_t* r, const uint32_t* e, uint64_t length, uint32_t p) {
    uint64_t blocksPerGrid = blocksPerGridFor(length);
    CudaAddVectorIfNonZeroKernel<<<blocksPerGrid, threadsPerBlock>>>(t, r, e, length, p);
}

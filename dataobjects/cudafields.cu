#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include "cudafields.h"

// #define _FIELDS_VALIDATE

#ifdef _FIELDS_VALIDATE
static void _error() {
    std::cerr << "blah" << std::endl;
    throw std::runtime_error("blah");
}
#endif

__global__ void FieldAddVectorsKernel(uint32_t* r, const uint32_t* a, const uint32_t* b, uint64_t length, uint32_t p) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        uint32_t sum = a[idx] + b[idx];
        r[idx] = sum >= p ? sum - p : sum;
    }
}

__global__ void FieldMulVectorKernel(uint32_t* r, const uint32_t* a, uint32_t b, uint64_t length, uint32_t p) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        uint64_t product = static_cast<uint64_t>(a[idx]) * b;
        r[idx] = static_cast<uint32_t>(product % p);
    }
}

__global__ void FieldMulVectorsKernel(uint32_t* r, const uint32_t* a, const uint32_t* b, uint64_t length, uint32_t p) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        uint64_t product = static_cast<uint64_t>(a[idx]) * static_cast<uint64_t>(b[idx]);
        r[idx] = static_cast<uint32_t>(product % p);
    }
}

__global__ void FieldSubVectorsKernel(uint32_t* r, const uint32_t* a, const uint32_t* b, uint64_t length, uint32_t p) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        uint32_t av = a[idx];
        uint32_t bv = b[idx];
        r[idx] = (av >= bv) ? (av - bv) : (p - (bv - av));
    }
}

__global__ void FieldNegVectorsKernel(uint32_t* r, uint64_t length, uint32_t p) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        uint32_t val = r[idx];
        r[idx] = (val == 0) ? 0 : (p - val);
    }
}

// Host wrappers
int CudaFieldAddVectors(uint32_t* r, uint64_t ro,
                     const uint32_t* a, uint64_t ao,
                     const uint32_t* b, uint64_t bo,
                     uint64_t length, uint32_t p) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (length + threadsPerBlock - 1) / threadsPerBlock;
    FieldAddVectorsKernel<<<blocksPerGrid, threadsPerBlock>>>(r + ro, a + ao, b + bo, length, p);
    cudaDeviceSynchronize();
#ifdef _FIELDS_VALIDATE
    for (uint64_t i = 0; i < length; i++) {
        uint32_t t = a[i] + b[i];
        if (r[i] != (t > p ? t - p : t)) {
            _error();
        }
    }
#endif
    return 2;
}

int CudaFieldMulVector(uint32_t* r, uint64_t ro,
                    const uint32_t* a, uint64_t ao,
                    uint32_t b, uint64_t length, uint32_t p) {
    int threads = 256;
    int blocks = (length + threads - 1) / threads;
    FieldMulVectorKernel<<<blocks, threads>>>(r + ro, a + ao, b, length, p);
    cudaDeviceSynchronize();
#ifdef _FIELDS_VALIDATE
    for (uint64_t i = 0; i < length; i++) {
        if (r[i] != (uint64_t(a[i]) * uint64_t(b)) % p) {
            _error();
        }
    }
#endif
    return 2;
}

int CudaFieldMulVectors(uint32_t* r, uint64_t ro,
                     const uint32_t* a, uint64_t ao,
                     const uint32_t* b, uint64_t bo,
                     uint64_t length, uint32_t p) {
    int threads = 256;
    int blocks = (length + threads - 1) / threads;
    FieldMulVectorsKernel<<<blocks, threads>>>(r + ro, a + ao, b + bo, length, p);
    cudaDeviceSynchronize();
#ifdef _FIELDS_VALIDATE
    for (uint64_t i = 0; i < length; i++) {
        if (r[i] != (uint64_t(a[i]) * uint64_t(b)) % p) {
            _error();
        }
    }
#endif
    return 2;
}

int CudaFieldSubVectors(uint32_t* r, uint64_t ro,
                     const uint32_t* a, uint64_t ao,
                     const uint32_t* b, uint64_t bo,
                     uint64_t length, uint32_t p) {
    int threads = 256;
    int blocks = (length + threads - 1) / threads;
    FieldSubVectorsKernel<<<blocks, threads>>>(r + ro, a + ao, b + bo, length, p);
    cudaDeviceSynchronize();
#ifdef _FIELDS_VALIDATE
    for (uint64_t i = 0; i < length; i++) {
        if (r[i] != ((a[i] >= b[i]) ? (a[i] - b[i]) : (p - (b[i] - a[i])))) {
            _error();
        }
    }
#endif
    return 2;
}

int CudaFieldNegVector(uint32_t* r, uint64_t ro,
                    uint64_t length, uint32_t p) {
#ifdef _FIELDS_VALIDATE
    uint32_t r0[length];
    for (uint64_t i = 0; i < length; i++) {
        r0[i] = r[i];
    }
#endif
    int threads = 256;
    int blocks = (length + threads - 1) / threads;
    FieldNegVectorsKernel<<<blocks, threads>>>(r + ro, length, p);
    cudaDeviceSynchronize();
#ifdef _FIELDS_VALIDATE
    for (uint64_t i = 0; i < length; i++) {
        if (r[i] != (r0[i] == 0) ? 0 : (p - r0[i])) {
            _error();
        }
    }
#endif
    return 2;
}
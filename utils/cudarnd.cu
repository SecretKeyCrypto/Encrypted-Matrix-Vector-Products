#ifdef USE_FAST_CODE_WITH_CUDA

#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <curand_kernel.h>

#include "../dataobjects/common.h"

constexpr uint32_t NO_MODULUS = 0;
constexpr uint32_t randElemPerThread = 8;

__global__ void randomize_kernel(uint32_t* data, uint32_t M, uint32_t N, bool transpose, int64_t seed, uint32_t modulus) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)M * N;
    uint64_t start = (uint64_t)tid * randElemPerThread;
    if (start >= total) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, 0, start, &state);

    // compute starting coordinates
    uint32_t row = (uint32_t)(start / N);
    uint32_t col = (uint32_t)(start % N);

    uint32_t produced = 0;
    for (uint32_t r = row; r < M && produced < 8; ++r) {
        for (uint32_t c = (r == row ? col : 0); c < N && produced < randElemPerThread; ++c) {
            uint64_t idx = transpose ? (uint64_t)c * M + r : (uint64_t)r * N + c;
            if (idx >= total) break;
            uint32_t v = curand(&state);
            if (modulus != NO_MODULUS) {
                v %= modulus;
            }
            data[idx] = v;
            ++produced;
        }
    }
}


void cuda_randomize_vector(uint32_t* data, uint32_t M, uint32_t N, bool transpose) {
    if (!data) return;
    uint64_t blocksPerGrid = blocksPerGridFor(M * N);
    randomize_kernel<<<blocksPerGrid, threadsPerBlock>>>(data, M, N, transpose, clock(), NO_MODULUS);
}

void cuda_randomize_vector_with_seed(uint32_t* data, uint32_t M, uint32_t N, bool transpose, int64_t seed) {
    if (!data) return;
    uint64_t blocksPerGrid = blocksPerGridFor(M * N);
    randomize_kernel<<<blocksPerGrid, threadsPerBlock>>>(data, M, N, transpose, seed, NO_MODULUS);
}

void cuda_randomize_vector_with_modulus(uint32_t* data, uint32_t M, uint32_t N, bool transpose, uint32_t modulus) {
    if (!data) return;
    uint64_t blocksPerGrid = blocksPerGridFor(M * N);
    randomize_kernel<<<blocksPerGrid, threadsPerBlock>>>(data, M, N, transpose, clock(), modulus);
}

void cuda_randomize_vector_with_modulus_and_seed(uint32_t* data, uint32_t M, uint32_t N, bool transpose, uint32_t modulus, int64_t seed) {
    if (!data) return;
    uint64_t blocksPerGrid = blocksPerGridFor(M * N);
    randomize_kernel<<<blocksPerGrid, threadsPerBlock>>>(data, M, N, transpose, seed, modulus);
}

__global__ void CudaLPNNoiseVectorKernel(uint32_t* r, uint64_t length, double epsi, uint32_t p, uint32_t pmask, unsigned long long seed) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length) return;

    curandState state;
    curand_init(seed, idx, 0, &state);

    double f = curand_uniform_double(&state);
    if (f <= epsi) {
        uint32_t u;
        do {
            u = curand(&state) & pmask;
        } while (u >= p - 1);
        r[idx] = u + 1;
    }
}

void cuda_lpn_noise_vector(uint32_t* r, uint64_t length, double epsi, uint32_t p) {
    uint64_t blocksPerGrid = blocksPerGridFor(length);
    CudaLPNNoiseVectorKernel<<<blocksPerGrid, threadsPerBlock>>>(r, length, epsi, p, bitmask_for(p), time(NULL));
}

#endif
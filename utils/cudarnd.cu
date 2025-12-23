#include <cuda.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

#include "cudarnd.h"
#include "../dataobjects/cudacommon.h"
#include "../dataobjects/docontextimpl.h"
#include "../dataobjects/cudagraph.h"
#include "../dataobjects/cudafields.h"
#include "../dataobjects/cudaalloc.h"
#include "../dataobjects/cudacub.h"

constexpr uint32_t NO_MODULUS = 0;

__global__ void randomize_kernel(uint32_t* data, uint32_t M, uint32_t N, bool transpose, bool circulant, int64_t seed, int64_t offset, uint32_t modulus, int randElemPerThread) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)M * N;
    uint64_t start = (uint64_t)tid * randElemPerThread;
    if (start >= total) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, offset >= 0 ? offset : 0, start, &state);

    // compute starting coordinates
    uint32_t row = (uint32_t)(start / N);
    uint32_t col = (uint32_t)(start % N);

    uint32_t produced = 0;
    for (uint32_t r = row; r < M && produced < randElemPerThread; ++r) {
        for (uint32_t c = (r == row ? col : 0); c < N && produced < randElemPerThread; ++c) {
            uint32_t cc = circulant ? (c == 0 ? 0 : N - c) : c;
            uint64_t idx = transpose ? (uint64_t)cc * M + r : (uint64_t)r * N + cc;
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

bool cuda_randomize_vector_main(DoContext* ctx, uint32_t* data, uint32_t M, uint32_t N, bool transpose, bool circulant, int64_t seed, int64_t offset, uint32_t modulus) {
    CudaVectorGraph* graph = CudaVectorGraph::Get(ctx);
    if (!graph) return false;

    int randElementsPerThread = graph->GetParams().randElementsPerThread;

    return graph->Compute({}, {data}, [ctx, data, M, N, transpose, circulant, seed, offset, modulus, randElementsPerThread](cudaStream_t stream) {
        uint64_t totalElements = (uint64_t)M * N;
        uint64_t threadsPerBlock = threadsPerBlockFor(totalElements, randElementsPerThread);
        uint64_t blocksPerGrid   = blocksPerGridFor(totalElements, threadsPerBlock, randElementsPerThread);

        dim3 block(threadsPerBlock);
        dim3 grid(blocksPerGrid);

        randomize_kernel<<<grid, block, 0, stream>>>(data, M, N, transpose, circulant, seed, offset, modulus, randElementsPerThread);
        if (_CUDA_CHECK(cudaGetLastError()) != cudaSuccess) {
            return false;
        }
        return true;
    });
}

#ifdef __cplusplus
extern "C" {
#endif

bool cuda_randomize_vector(DoContext* ctx, uint32_t* data, uint32_t M, uint32_t N, bool transpose, bool circulant) {
    return cuda_randomize_vector_main(ctx, data, M, N, transpose, circulant, clock(), 0, NO_MODULUS);
}

bool cuda_randomize_vector_with_seed(DoContext* ctx, uint32_t* data, uint32_t M, uint32_t N, bool transpose, bool circulant, int64_t seed, int64_t offset) {
    return cuda_randomize_vector_main(ctx, data, M, N, transpose, circulant, seed, offset, NO_MODULUS);
}

bool cuda_randomize_vector_with_modulus(DoContext* ctx, uint32_t* data, uint32_t M, uint32_t N, bool transpose, bool circulant, uint32_t modulus) {
    return cuda_randomize_vector_main(ctx, data, M, N, transpose, circulant, clock(), 0, modulus);
}

bool cuda_randomize_vector_with_modulus_and_seed(DoContext* ctx, uint32_t* data, uint32_t M, uint32_t N, bool transpose, bool circulant, uint32_t modulus, int64_t seed, int64_t offset) {
    return cuda_randomize_vector_main(ctx, data, M, N, transpose, circulant, seed, offset, modulus);
}

#ifdef __cplusplus
}
#endif

__global__ void CudaLPNNoiseVectorKernel(uint32_t* r,
                                         uint32_t length,
                                         double epsi,
                                         uint32_t p,
                                         uint32_t pmask,
                                         int64_t seed,
                                         int64_t offset,
                                         int randElementsPerThread)
{
    // Global thread id
    uint32_t tid   = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start = tid * randElementsPerThread;

    // Initialize RNG state once per thread
    curandState state;
    curand_init(seed, offset >= 0 ? offset : 0, start, &state);

    // Each thread generates randElementsPerThread consecutive entries
    for (int e = 0; e < randElementsPerThread; ++e) {
        uint32_t idx = start + e;
        if (idx >= length) break;

        double f = curand_uniform_double(&state);
        if (f <= epsi) {
            uint32_t u;
            do {
                u = curand(&state) & pmask;
            } while (u >= p - 1);
            r[idx] = u + 1;
        } else {
            r[idx] = 0;
        }
    }
}

#ifdef __cplusplus
extern "C" {
#endif

bool cuda_lpn_noise_vector(DoContext* ctx, uint32_t* r, uint64_t ro, uint64_t length, double epsi, uint32_t p, int64_t seed, int64_t offset) {
    CudaVectorGraph* graph = CudaVectorGraph::Get(ctx);
    if (!graph) return false;

    int randElementsPerThread = graph->GetParams().randElementsPerThread;

    return graph->Compute({}, {r}, [ctx, r = r + ro, length, epsi, p, seed, offset, randElementsPerThread](cudaStream_t stream) {
        uint64_t threadsPerBlock = threadsPerBlockFor(length, randElementsPerThread);
        uint64_t blocksPerGrid   = blocksPerGridFor(length, threadsPerBlock, randElementsPerThread);

        dim3 block(threadsPerBlock);
        dim3 grid(blocksPerGrid);

        CudaLPNNoiseVectorKernel<<<grid, block, 0, stream>>>(
            r, length, epsi, p, bitmask_for(p), seed, offset, randElementsPerThread
        );
        if (_CUDA_CHECK(cudaGetLastError()) != cudaSuccess) {
            return false;
        }
        return true;
    });
}

#ifdef __cplusplus
}
#endif

// Kernel to generate random 64-bit keys using cuRAND
__global__ void random_permutation_keys_kernel(uint64_t* keys, uint32_t n, uint64_t seed, int randElementsPerThread) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t base = tid * randElementsPerThread;

    if (base >= n) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, 0, base, &state); // FIXME - add offset

    for (int i = 0; i < randElementsPerThread && (base + i) < n; ++i) {
        uint64_t lo = curand(&state);
        uint64_t hi = curand(&state);
        keys[base + i] = (hi << 32) | lo;
    }
}

#ifdef __cplusplus
extern "C" {
#endif

bool random_permutation_keys(DoContext* ctx, uint64_t* keys, uint32_t n, uint64_t seed) {
    CudaVectorGraph* graph = CudaVectorGraph::Get(ctx);
    if (!graph) return false;

    int randElementsPerThread = graph->GetParams().randElementsPerThread;
    return graph->Compute({}, {keys}, [ctx, keys, n, seed, randElementsPerThread](cudaStream_t stream) mutable {
        uint64_t threadsPerBlock = threadsPerBlockFor(n, randElementsPerThread);
        uint64_t blocksPerGrid   = blocksPerGridFor(n, threadsPerBlock, randElementsPerThread);

        dim3 block(threadsPerBlock);
        dim3 grid(blocksPerGrid);

        random_permutation_keys_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            keys, n, static_cast<uint64_t>(seed), randElementsPerThread
        );
        if (_CUDA_CHECK(cudaGetLastError()) != cudaSuccess) {
            return false;
        }
        return true;
    });
}

// Main permutation function
bool cuda_random_permutation(DoContext* ctx, uint32_t* d_perm, uint32_t n, int64_t seed, int64_t offset) {
    CudaVectorGraph* graph = CudaVectorGraph::Get(ctx);
    if (!graph) return false;

    // Allocate keys
    uint64_t* d_keys = static_cast<uint64_t*>(graph->Malloc(sizeof(uint64_t) * n));
    if (!d_keys) return false;

    cudaError_t status;
    GraphAllocator<char> galloc(graph);

    auto cub_sort = [d_keys, d_perm, n](char* temp, size_t& bytes, cudaStream_t s) {
        return cub::DeviceRadixSort::SortPairs(
            temp, bytes,
            d_keys,
            d_keys,
            d_perm,
            d_perm,
            n,
            0, // begin bit
            sizeof(uint64_t) * 8, // end bit
            s);
    };
    auto cubcall_sort = CubCall(std::function(cub_sort), galloc);
    status = cubcall_sort.Prepare();
    if (status != cudaSuccess) return false;

    bool success;

    // Fill perm
    success = CudaFieldRangeVector(ctx, d_perm, 0, 0, n);
    if (!success) return false;

    success = random_permutation_keys(ctx, d_keys, n, seed);
    if (!success) return false;

    int elementsPerThread = graph->GetParams().elementsPerThread;
    success = graph->Compute({d_perm, d_keys}, {d_perm, d_keys}, [
        ctx, d_perm, n, seed, d_keys, elementsPerThread, cubcall_sort
    ](cudaStream_t stream) mutable {
        // Sort by keys
        cudaError_t status = cubcall_sort.Call(stream);
        return status == cudaSuccess;
    });
    if (!success) return false;

    success = graph->Free(d_keys);
    if (!success) return false;

    return true;
}

#ifdef __cplusplus
}
#endif

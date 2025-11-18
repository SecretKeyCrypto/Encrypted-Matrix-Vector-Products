#include "cudalpn.h"
#include <cuda_runtime.h>
#include <cstdint>

#include "../dataobjects/cudacommon.h"
#include "../dataobjects/cudagraph.h"

// Assumptions:
// - All arrays are on device.
// - params.M_1 divides params.M.
// - message is not needed as a global array; we use shared memory per block.
// - rlcMatrix has params.L * params.K elements (row-major: K rows, L cols).
// - All calculations are modulo params.P (prime).
// - N = K + L (each encoded row has length L followed by K).
// - encoded is treated as a 3D tensor flattened as: encoded[t*entryPerSlice + i*N + j],
//   where t in [0..ECCLength-1], i in [0..rowPerSlice-1], j in [0..N-1],
//   entryPerSlice = rowPerSlice * N, rowPerSlice = M / M_1.
//
// Design:
// - Kernel 1 (copy + RLC): heavy stage; large L, moderate K. Uses elementsPerThread.
//   Each block processes one (i, j0) with j0 in [0..M_1-1]. Writes a full row of length N.
// - Kernel 2 (ECC): light stage; small M_1. For each (i, j), gathers across t, mat-vec with
//   generatorMatrix, then scatters parity for t in [M_1..ECCLength-1].
//
// Note: Adjust block sizes and elementsPerThread based on profiling.

#include <cuda_runtime.h>
#include <stdint.h>

// Kernel 1: Copy input row into encoded and compute K outputs via mat-vec with rlcMatrix.
// One block per (i, j0) where i in [0..rowPerSlice-1], j0 in [0..M_1-1].
__global__ void kernel_copy_rlc_ept(
    const uint32_t* __restrict__ input,    // [M * L]
    const uint32_t* __restrict__ rlc,      // [K * L], row-major
    uint32_t* __restrict__ encoded,        // [ECCLength * rowPerSlice * N], N = L + K
    uint32_t M,
    uint32_t L,
    uint32_t K,
    uint32_t N,
    uint32_t M_1,
    uint32_t rowPerSlice,
    uint32_t entryPerSlice,
    uint32_t P,
    uint32_t /*elementsPerThread*/         // unused for correctness-first version
) {
    extern __shared__ uint32_t shRow[]; // size >= L

    const uint32_t bid = blockIdx.x;    // 0..rowPerSlice*M_1-1
    const uint32_t i   = bid / M_1;
    const uint32_t j0  = bid % M_1;
    if (i >= rowPerSlice || j0 >= M_1) return;

    const uint32_t inputStart  = (i * M_1 + j0) * L;
    const uint32_t outputStart = j0 * entryPerSlice + i * N;

    // 1) Cache the input row in shared memory
    for (uint32_t x = threadIdx.x; x < L; x += blockDim.x) {
        shRow[x] = input[inputStart + x];
    }
    __syncthreads();

    // 2) Copy cached row to encoded (systematic L)
    for (uint32_t x = threadIdx.x; x < L; x += blockDim.x) {
        encoded[outputStart + x] = shRow[x];
    }
    __syncthreads();

    // 3) Compute K RLC outputs: full dot-product per kIdx
    for (uint32_t kIdx = threadIdx.x; kIdx < K; kIdx += blockDim.x) {
        const uint32_t* rlcRow = &rlc[kIdx * L];
        uint64_t acc = 0;
        for (uint32_t c = 0; c < L; ++c) {
            acc += (uint64_t)rlcRow[c] * (uint64_t)shRow[c];
        }
        encoded[outputStart + L + kIdx] = (uint32_t)(acc % P);
    }
}

// Helper for ceiling division: ceil(a / b) for unsigned
__device__ __forceinline__ uint32_t ceil_div_u32(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

// Kernel 2: ECC per (i, j). Gather across t in [0..M_1-1],
// compute parity rows via generatorMatrix (rows 0..ECCLength-M_1-1),
// then scatter parity for u in [M_1..ECCLength-1].
__global__ void kernel_ecc_ept(
    const uint32_t* __restrict__ generator, // [ECCLength * M_1], row-major
    uint32_t* __restrict__ encoded,         // [ECCLength * rowPerSlice * N]
    uint32_t /*L*/,
    uint32_t /*K*/,
    uint32_t N,             // N = L + K
    uint32_t M_1,
    uint32_t ECCLength,
    uint32_t rowPerSlice,
    uint32_t entryPerSlice,
    uint32_t P,
    uint32_t /*elementsPerThread*/
) {
    extern __shared__ uint32_t sh[];
    uint32_t* message = sh; // size >= max(ECCLength, M_1)

    const uint32_t bid = blockIdx.x;        // 0..rowPerSlice*N-1
    const uint32_t i   = bid / N;
    const uint32_t j   = bid % N;
    if (i >= rowPerSlice || j >= N) return;

    const uint64_t entryPerSlice64 = (uint64_t)entryPerSlice;
    const uint64_t base64          = (uint64_t)i * (uint64_t)N + (uint64_t)j;

    // ---- Gather M_1 original symbols across slices
    for (uint32_t t = threadIdx.x; t < M_1; t += blockDim.x) {
        const uint64_t off64 = (uint64_t)t * entryPerSlice64 + base64;
        message[t] = encoded[off64];
    }
    __syncthreads();

    // ---- Compute parity-only for u in [M_1..ECCLength-1]
    for (uint32_t u = threadIdx.x + M_1; u < ECCLength; u += blockDim.x) {
        const uint32_t gr = u - M_1;                 // CPU uses generator rows 0..ECCLength-M_1-1
        const uint32_t* gRow = &generator[(uint64_t)gr * (uint64_t)M_1];
        uint64_t acc = 0;
        for (uint32_t t = 0; t < M_1; ++t) {
            acc += (uint64_t)gRow[t] * (uint64_t)message[t];
        }
        message[u] = (uint32_t)(acc % P);
    }
    __syncthreads();

    // ---- Scatter parity back across slices (systematic ECC)
    for (uint32_t u = threadIdx.x + M_1; u < ECCLength; u += blockDim.x) {
        const uint64_t off64 = (uint64_t)u * entryPerSlice64 + base64;
        encoded[off64] = message[u];
    }
}

#ifdef __cplusplus
extern "C" {
#endif

// Host-side launcher combining both kernels in sequence.
// N = L + K must be provided consistently.
// Choose elementsPerThread (e.g., 4 or 8) based on profiling.
bool CudaLpnEncode(
    DoContext* ctx,
    const uint32_t* input,            // [M * L]
    const uint32_t* rlcMatrix,        // [K * L]
    const uint32_t* generatorMatrix,  // [ECCLength * M_1]
    uint32_t* encoded,                // [ECCLength * rowPerSlice * N]
    uint32_t M, uint32_t L, uint32_t K,
    uint32_t M_1, uint32_t ECCLength,
    uint32_t P
) {
    CudaVectorGraph* graph = CudaVectorGraph::Get(ctx);
    if (!graph) return false;

    int elementsPerThread = graph->GetParams().elementsPerThread;

    // Derived quantities
    uint32_t N = K + L;
    uint32_t rowPerSlice   = M / M_1;
    uint32_t entryPerSlice = rowPerSlice * N;

    bool success;

    success = graph->Compute({input, rlcMatrix, generatorMatrix}, {encoded}, [
        input, rlcMatrix, generatorMatrix, encoded,
        M, L, K, N, M_1, ECCLength, P,
        rowPerSlice, entryPerSlice, elementsPerThread
    ](cudaStream_t stream) {
        // Kernel 1 configuration (heavy stage: copy + RLC)
        {
            uint64_t blocksPerGridX   = (uint64_t)rowPerSlice * (uint64_t)M_1;
            uint64_t threadsPerBlockX = defaultMaxThreads; // e.g., 256
            size_t   sharedMemSize    = sizeof(uint32_t) * (size_t)L;

            dim3 grid(blocksPerGridX);
            dim3 block(threadsPerBlockX);

            kernel_copy_rlc_ept<<<grid, block, sharedMemSize, stream>>>(
                input, rlcMatrix, encoded,
                M, L, K, N, M_1, rowPerSlice, entryPerSlice, P, /*elementsPerThread*/ 0
            );
            if (cudaGetLastError() != cudaSuccess) {
                return false;
            }
        }

        // Kernel 2 configuration (light stage: ECC)
        {
            uint64_t blocksPerGridX   = (uint64_t)rowPerSlice * (uint64_t)N;
            uint64_t threadsPerBlockX = defaultMaxThreads; // e.g., 256
            size_t   sharedMemSize    = sizeof(uint32_t) * (size_t)max(ECCLength, M_1);

            dim3 grid(blocksPerGridX);
            dim3 block(threadsPerBlockX);

            kernel_ecc_ept<<<grid, block, sharedMemSize, stream>>>(
                generatorMatrix, encoded,
                /*L*/ 0, /*K*/ 0, N, M_1, ECCLength, rowPerSlice, entryPerSlice, P, /*elementsPerThread*/ 0
            );
            if (cudaGetLastError() != cudaSuccess) {
                return false;
            }
        }
        return true;
    });
    if (!success) {
        return false;
    }

    return true;
}

#ifdef __cplusplus
} // extern "C"
#endif

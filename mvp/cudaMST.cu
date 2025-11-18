#include <cassert>

#include "cudaMST.h"
#include "../dataobjects/cudacommon.h"
#include "../dataobjects/cudagraph.h"

__global__ void TransformRowMajorToBlockRowMajorKernel(
    const uint32_t* __restrict__ mat,   // device input: size n × m, row-major
    uint32_t ms,
    uint32_t* __restrict__ matBlocked,  // device output: size n × m, block-row-major
    uint32_t bs,
    uint32_t n, uint32_t m, uint32_t s, uint32_t b,
    uint32_t elementsPerThread
) {
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t colStart = (blockIdx.x * blockDim.x + threadIdx.x) * elementsPerThread;

    if (row >= n) return;

    // Initialize once before the loop
    uint32_t col = colStart;
    uint32_t blk = col / b;
    uint32_t j   = col - (blk * b);

    for (uint32_t k = 0; k < elementsPerThread; ++k) {
        if (col < m) {
            uint32_t src_idx = row * ms + col;
            uint32_t val = mat[src_idx];

            uint32_t dest_idx = (blk * bs + row) * b + j;
            matBlocked[dest_idx] = val;
        }

        // Increment for next iteration
        ++col;
        ++j;
        if (j == b) {   // wrap around to next block
            j = 0;
            ++blk;
        }
    }
}

#ifdef __cplusplus
extern "C" {
#endif

// Wrapper: assumes device pointers are passed in
bool CudaTransformRowMajorToBlockRowMajor(
    DoContext* ctx,
    const uint32_t* d_mat,        // device input
    uint32_t mo, uint32_t ms,
    uint32_t* d_matBlocked,       // device output
    uint32_t bo, uint32_t bs,
    uint32_t n, uint32_t m, uint32_t s
) {
    CudaVectorGraph* graph = CudaVectorGraph::Get(ctx);
    if (!graph) return false;

    assert(m % s == 0);
    uint32_t b = m / s;
    uint32_t elementsPerThread = graph->GetParams().elementsPerThread;

    return graph->Compute({d_mat}, {d_matBlocked}, [d_mat, mo, ms, d_matBlocked, bo, bs, n, m, s, b, elementsPerThread](cudaStream_t stream) {
        // Launch configuration
        uint64_t threadsPerBlockX = threadsPerBlockFor(m, elementsPerThread, defaultMaxThreads2D);
        uint64_t threadsPerBlockY = threadsPerBlockFor(n, 1, defaultMaxThreads2D);
        uint64_t blocksPerGridX = blocksPerGridFor(m, threadsPerBlockX, elementsPerThread);
        uint64_t blocksPerGridY = blocksPerGridFor(n, threadsPerBlockY, 1);

        dim3 block(threadsPerBlockX, threadsPerBlockY);
        dim3 grid(blocksPerGridX, blocksPerGridY);

        TransformRowMajorToBlockRowMajorKernel<<<grid, block, 0, stream>>>(
            d_mat + mo, ms, d_matBlocked + bo, bs, n, m, s, b, elementsPerThread
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

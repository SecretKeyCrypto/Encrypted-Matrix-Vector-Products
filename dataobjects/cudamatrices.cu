#include "cudamatrices.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "cudacommon.h"
#include "cudagraph.h"

__global__ void CudaMatrixTransposeKernel(
    uint32_t* output,
    const uint32_t* input,
    uint32_t M,
    uint32_t N,
    int dimElementsPerThread)
{
    // Each thread covers a square tile of size dimElementsPerThread × dimElementsPerThread
    uint32_t rowStart = (blockIdx.y * blockDim.y + threadIdx.y) * dimElementsPerThread;
    uint32_t colStart = (blockIdx.x * blockDim.x + threadIdx.x) * dimElementsPerThread;

    for (int dy = 0; dy < dimElementsPerThread; ++dy) {
        uint32_t row = rowStart + dy;
        if (row < M) {
            for (int dx = 0; dx < dimElementsPerThread; ++dx) {
                uint32_t col = colStart + dx;
                if (col < N) {
                    output[col * M + row] = input[row * N + col];
                }
            }
        }
    }
}

#ifdef __cplusplus
extern "C" {
#endif

bool CudaMatrixTranspose(DoContext* ctx, uint32_t* result, uint32_t ro, const uint32_t* matrix, uint32_t mo, uint32_t M, uint32_t N) {
    CudaVectorGraph* graph = CudaVectorGraph::Get(ctx);
    if (!graph) return false;

    int dimElementsPerThread = graph->GetParams().dimElementsPerThread;

    return graph->Compute({matrix}, {result}, [result = result + ro, matrix = matrix + mo, M, N, dimElementsPerThread](cudaStream_t stream) {
        // Threads per block along X and Y
        uint64_t threadsPerBlockX = threadsPerBlockFor(N, dimElementsPerThread, 16);
        uint64_t threadsPerBlockY = threadsPerBlockFor(M, dimElementsPerThread, 16);

        // Blocks per grid along X and Y
        uint64_t blocksPerGridX = blocksPerGridFor(N, threadsPerBlockX, dimElementsPerThread);
        uint64_t blocksPerGridY = blocksPerGridFor(M, threadsPerBlockY, dimElementsPerThread);

        dim3 blockDim(threadsPerBlockX, threadsPerBlockY);
        dim3 gridDim(blocksPerGridX, blocksPerGridY);

        CudaMatrixTransposeKernel<<<gridDim, blockDim, 0, stream>>>(result, matrix, M, N, dimElementsPerThread);
        if (_CUDA_CHECK(cudaGetLastError()) != cudaSuccess) {
            return false;
        }
        return true;
    });
}

#ifdef __cplusplus
}
#endif

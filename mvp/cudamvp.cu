#include "cudamvp.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cassert>

#include "../dataobjects/cudacommon.h"
#include "../dataobjects/cudagraph.h"

__global__ void GeneralMatVecKernelExt(
    const uint32_t* __restrict__ mat,
    uint32_t ms,
    const uint32_t* __restrict__ vec,
    uint32_t vs,
    uint32_t* __restrict__ result,
    uint32_t rs,
    uint32_t n,
    uint32_t m,
    uint32_t b,
    uint32_t steps,
    uint32_t p,
    bool blockAlongColumns,
    int elementsPerThread
) {
    uint32_t step = blockIdx.y;
    if (step >= steps) return;

    uint32_t outLen = blockAlongColumns ? n : m;
    uint32_t baseOutIdx = (blockIdx.x * blockDim.x + threadIdx.x) * elementsPerThread;

    const uint32_t* mat_step = mat + step * ms;
    const uint32_t* vec_step = vec + step * vs;
    uint32_t*       res_step = result + step * rs;

    for (int e = 0; e < elementsPerThread; ++e) {
        uint32_t outIdx = baseOutIdx + e;
        if (outIdx >= outLen) break;

        uint64_t acc = 0;
        if (blockAlongColumns) {
            const uint32_t* row_ptr = mat_step + size_t(outIdx) * b;
            #pragma unroll 4
            for (uint32_t k = 0; k < b; ++k) {
                acc += uint64_t(row_ptr[k]) * uint64_t(vec_step[k]);
            }
            res_step[outIdx] = uint32_t(acc % p);
        } else {
            #pragma unroll 4
            for (uint32_t i = 0; i < b; ++i) {
                const uint32_t* row_ptr = mat_step + size_t(i) * m;
                acc += uint64_t(row_ptr[outIdx]) * uint64_t(vec_step[i]);
            }
            res_step[outIdx] = uint32_t(acc % p);
        }
    }
}

bool launchGeneralMatVecKernelExt(
    const uint32_t* mat, uint32_t ms,
    const uint32_t* vec, uint32_t vs,
    uint32_t* result, uint32_t rs,
    uint32_t n, uint32_t m, uint32_t b, uint32_t steps,
    uint32_t p, bool blockAlongColumns,
    int elementsPerThread,
    cudaStream_t stream = 0
) {
    uint32_t outLen = blockAlongColumns ? n : m;

    uint64_t threadsX = threadsPerBlockFor(outLen, elementsPerThread);
    uint64_t blocksX  = blocksPerGridFor(outLen, threadsX, elementsPerThread);

    dim3 block(threadsX);
    dim3 grid(blocksX, steps);

    GeneralMatVecKernelExt<<<grid, block, 0, stream>>>(
        mat, ms, vec, vs, result, rs,
        n, m, b, steps, p, blockAlongColumns, elementsPerThread
    );
    if (_CUDA_CHECK(cudaGetLastError()) != cudaSuccess) {
        return false;
    }
    return true;
}

bool CudaMatVecProduct(DoContext* ctx,
                       const uint32_t* mat,
                       const uint32_t* vec,
                       uint32_t* result,
                       uint32_t n, uint32_t m, uint32_t p)
{
    CudaVectorGraph* graph = CudaVectorGraph::Get(ctx);
    if (!graph) return false;

    int elementsPerThread = graph->GetParams().mvpElementsPerThread;

    return graph->Compute({mat, vec}, {result}, [mat, vec, result, n, m, p, elementsPerThread](cudaStream_t stream) {
        return launchGeneralMatVecKernelExt(
            mat, 0, vec, 0, result, 0, n, m, /*b=*/m, /*steps=*/1, p, /*blockAlongColumns=*/true, elementsPerThread, stream
        );
    });
}

bool CudaMatVecProductExt(
    DoContext* ctx,
    const uint32_t* mat, uint32_t mo, uint32_t ms,
    const uint32_t* vec, uint32_t vo, uint32_t vs,
    uint32_t* result, uint32_t ro, uint32_t rs,
    uint32_t n, uint32_t m, uint32_t steps, uint32_t p
) {
    CudaVectorGraph* graph = CudaVectorGraph::Get(ctx);
    if (!graph) return false;

    int elementsPerThread = graph->GetParams().mvpElementsPerThread;

    return graph->Compute({mat, vec}, {result}, [mat = mat + mo, ms, vec = vec + vo, vs, result = result + ro, rs, steps, n, m, p, elementsPerThread](cudaStream_t stream) {
        return launchGeneralMatVecKernelExt(
            mat, ms, vec, vs, result, rs, n, m, /*b=*/m, steps, p, /*blockAlongColumns=*/true, elementsPerThread, stream
        );
    });
}

bool CudaBlockMatVecProduct(DoContext* ctx,
                            const uint32_t* mat,
                            const uint32_t* vec,
                            uint32_t* result,
                            uint32_t n, uint32_t m,
                            uint32_t s, uint32_t p)
{
    CudaVectorGraph* graph = CudaVectorGraph::Get(ctx);
    if (!graph) return false;

    int elementsPerThread = graph->GetParams().mvpElementsPerThread;

    assert(m % s == 0);
    return graph->Compute({mat, vec}, {result}, [mat, vec, result, n, m, s, p, elementsPerThread](cudaStream_t stream) {
        uint32_t b = m / s;
        return launchGeneralMatVecKernelExt(
            mat, /*ms=*/n*b, vec, /*vs=*/b, result, /*rs=*/n, n, m, b, /*steps=*/s, p, /*blockAlongColumns=*/true, elementsPerThread, stream
        );
    });
}

bool CudaBlockVecMatProduct(DoContext* ctx,
                            const uint32_t* mat,
                            const uint32_t* vec,
                            uint32_t* result,
                            uint32_t n, uint32_t m,
                            uint32_t s, uint32_t p)
{
    CudaVectorGraph* graph = CudaVectorGraph::Get(ctx);
    if (!graph) return false;

    int elementsPerThread = graph->GetParams().mvpElementsPerThread;

    assert(n % s == 0);
    return graph->Compute({mat, vec}, {result}, [mat, vec, result, n, m, s, p, elementsPerThread](cudaStream_t stream) {
        uint32_t b = n / s;
        return launchGeneralMatVecKernelExt(
            mat, /*ms=*/b*m, vec, /*vs=*/b, result, /*rs=*/m, n, m, b, /*steps=*/s, p, /*blockAlongColumns=*/false, elementsPerThread, stream
        );
    });
}

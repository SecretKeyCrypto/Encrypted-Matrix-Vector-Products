#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <stdexcept>
#include <cstddef>

#include "docontextimpl.h"
#include "cudacommon.h"
#include "cudagraph.h"
#include "cudafields.h"
#include "hdcommon.h"

__global__ void CudaFieldRangeVectorKernel(uint32_t* r, uint32_t start, uint64_t length, int elementsPerThread) {
    size_t startIdx = (blockIdx.x * blockDim.x + threadIdx.x) * elementsPerThread;
    for (int j = 0; j < elementsPerThread; ++j) {
        size_t idx = startIdx + j;
        if (idx < length) {
            r[idx] = start + idx;
        }
    }
}

__global__ void CudaFieldCopyVectorKernel(uint32_t* r, const uint32_t* a, uint64_t length, int elementsPerThread) {
    size_t startIdx = (blockIdx.x * blockDim.x + threadIdx.x) * elementsPerThread;
    for (int j = 0; j < elementsPerThread; ++j) {
        size_t idx = startIdx + j;
        if (idx < length) {
            r[idx] = a[idx];
        }
    }
}

__global__ void CudaFieldSetVectorKernel(uint32_t* r, uint64_t length, uint32_t v, int elementsPerThread) {
    size_t startIdx = (blockIdx.x * blockDim.x + threadIdx.x) * elementsPerThread;
    for (int j = 0; j < elementsPerThread; ++j) {
        size_t idx = startIdx + j;
        if (idx < length) {
            r[idx] = v;
        }
    }
}

__global__ void CudaFieldAddToVectorKernel(uint32_t* r, uint32_t v, uint64_t length, int elementsPerThread) {
    size_t startIdx = (blockIdx.x * blockDim.x + threadIdx.x) * elementsPerThread;
    for (int j = 0; j < elementsPerThread; ++j) {
        size_t idx = startIdx + j;
        if (idx < length) {
            r[idx] += v;
        }
    }
}

__global__ void CudaFieldAddVectorsKernel(uint32_t* r, const uint32_t* a, const uint32_t* b, uint64_t length, uint32_t p, int elementsPerThread) {
    size_t startIdx = (blockIdx.x * blockDim.x + threadIdx.x) * elementsPerThread;
    for (int j = 0; j < elementsPerThread; ++j) {
        size_t idx = startIdx + j;
        if (idx < length) {
            uint32_t sum = a[idx] + b[idx];
            r[idx] = sum >= p ? sum - p : sum;
        }
    }
}

__global__ void CudaFieldMulVectorKernel(uint32_t* r, const uint32_t* a, uint32_t b, uint64_t length, uint32_t p, int elementsPerThread) {
    size_t startIdx = (blockIdx.x * blockDim.x + threadIdx.x) * elementsPerThread;
    for (int j = 0; j < elementsPerThread; ++j) {
        size_t idx = startIdx + j;
        if (idx < length) {
            uint64_t product = static_cast<uint64_t>(a[idx]) * b;
            r[idx] = static_cast<uint32_t>(product % p);
        }
    }
}

__global__ void CudaFieldMulVectorsKernel(uint32_t* r, const uint32_t* a, const uint32_t* b, uint64_t length, uint32_t p, int elementsPerThread) {
    size_t startIdx = (blockIdx.x * blockDim.x + threadIdx.x) * elementsPerThread;
    for (int j = 0; j < elementsPerThread; ++j) {
        size_t idx = startIdx + j;
        if (idx < length) {
            uint64_t product = static_cast<uint64_t>(a[idx]) * static_cast<uint64_t>(b[idx]);
            r[idx] = static_cast<uint32_t>(product % p);
        }
    }
}

__global__ void CudaFieldSubVectorsKernel(uint32_t* r, const uint32_t* a, const uint32_t* b, uint64_t length, uint32_t p, int elementsPerThread) {
    size_t startIdx = (blockIdx.x * blockDim.x + threadIdx.x) * elementsPerThread;
    for (int j = 0; j < elementsPerThread; ++j) {
        size_t idx = startIdx + j;
        if (idx < length) {
            uint32_t av = a[idx];
            uint32_t bv = b[idx];
            r[idx] = (av >= bv) ? (av - bv) : (p - (bv - av));
        }
    }
}

__global__ void CudaFieldNegVectorsKernel(uint32_t* r, uint64_t length, uint32_t p, int elementsPerThread) {
    size_t startIdx = (blockIdx.x * blockDim.x + threadIdx.x) * elementsPerThread;
    for (int j = 0; j < elementsPerThread; ++j) {
        size_t idx = startIdx + j;
        if (idx < length) {
            uint32_t val = r[idx];
            r[idx] = (val == 0) ? 0 : (p - val);
        }
    }
}

__global__ void CudaFieldIsZeroVectorKernel(bool* t, const uint32_t* e, uint64_t length, int elementsPerThread) {
    __shared__ bool has_non_zero;
    if (threadIdx.x == 0) {
        has_non_zero = false;
    }
    __syncthreads();

    size_t startIdx = (blockIdx.x * blockDim.x + threadIdx.x) * elementsPerThread;
    for (int j = 0; j < elementsPerThread; ++j) {
        size_t idx = startIdx + j;
        if (idx < length && e[idx] != 0) {
            atomicExch((int*)&has_non_zero, true);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        *t = !has_non_zero;
    }
}

__global__ void CudaFieldAddVectorIfNonZeroKernel(bool* t, uint32_t* r, const uint32_t* e, uint64_t length, uint32_t p, int elementsPerThread) {
    __shared__ bool has_non_zero;
    if (threadIdx.x == 0) {
        has_non_zero = false;
    }
    __syncthreads();

    size_t startIdx = (blockIdx.x * blockDim.x + threadIdx.x) * elementsPerThread;
    for (int j = 0; j < elementsPerThread; ++j) {
        size_t idx = startIdx + j;
        if (idx < length && e[idx] != 0) {
            atomicExch((int*)&has_non_zero, true);
        }
    }
    __syncthreads();

    if (has_non_zero) {
        for (int j = 0; j < elementsPerThread; ++j) {
            size_t idx = startIdx + j;
            r[idx] += e[idx];
        }
    }

    if (threadIdx.x == 0) {
        *t = has_non_zero;
    }
}

__device__ __forceinline__ uint32_t modmul_u32_dev(uint32_t a, uint32_t b, uint32_t p) {
    unsigned long long prod = (unsigned long long)a * (unsigned long long)b;
    return (uint32_t)(prod % p);
}

__device__ __forceinline__ uint32_t modexp_u32_dev(uint32_t base, uint32_t exp, uint32_t p) {
    uint32_t result = 1;
    uint32_t x = base % p;
    while (exp) {
        if (exp & 1) result = modmul_u32_dev(result, x, p);
        x = modmul_u32_dev(x, x, p);
        exp >>= 1;
    }
    return result;
}

__global__ void CudaFieldInvVectorKernel(uint32_t* r, const uint32_t* a, size_t n, uint32_t p, int elementsPerThread) {
    size_t startIdx = (blockIdx.x * blockDim.x + threadIdx.x) * elementsPerThread;
    for (int j = 0; j < elementsPerThread; ++j) {
        size_t idx = startIdx + j;
        if (idx >= n) return;

        if (a[idx] == 0) {
            r[idx] = 0;
        } else {
            uint32_t inv = modexp_u32_dev(a[idx], p - 2, p);
            r[idx] = inv;
        }
    }
}

// Ext kernels

__global__ void CudaFieldAddVectorsExtKernel(
    uint32_t* r, uint32_t rs,
    const uint32_t* a, uint32_t as,
    const uint32_t* b, uint32_t bs,
    uint64_t length, uint64_t steps, uint32_t p, int elementsPerThread
) {
    uint64_t total = steps * length;

    uint64_t tid   = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    uint64_t start = tid * elementsPerThread;

    if (start >= total) return;

    // Compute initial row and column once
    uint64_t row = start / length;
    uint64_t col = start - (row * length);

    for (int t = 0; t < elementsPerThread; ++t) {
        uint64_t gidx = start + t;
        if (gidx >= total) break;

        // Compute base indices with respective strides
        uint64_t r_idx = row * rs + col;
        uint64_t a_idx = row * as + col;
        uint64_t b_idx = row * bs + col;

        uint32_t sum = a[a_idx] + b[b_idx];
        r[r_idx] = (sum >= p) ? (sum - p) : sum;

        // Advance column; wrap to next row when needed
        ++col;
        if (col == length) {
            col = 0;
            ++row;
        }
    }
}

__global__ void CudaFieldMulVectorExtKernel(uint32_t* r, const uint32_t* a, uint32_t b, uint64_t length, uint64_t stride, uint64_t steps, uint32_t p, int elementsPerThread) {
    uint64_t total = (uint64_t)steps * length;

    uint64_t tid   = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t start = tid * (uint64_t)elementsPerThread;

    if (start >= total) return;

    // Compute initial row and column once
    uint64_t row = start / length;
    uint64_t col = start - (row * length);

    for (uint64_t t = 0; t < (uint64_t)elementsPerThread; ++t) {
        uint64_t gidx = start + t;
        if (gidx >= total) break;

        uint64_t base = row * stride + col;

        uint64_t product = (uint64_t)a[base] * (uint64_t)b;
        r[base] = (uint32_t)(product % p);

        // Advance column; wrap to next row when needed
        ++col;
        if (col == length) {
            col = 0;
            ++row;
        }
    }
}

__global__ void CudaFieldMulVectorsExtKernel(uint32_t* r,
                                             uint64_t rs,
                                             const uint32_t* a,
                                             uint64_t as,
                                             const uint32_t* b,
                                             uint64_t bs,
                                             uint64_t length,
                                             uint64_t steps,
                                             uint32_t p,
                                             int elementsPerThread) {
    uint64_t total = steps * length;

    uint64_t tid   = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t start = tid * (uint64_t)elementsPerThread;

    if (start >= total) return;

    // Compute initial row and column once
    uint64_t row = start / length;
    uint64_t col = start - (row * length);

    for (uint64_t t = 0; t < (uint64_t)elementsPerThread; ++t) {
        uint64_t gidx = start + t;
        if (gidx >= total) break;

        uint64_t r_base = row * rs + col;
        uint64_t a_base = row * as + col;
        uint64_t b_base = row * bs + col;

        uint64_t product = (uint64_t)a[a_base] * (uint64_t)b[b_base];
        r[r_base] = (uint32_t)(product % p);

        // Advance column; wrap to next row when needed
        ++col;
        if (col == length) {
            col = 0;
            ++row;
        }
    }
}

__global__ void CudaFieldNegVectorsExtKernel(uint32_t* r, uint64_t length, uint64_t stride, uint64_t steps, uint32_t p, int elementsPerThread) {
    uint64_t total = (uint64_t)steps * length;

    uint64_t tid   = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t start = tid * (uint64_t)elementsPerThread;

    if (start >= total) return;

    // Compute initial row and column once
    uint64_t row = start / length;
    uint64_t col = start - (row * length);

    for (uint64_t t = 0; t < (uint64_t)elementsPerThread; ++t) {
        uint64_t gidx = start + t;
        if (gidx >= total) break;

        uint64_t base = row * stride + col;

        uint32_t val = r[base];
        r[base] = (val == 0) ? 0 : (p - val);

        // Advance column; wrap to next row when needed
        ++col;
        if (col == length) {
            col = 0;
            ++row;
        }
    }
}

__global__ void CudaFieldAddVectorIfNonZeroExtKernel(
    bool* t, uint64_t ts,
    uint32_t* r, uint64_t rs,
    const uint32_t* e, uint64_t es,
    uint64_t length, uint64_t steps,
    uint32_t p, int elementsPerThread)
{
    // Step index is mapped to gridDim.y
    uint64_t step = blockIdx.y;
    if (step >= steps) return;

    // Compute per-step base pointers
    bool* t_step = t + step * ts;
    uint32_t* r_step = r + step * rs;
    const uint32_t* e_step = e + step * es;

    // Shared flag for this block
    __shared__ int has_non_zero;
    if (threadIdx.x == 0) {
        has_non_zero = 0;
    }
    __syncthreads();

    // Compute element index within this step
    size_t startIdx = (blockIdx.x * blockDim.x + threadIdx.x) * elementsPerThread;
    for (int j = 0; j < elementsPerThread; ++j) {
        size_t idx = startIdx + j;
        if (idx < length && e_step[idx] != 0) {
            atomicExch(&has_non_zero, 1);
        }
    }
    __syncthreads();

    if (has_non_zero) {
        for (int j = 0; j < elementsPerThread; ++j) {
            size_t idx = startIdx + j;
            r_step[idx] += e_step[idx];
        }
    }

    // One thread per block writes result and launches addition if needed
    if (threadIdx.x == 0) {
        *t_step = has_non_zero != 0;
    }
}

#ifdef __cplusplus
extern "C" {
#endif

// Host wrappers
bool CudaFieldRangeVector(DoContext* ctx, uint32_t* r, uint64_t ro, uint32_t start, uint64_t length) {
    CudaVectorGraph* graph = CudaVectorGraph::Get(ctx);
    if (!graph) return false;

    int elementsPerThread = graph->GetParams().elementsPerThread;
    return graph->Compute({}, {r}, [r = r + ro, start, length, elementsPerThread](cudaStream_t stream) {
        uint64_t threadsPerBlock = threadsPerBlockFor(length, elementsPerThread);
        uint64_t blocksPerGrid = blocksPerGridFor(length, threadsPerBlock, elementsPerThread);
        CudaFieldRangeVectorKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(r, start, length, elementsPerThread);
        if (_CUDA_CHECK(cudaGetLastError()) != cudaSuccess) {
            return false;
        }
        return true;
    });
}

bool CudaFieldCopyVector(DoContext* ctx, uint32_t* r, uint64_t ro, const uint32_t* a, uint64_t ao, uint64_t length) {
    CudaVectorGraph* graph = CudaVectorGraph::Get(ctx);
    if (!graph) return false;

    int elementsPerThread = graph->GetParams().elementsPerThread;
    return graph->Compute({a}, {r}, [r = r + ro, a = a + ao, length, elementsPerThread](cudaStream_t stream) {
        uint64_t threadsPerBlock = threadsPerBlockFor(length, elementsPerThread);
        uint64_t blocksPerGrid = blocksPerGridFor(length, threadsPerBlock, elementsPerThread);
        CudaFieldCopyVectorKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(r, a, length, elementsPerThread);
        if (_CUDA_CHECK(cudaGetLastError()) != cudaSuccess) {
            return false;
        }
        return true;
    });
}

bool CudaFieldSetVector(DoContext* ctx, uint32_t* r, uint64_t ro, uint64_t length, uint32_t v) {
    CudaVectorGraph* graph = CudaVectorGraph::Get(ctx);
    if (!graph) return false;

    int elementsPerThread = graph->GetParams().elementsPerThread;
    return graph->Compute({}, {r}, [r = r + ro, length, v, elementsPerThread](cudaStream_t stream) {
        uint64_t threadsPerBlock = threadsPerBlockFor(length, elementsPerThread);
        uint64_t blocksPerGrid = blocksPerGridFor(length, threadsPerBlock, elementsPerThread);
        CudaFieldSetVectorKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(r, length, v, elementsPerThread);
        if (_CUDA_CHECK(cudaGetLastError()) != cudaSuccess) {
            return false;
        }
        return true;
    });
}

bool CudaFieldAddToVector(DoContext* ctx, uint32_t* r, uint64_t ro, uint32_t v, uint64_t length) {
    CudaVectorGraph* graph = CudaVectorGraph::Get(ctx);
    if (!graph) return false;

    int elementsPerThread = graph->GetParams().elementsPerThread;
    return graph->Compute({r}, {r}, [r = r + ro, v, length, elementsPerThread](cudaStream_t stream) {
        uint64_t threadsPerBlock = threadsPerBlockFor(length, elementsPerThread);
        uint64_t blocksPerGrid = blocksPerGridFor(length, threadsPerBlock, elementsPerThread);
        CudaFieldAddToVectorKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(r, v, length, elementsPerThread);
        if (_CUDA_CHECK(cudaGetLastError()) != cudaSuccess) {
            return false;
        }
        return true;
    });
}

bool CudaFieldAddVectors(DoContext* ctx, uint32_t* r, uint64_t ro, const uint32_t* a, uint64_t ao, const uint32_t* b, uint64_t bo, uint64_t length, uint32_t p) {
    CudaVectorGraph* graph = CudaVectorGraph::Get(ctx);
    if (!graph) return false;

    int elementsPerThread = graph->GetParams().elementsPerThread;
    return graph->Compute({a, b}, {r}, [r = r + ro, a = a + ao, b = b + bo, length, p, elementsPerThread](cudaStream_t stream) {
        uint64_t threadsPerBlock = threadsPerBlockFor(length, elementsPerThread);
        uint64_t blocksPerGrid = blocksPerGridFor(length, threadsPerBlock, elementsPerThread);
        CudaFieldAddVectorsKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(r, a, b, length, p, elementsPerThread);
        if (_CUDA_CHECK(cudaGetLastError()) != cudaSuccess) {
            return false;
        }
        return true;
    });
}

bool CudaFieldMulVector(DoContext* ctx, uint32_t* r, uint64_t ro, const uint32_t* a, uint64_t ao, uint32_t b, uint64_t length, uint32_t p) {
    CudaVectorGraph* graph = CudaVectorGraph::Get(ctx);
    if (!graph) return false;

    int elementsPerThread = graph->GetParams().elementsPerThread;
    return graph->Compute({a}, {r}, [r = r + ro, a = a + ao, b, length, p, elementsPerThread](cudaStream_t stream) {
        uint64_t threadsPerBlock = threadsPerBlockFor(length, elementsPerThread);
        uint64_t blocksPerGrid = blocksPerGridFor(length, threadsPerBlock, elementsPerThread);
        CudaFieldMulVectorKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(r, a, b, length, p, elementsPerThread);
        if (_CUDA_CHECK(cudaGetLastError()) != cudaSuccess) {
            return false;
        }
        return true;
    });
}

bool CudaFieldMulVectors(DoContext* ctx, uint32_t* r, uint64_t ro, const uint32_t* a, uint64_t ao, const uint32_t* b, uint64_t bo, uint64_t length, uint32_t p) {
    CudaVectorGraph* graph = CudaVectorGraph::Get(ctx);
    if (!graph) return false;

    int elementsPerThread = graph->GetParams().elementsPerThread;
    return graph->Compute({a, b}, {r}, [r = r + ro, a = a + ao, b = b + bo, length, p, elementsPerThread](cudaStream_t stream) {
        uint64_t threadsPerBlock = threadsPerBlockFor(length, elementsPerThread);
        uint64_t blocksPerGrid = blocksPerGridFor(length, threadsPerBlock, elementsPerThread);
        CudaFieldMulVectorsKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(r, a, b, length, p, elementsPerThread);
        if (_CUDA_CHECK(cudaGetLastError()) != cudaSuccess) {
            return false;
        }
        return true;
    });
}

bool CudaFieldSubVectors(DoContext* ctx, uint32_t* r, uint64_t ro, const uint32_t* a, uint64_t ao, const uint32_t* b, uint64_t bo, uint64_t length, uint32_t p) {
    CudaVectorGraph* graph = CudaVectorGraph::Get(ctx);
    if (!graph) return false;

    int elementsPerThread = graph->GetParams().elementsPerThread;
    return graph->Compute({a, b}, {r}, [r = r + ro, a = a + ao, b = b + bo, length, p, elementsPerThread](cudaStream_t stream) {
        uint64_t threadsPerBlock = threadsPerBlockFor(length, elementsPerThread);
        uint64_t blocksPerGrid = blocksPerGridFor(length, threadsPerBlock, elementsPerThread);
        CudaFieldSubVectorsKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(r, a, b, length, p, elementsPerThread);
        if (_CUDA_CHECK(cudaGetLastError()) != cudaSuccess) {
            return false;
        }
        return true;
    });
}

bool CudaFieldNegVector(DoContext* ctx, uint32_t* r, uint64_t ro, uint64_t length, uint32_t p) {
    CudaVectorGraph* graph = CudaVectorGraph::Get(ctx);
    if (!graph) return false;

    int elementsPerThread = graph->GetParams().elementsPerThread;
    return graph->Compute({r}, {r}, [r, length, p, elementsPerThread](cudaStream_t stream) {
        uint64_t threadsPerBlock = threadsPerBlockFor(length, elementsPerThread);
        uint64_t blocksPerGrid = blocksPerGridFor(length, threadsPerBlock, elementsPerThread);
        CudaFieldNegVectorsKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(r, length, p, elementsPerThread);
        if (_CUDA_CHECK(cudaGetLastError()) != cudaSuccess) {
            return false;
        }
        return true;
    });
}

bool CudaFieldIsZeroVector(DoContext* ctx, bool* t, const uint32_t* e, uint64_t length) {
    CudaVectorGraph* graph = CudaVectorGraph::Get(ctx);
    if (!graph) return false;

    int elementsPerThread = graph->GetParams().elementsPerThread;
    return graph->Compute({e}, {t}, [t, e, length, elementsPerThread](cudaStream_t stream) {
        uint64_t threadsPerBlock = threadsPerBlockFor(length, elementsPerThread);
        uint64_t blocksPerGrid = blocksPerGridFor(length, threadsPerBlock, elementsPerThread);
        CudaFieldIsZeroVectorKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(t, e, length, elementsPerThread);
        if (_CUDA_CHECK(cudaGetLastError()) != cudaSuccess) {
            return false;
        }
        return true;
    });
}

bool CudaFieldAddVectorIfNonZero(DoContext* ctx, bool* t, uint64_t t_index, uint32_t* r, uint64_t ro, const uint32_t* e, uint64_t eo, uint64_t length, uint32_t p) {
    CudaVectorGraph* graph = CudaVectorGraph::Get(ctx);
    if (!graph) return false;

    int elementsPerThread = graph->GetParams().elementsPerThread;
    return graph->Compute({e}, {t, r}, [t = t + t_index, r = r + ro, e = e + eo, length, p, elementsPerThread](cudaStream_t stream) {
        uint64_t threadsPerBlock = threadsPerBlockFor(length, elementsPerThread);
        uint64_t blocksPerGrid = blocksPerGridFor(length, threadsPerBlock, elementsPerThread);
        CudaFieldAddVectorIfNonZeroKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(t, r, e, length, p, elementsPerThread);
        if (_CUDA_CHECK(cudaGetLastError()) != cudaSuccess) {
            return false;
        }
        return true;
    });
}

bool CudaFieldInvVector(DoContext* ctx, uint32_t* r, uint64_t ro, const uint32_t* a, uint64_t ao, uint64_t length, uint32_t p) {
    CudaVectorGraph* graph = CudaVectorGraph::Get(ctx);
    if (!graph) return false;

    int elementsPerThread = graph->GetParams().elementsPerThread;
    return graph->Compute({a}, {r}, [r = r + ro, a = a + ao, length, p, elementsPerThread](cudaStream_t stream) {
        uint64_t threadsPerBlock = threadsPerBlockFor(length, elementsPerThread);
        uint64_t blocksPerGrid = blocksPerGridFor(length, threadsPerBlock, elementsPerThread);
        CudaFieldInvVectorKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(r, a, length, p, elementsPerThread);
        if (_CUDA_CHECK(cudaGetLastError()) != cudaSuccess) {
            return false;
        }
        return true;
    });
}

// Ext functions

bool CudaFieldAddVectorsExt(
    DoContext* ctx,
    uint32_t* r, uint64_t ro, uint32_t rs,
    const uint32_t* a, uint64_t ao, uint32_t as,
    const uint32_t* b, uint64_t bo, uint32_t bs,
    uint64_t length, uint64_t steps, uint32_t p
) {
    CudaVectorGraph* graph = CudaVectorGraph::Get(ctx);
    if (!graph) return false;

    int elementsPerThread = graph->GetParams().elementsPerThread;
    return graph->Compute({a, b}, {r}, [r = r + ro, rs, a = a + ao, as, b = b + bo, bs, length, steps, p, elementsPerThread](cudaStream_t stream) {
        uint64_t threadsPerBlock = threadsPerBlockFor(length * steps, elementsPerThread);
        uint64_t blocksPerGrid = blocksPerGridFor(length * steps, threadsPerBlock, elementsPerThread);
        CudaFieldAddVectorsExtKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(r, rs, a, as, b, bs, length, steps, p, elementsPerThread);
        if (_CUDA_CHECK(cudaGetLastError()) != cudaSuccess) {
            return false;
        }
        return true;
    });
}

bool CudaFieldMulVectorExt(
    DoContext* ctx, uint32_t* r, uint64_t ro, const uint32_t* a, uint64_t ao, uint32_t b, uint64_t length, uint64_t stride, uint64_t steps, uint32_t p
) {
    CudaVectorGraph* graph = CudaVectorGraph::Get(ctx);
    if (!graph) return false;

    int elementsPerThread = graph->GetParams().elementsPerThread;
    return graph->Compute({a}, {r}, [r = r + ro, a = a + ao, b, length, stride, steps, p, elementsPerThread](cudaStream_t stream) {
        uint64_t threadsPerBlock = threadsPerBlockFor(length * steps, elementsPerThread);
        uint64_t blocksPerGrid = blocksPerGridFor(length * steps, threadsPerBlock, elementsPerThread);
        CudaFieldMulVectorExtKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(r, a, b, length, stride, steps, p, elementsPerThread);
        if (_CUDA_CHECK(cudaGetLastError()) != cudaSuccess) {
            return false;
        }
        return true;
    });
}

bool CudaFieldMulVectorsExt(
    DoContext* ctx,
    uint32_t* r, uint64_t ro, uint32_t rs,
    const uint32_t* a, uint64_t ao, uint32_t as,
    const uint32_t* b, uint64_t bo, uint32_t bs,
    uint64_t length, uint64_t steps, uint32_t p
) {
    CudaVectorGraph* graph = CudaVectorGraph::Get(ctx);
    if (!graph) return false;

    int elementsPerThread = graph->GetParams().elementsPerThread;
    return graph->Compute({a, b}, {r}, [r = r + ro, rs, a = a + ao, as, b = b + bo, bs, length, steps, p, elementsPerThread](cudaStream_t stream) {
        uint64_t threadsPerBlock = threadsPerBlockFor(length * steps, elementsPerThread);
        uint64_t blocksPerGrid = blocksPerGridFor(length * steps, threadsPerBlock, elementsPerThread);
        CudaFieldMulVectorsExtKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(r, rs, a, as, b, bs, length, steps, p, elementsPerThread);
        if (_CUDA_CHECK(cudaGetLastError()) != cudaSuccess) {
            return false;
        }
        return true;
    });
}

bool CudaFieldNegVectorExt(DoContext* ctx, uint32_t* r, uint64_t ro, uint64_t length, uint64_t stride, uint64_t steps, uint32_t p) {
    CudaVectorGraph* graph = CudaVectorGraph::Get(ctx);
    if (!graph) return false;

    int elementsPerThread = graph->GetParams().elementsPerThread;
    return graph->Compute({r}, {r}, [r = r + ro, length, stride, steps, p, elementsPerThread](cudaStream_t stream) {
        uint64_t threadsPerBlock = threadsPerBlockFor(length * steps, elementsPerThread);
        uint64_t blocksPerGrid = blocksPerGridFor(length * steps, threadsPerBlock, elementsPerThread);
        CudaFieldNegVectorsExtKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(r, length, stride, steps, p, elementsPerThread);
        if (_CUDA_CHECK(cudaGetLastError()) != cudaSuccess) {
            return false;
        }
        return true;
    });
}

bool CudaFieldAddVectorIfNonZeroExt(
    DoContext* ctx,
    bool* t, uint64_t to, uint64_t ts,
    uint32_t* r, uint64_t ro, uint64_t rs,
    const uint32_t* e, uint64_t eo, uint64_t es,
    uint64_t length, uint64_t steps, uint32_t p
) {
    CudaVectorGraph* graph = CudaVectorGraph::Get(ctx);
    if (!graph) return false;

    int elementsPerThread = graph->GetParams().elementsPerThread;
    return graph->Compute({e}, {t, r}, [t = t + to, ts, r = r + ro, rs, e = e + eo, es, length, steps, p, elementsPerThread](cudaStream_t stream) {
        uint64_t threadsPerBlockX = threadsPerBlockFor(length, elementsPerThread);
        uint64_t blocksPerGridX   = blocksPerGridFor(length, threadsPerBlockX, elementsPerThread);
        dim3 block(threadsPerBlockX, 1);
        dim3 grid(blocksPerGridX, steps);
        CudaFieldAddVectorIfNonZeroExtKernel<<<grid, block, 0, stream>>>(t, ts, r, rs, e, es, length, steps, p, elementsPerThread);
        if (_CUDA_CHECK(cudaGetLastError()) != cudaSuccess) {
            return false;
        }
        return true;
    });
}

#ifdef __cplusplus
} // extern "C"
#endif

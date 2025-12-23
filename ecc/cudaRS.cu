#include "cudaRS.h"
#include "../dataobjects/cudacommon.h"
#include "../dataobjects/cudaalloc.h"
#include "../dataobjects/cudacub.h"
#include "../dataobjects/cudagraph.h"
#include "../dataobjects/hdcommon.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <stdexcept>

__device__ uint32_t mod_add_u32(uint32_t a, uint32_t b, uint32_t q) {
    uint32_t s = a + b;
    if (s >= q) s -= q;
    return s;
}

__device__ uint32_t mod_sub_u32(uint32_t a, uint32_t b, uint32_t q) {
    return (a >= b) ? (a - b) : (uint32_t)(a + (uint64_t)q - b);
}

__device__ uint32_t mod_mul_u32(uint32_t a, uint32_t b, uint32_t q) {
    // Use 64-bit intermediate to avoid overflow
    uint64_t p = (uint64_t)a * (uint64_t)b;
    return (uint32_t)(p % q);
}

__device__ uint32_t mod_inv_u32(uint32_t a, uint32_t q) {
    // a * x + q * y = gcd(a, q)  -> if gcd == 1, x is the inverse mod q
    int64_t t0 = 0, t1 = 1;
    int64_t r0 = (int64_t)q, r1 = (int64_t)a;

    while (r1 != 0) {
        int64_t qout = r0 / r1;
        int64_t r2 = r0 - qout * r1; r0 = r1; r1 = r2;
        int64_t t2 = t0 - qout * t1; t0 = t1; t1 = t2;
    }

    // If r0 != 1, inverse doesn't exist (caller is responsible to ensure prime q and nonzero denom)
    // Normalize t0 mod q
    int64_t inv = t0 % (int64_t)q;
    if (inv < 0) inv += q;
    return (uint32_t)inv;
}

// Kernel: fill an array with a constant value
__global__ void fill_kernel(uint32_t* arr, size_t N, uint32_t value, int elementsPerThread) {
    size_t tid   = blockIdx.x * blockDim.x + threadIdx.x;
    size_t start = tid * elementsPerThread;
    size_t end   = start + elementsPerThread;

    if (start < N) {
        if (end > N) end = N;
        for (size_t i = start; i < end; ++i) {
            arr[i] = value;
        }
    }
}

// Kernel: replicate selection across remaining blocks using x_in as selected indices.
__global__ void replicate_selection_kernel(const uint32_t* __restrict__ code,
                                           const uint32_t* __restrict__ x_in,     // selected indices; first *z_count valid
                                           const uint32_t* __restrict__ z_count,  // device scalar
                                           uint32_t* __restrict__ y_in,
                                           uint32_t ecc_len,
                                           uint32_t extra_steps,
                                           int elementsPerThread)
{
    // Base index this thread starts at
    uint32_t base_i = (blockIdx.x * blockDim.x + threadIdx.x) * elementsPerThread;
    // Offset s by +1 so we skip block 0
    uint32_t s = 1 + (blockIdx.y * blockDim.y + threadIdx.y);

    if (s < extra_steps) {
        uint32_t count = *z_count; // device read
        // Each thread handles elementsPerThread contiguous indices
        for (int e = 0; e < elementsPerThread; ++e) {
            uint32_t i = base_i + e;
            if (i < count) {
                uint32_t pos = x_in[i];              // selected position within [0..ecc_len)
                uint32_t src = s * ecc_len + pos;    // source in code for block s
                uint32_t dst = s * ecc_len + i;      // compacted destination in y_in for block s
                y_in[dst] = code[src];
            }
        }
    }
}

// Device function: evaluate Lagrange basis polynomial L_i(x_star)
__device__ uint32_t lagrange_basis_eval_i_u32(
    uint32_t i, const uint32_t* x, const uint32_t* w, uint32_t k, uint32_t x_star, uint32_t q
) {
    uint32_t num = 1;
    for (uint32_t j = 0; j < k; ++j) {
        if (i == j) continue;
        uint32_t term = mod_sub_u32(x_star, x[j], q);
        num = mod_mul_u32(num, term, q);
    }
    return mod_mul_u32(w[i], num, q);
}

__global__ void barycentric_weights_u32_kernel(
    const uint32_t* __restrict__ x, uint32_t k, uint32_t q, uint32_t* __restrict__ w
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;

    // Each thread computes w[i]
    uint32_t den = 1;
    for (uint32_t j = 0; j < k; ++j) {
        if (i == j) continue;
        uint32_t diff = mod_sub_u32(x[i], x[j], q);
        den = mod_mul_u32(den, diff, q);
    }
    // inverse exists only if gcd(den, q) == 1; assume q prime and x distinct
    w[i] = mod_inv_u32(den, q);
}

// Host-side convenience wrapper
static inline bool barycentric_weights_u32_cuda(
    const uint32_t* x_d, uint32_t k, uint32_t q, uint32_t* w_d, cudaStream_t stream
) {
    uint64_t threadsPerBlock = threadsPerBlockFor(k, 1);
    uint64_t blocksPerGrid = blocksPerGridFor(k, threadsPerBlock, 1);
    barycentric_weights_u32_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(x_d, k, q, w_d);
    if (_CUDA_CHECK(cudaGetLastError()) != cudaSuccess) {
        return false;
    }
    return true;
}

// Multi-point Lagrange-interpolation evaluation
__global__ void LagrangeInterpEvalKernel(
    const uint32_t* __restrict__ x_in,
    const uint32_t* __restrict__ y_in,
    uint32_t k, uint32_t len,
    const uint32_t* __restrict__ eval_points,
    const uint32_t* __restrict__ point_count, // device scalar
    uint32_t q,
    const uint32_t* __restrict__ w,
    uint32_t* __restrict__ results,
    uint64_t stride,                           // number of result entries per step
    const bool* noisyQuery,
    uint32_t* __restrict__ success,            // device array; each step has its own entry
    uint64_t steps                             // total steps (grid.y)
) {
    // Identify step by the second grid dimension
    const uint64_t s = static_cast<uint64_t>(blockIdx.y);
    if (s >= steps) return;

    // Per-step views
    const uint32_t* __restrict__ x_step   = x_in + s * len;
    const uint32_t* __restrict__ y_step   = y_in + s * len;
    uint32_t*       __restrict__ res_step = results + s * stride;
    uint32_t*       __restrict__ ok_step  = success + s;

    // Read the current number of points once
    const uint32_t n_points = *point_count;

    // Global failure check for this step (done by a single thread in the step)
    if (n_points < k) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            atomicCAS(ok_step, 1u, 0u);
        }
        return; // Abort all blocks in this step
    }

    // Early exit for overprovisioned blocks
    const uint32_t i_point = blockIdx.x;
    if (i_point >= n_points || !noisyQuery[i_point]) {
        return;
    }

    // Per-thread partial accumulation over basis terms
    uint32_t local = 0;
    // Grid-stride loop within the block: threads cover all k terms
    for (uint32_t i = threadIdx.x; i < k; i += blockDim.x) {
        const uint32_t li   = lagrange_basis_eval_i_u32(i, x_step, w, k, eval_points[i_point], q);
        const uint32_t term = mod_mul_u32(y_step[i], li, q);
        local = mod_add_u32(local, term, q);
    }

    // Block-wide reduction in shared memory (threads > k contribute 0)
    extern __shared__ uint32_t shmem[];
    shmem[threadIdx.x] = local;
    __syncthreads();

    // Power-of-two style reduction works regardless of k; extra lanes are zero
    for (uint32_t rstride = blockDim.x >> 1; rstride > 0; rstride >>= 1) {
        if (threadIdx.x < rstride) {
            shmem[threadIdx.x] = mod_add_u32(shmem[threadIdx.x],
                                             shmem[threadIdx.x + rstride], q);
        }
        __syncthreads();
    }

    // Final result per point
    if (threadIdx.x == 0) {
        res_step[i_point] = shmem[0];
    }
}

// Kernel: fill generator matrix rows >= m
// Each thread computes one entry output[row * m + col]
__global__ void fill_generator_matrix_kernel(
    uint32_t n, uint32_t m, uint32_t q, const uint32_t* __restrict__ alphas_in,
    const uint32_t* __restrict__ w, uint32_t* __restrict__ output
) {
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y + m; // rows m..n-1
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;     // cols 0..m-1

    if (row < n && col < m) {
        uint32_t xstar = alphas_in[row];
        uint32_t lij = lagrange_basis_eval_i_u32(col, alphas_in, w, m, xstar, q);
        output[row * m + col] = lij;
    }
}

// Kernel: fill top m rows with identity
__global__ void fill_identity_kernel(uint32_t m, uint32_t* output, uint32_t elementsPerThread)
{
    // Global thread id
    uint32_t tid   = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start = tid * elementsPerThread;

    // Compute row and col for the first element this thread will handle
    uint32_t row = start / m;
    uint32_t col = start - (row * m);

    for (uint32_t e = 0; e < elementsPerThread; ++e) {
        uint32_t idx = start + e;
        if (idx >= m * m) break;

        output[idx] = (row == col) ? 1u : 0u;

        // Increment col, wrap to next row if needed
        ++col;
        if (col == m) {
            col = 0;
            ++row;
        }
    }
}

// Host wrapper
bool CudaGenerateSystematicRSMatrix_op(
    uint32_t n, uint32_t m, uint32_t q, const uint32_t* alphas_in_d, uint32_t* output_d,
    uint32_t* w_d, int elementsPerThread, cudaStream_t stream
) {
    if (m == 0 || n == 0) return cudaSuccess;

    // Compute barycentric weights for first m alphas
    if (!barycentric_weights_u32_cuda(alphas_in_d, m, q, w_d, stream)) {
        return false;
    }

    // Fill identity top m rows
    {
        uint64_t totalElements   = m * m;
        uint64_t threadsPerBlock = threadsPerBlockFor(totalElements, elementsPerThread);
        uint64_t blocksPerGrid   = blocksPerGridFor(totalElements, threadsPerBlock, elementsPerThread);

        dim3 block(threadsPerBlock);
        dim3 grid(blocksPerGrid);

        fill_identity_kernel<<<grid, block, 0, stream>>>(m, output_d, elementsPerThread);
        if (_CUDA_CHECK(cudaGetLastError()) != cudaSuccess) {
            return false;
        }
    }

    // Fill remaining rows
    {
        uint64_t threadsPerBlockX = threadsPerBlockFor(m, 1, defaultMaxThreads2D);
        uint64_t threadsPerBlockY = threadsPerBlockFor(n, 1, defaultMaxThreads2D);

        uint64_t blocksPerGridX   = blocksPerGridFor(m, threadsPerBlockX, 1);
        uint64_t blocksPerGridY   = blocksPerGridFor(n, threadsPerBlockY, 1);

        dim3 block(threadsPerBlockX, threadsPerBlockY);
        dim3 grid(blocksPerGridX, blocksPerGridY);

        fill_generator_matrix_kernel<<<grid, block, 0, stream>>>(n, m, q, alphas_in_d, w_d, output_d);
        if (_CUDA_CHECK(cudaGetLastError()) != cudaSuccess) {
            return false;
        }
    }

    return true;
}
struct IsNotNoisyQuery {
    __device__ bool operator()(const thrust::tuple<uint32_t, uint32_t, uint32_t>& t) const {
        // third element is noisyQuery
        return !thrust::get<2>(t);
    }
};

#ifdef __cplusplus
extern "C" {
#endif

bool CudaGenerateSystematicRSMatrix(DoContext* ctx, uint32_t n, uint32_t m, uint32_t Q, const uint32_t* alphas_in, uint32_t* output) {
    CudaVectorGraph* graph = CudaVectorGraph::Get(ctx);
    if (!graph) return false;

    int elementsPerThread = graph->GetParams().elementsPerThread;

    bool success;

    uint32_t* w = static_cast<uint32_t*>(graph->Malloc(sizeof(uint32_t) * m));
    if (!w) return false;

    success = graph->Compute({alphas_in}, {output, w}, [n, m, Q, alphas_in, w, output, elementsPerThread](cudaStream_t stream) {
        return CudaGenerateSystematicRSMatrix_op(n, m, Q, alphas_in, output, w, elementsPerThread, stream);
    });
    if (!success) return false;

    return graph->Free(w);
}

bool CudaReedSolomonDecode(
    DoContext* ctx,
    uint32_t *code, uint64_t co, uint64_t cs,
    const bool* noisyQuery,
    uint32_t ecc_len, uint32_t ecc_k, uint32_t q,
    uint32_t* success,
    uint64_t steps
) {
    CudaVectorGraph* graph = CudaVectorGraph::Get(ctx);
    if (!graph) return false;
    
    uint32_t* x_in = static_cast<uint32_t*>(graph->Malloc(sizeof(uint32_t) * ecc_len * steps));
    if (!x_in) return false;
    uint32_t* y_in = static_cast<uint32_t*>(graph->Malloc(sizeof(uint32_t) * ecc_len * steps));
    if (!y_in) return false;
    uint32_t* f_in = static_cast<uint32_t*>(graph->Malloc(sizeof(uint32_t) * ecc_len));
    if (!f_in) return false;
    uint32_t* z_count = static_cast<uint32_t*>(graph->Malloc(sizeof(uint32_t)));
    if (!z_count) return false;

    // Allocate weights once
    uint32_t* w_in = static_cast<uint32_t*>(graph->Malloc(sizeof(uint32_t) * ecc_k));
    if (!w_in) return false;

    
    // Compact (i, code[i]) into (x_in, y_in)
    auto zip_input_xyf = thrust::make_zip_iterator(
        thrust::make_tuple(thrust::counting_iterator<uint32_t>(0), code, noisyQuery));
    auto zip_output_xyf = thrust::make_zip_iterator(
        thrust::make_tuple(x_in, y_in, f_in));

    GraphAllocator<char> galloc(graph);

    auto cub_xyf = [zip_input_xyf, zip_output_xyf, z_count, ecc_len, x_in, y_in, f_in, graph](char* temp, size_t& bytes, cudaStream_t s) {
        return cub::DeviceSelect::If(
            temp, bytes,
            zip_input_xyf,
            zip_output_xyf,
            z_count,
            ecc_len,
            IsNotNoisyQuery(),
            s);
    };
    auto cubcall_xyf = CubCall(std::function(cub_xyf), galloc);
    if (_CUDA_CHECK(cubcall_xyf.Prepare()) != cudaSuccess) {
        return false;
    }

    int elementsPerThread = graph->GetParams().elementsPerThread;

    bool gsuccess;

    gsuccess = graph->Compute({code, noisyQuery}, {code, success, x_in, y_in, f_in, z_count, w_in}, [
        code = code + co, cs, noisyQuery, ecc_len, ecc_k, q , success, steps, x_in, y_in, z_in = x_in, z_count, w_in, cubcall_xyf, elementsPerThread
    ](cudaStream_t stream) mutable {
        // step 1: initialize
        {
            uint64_t threadsPerBlock = threadsPerBlockFor(steps, elementsPerThread);
            uint64_t blocksPerGrid = blocksPerGridFor(steps, threadsPerBlock, elementsPerThread);
            fill_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(success, steps, 1, elementsPerThread);
            if (_CUDA_CHECK(cudaGetLastError()) != cudaSuccess) {
                return false;
            }
            if (_CUDA_CHECK(cudaMemsetAsync(z_count, 0, sizeof(uint32_t), stream)) != cudaSuccess) {
                return false;
            }
        }

        // step 2: first block - also prepares first block of x_in for get-weights step
        {
            if (_CUDA_CHECK(cubcall_xyf.Call(stream)) != cudaSuccess) {
                return false;
            }
        }

        // step 3: remaining blocks
        {
            uint64_t threadsPerBlockX = threadsPerBlockFor(ecc_len, elementsPerThread);
            uint64_t blocksPerGridX   = blocksPerGridFor(ecc_len, threadsPerBlockX, elementsPerThread);

            dim3 block(threadsPerBlockX, 1);
            dim3 grid(blocksPerGridX, steps - 1);  // height dimension untouched

            replicate_selection_kernel<<<grid, block, 0, stream>>>(
                code, x_in, z_count, y_in, ecc_len, steps - 1, elementsPerThread
            );
            if (_CUDA_CHECK(cudaGetLastError()) != cudaSuccess) {
                return false;
            }
        }
      
        // step 4: get weights
        {
            if (!barycentric_weights_u32_cuda(x_in, ecc_k, q, w_in, stream)) {
                return false;
            }
        }

        // step 5: interpolation
        {
            uint32_t selectedThreadsPerBlock = 1;
            while (selectedThreadsPerBlock < ecc_k && selectedThreadsPerBlock < defaultMaxThreads) {
                selectedThreadsPerBlock <<= 1;
            }

            dim3 grid(ecc_len, steps); // Grid: (points, steps)
            dim3 block(selectedThreadsPerBlock);

            size_t shmem_size = selectedThreadsPerBlock * sizeof(uint32_t);

            // 'stride' is the number of result entries per step (>= ecc_len)
            uint32_t* z_in = x_in;
            LagrangeInterpEvalKernel<<<grid, block, shmem_size, stream>>>(
                x_in, y_in, ecc_k, ecc_len,
                z_in, z_count, q,
                w_in,
                code,
                cs,
                noisyQuery,
                success,
                steps
            );
            if (_CUDA_CHECK(cudaGetLastError()) != cudaSuccess) {
                return false;
            }
        }

        return true;
    });
    if (!gsuccess) return false;

    gsuccess = graph->Free(f_in);
    if (!gsuccess) return false;
    gsuccess = graph->Free(w_in);
    if (!gsuccess) return false;
    gsuccess = graph->Free(x_in);
    if (!gsuccess) return false;
    gsuccess = graph->Free(y_in);
    if (!gsuccess) return false;
    gsuccess = graph->Free(z_count);
    if (!gsuccess) return false;

    return true;
}

#ifdef __cplusplus
} // extern "C"
#endif

// ntt_cuda.cu

#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "cudantt.h"
#include "cudatdm.h"
#include "../dataobjects/docontextimpl.h"
#include "../dataobjects/cudagraph.h"
#include "../dataobjects/cudacommon.h"
#include "../dataobjects/hdcommon.h"
#include "../dataobjects/plainfields.h"
#include "../dataobjects/cudafields.h"


// ==============================
// Helpers
// ==============================
__host__ __device__ __forceinline__ uint32_t mod_add(uint32_t a, uint32_t b, uint32_t mod) {
    uint32_t s = a + b;
    return s % mod;
}

__host__ __device__ __forceinline__ uint32_t mod_sub(uint32_t a, uint32_t b, uint32_t mod) {
    uint32_t s = a + mod - b;
    return s % mod;
}

__host__ __device__ __forceinline__ uint32_t mod_mul(uint32_t a, uint32_t b, uint32_t mod) {
    return (uint64_t)a * (uint64_t)b % (uint64_t)mod;
}

__host__ __device__ __forceinline__ uint32_t mod_pow(uint32_t base, uint32_t exp, uint32_t mod) {
    uint64_t result = 1;
    uint64_t b = base % mod;
    while (exp) {
        if (exp & 1) result = (result * b) % mod;
        b = (b * b) % mod;
        exp >>= 1;
    }
    return (uint32_t)result;
}

__host__ __device__ __forceinline__ uint32_t mod_inv(uint32_t a, uint32_t mod) {
    return mod_pow(a, mod - 2, mod);
}

// ==============================
// Power-of-two check and fast ilog2
// ==============================
__host__ __device__ __forceinline__ int ilog2_power_of_two(uint32_t n) {
    if (!n || ((n & (n - 1)) != 0)) return -1;
    int index = 0;
    if (0xFFFF0000U & n) index += 16;
    if (0xFF00FF00U & n) index += 8;
    if (0xF0F0F0F0U & n) index += 4;
    if (0xCCCCCCCCU & n) index += 2;
    if (0xAAAAAAAAU & n) index += 1;
    return index;
}

// ==============================
// Bit-reversal permutation
// ==============================
__device__ __forceinline__ uint32_t bit_reverse_index(uint32_t i, int logn) {
    uint32_t r = 0;
    for (int k = 0; k < logn; ++k) {
        r = (r << 1) | ((i >> k) & 1U);
    }
    return r;
}

__global__ void compute_bitrev_perm_kernel(uint32_t* __restrict__ perm,
                                           uint32_t n,
                                           int elementsPerThread)
{
    int logn = ilog2_power_of_two(n);
    if (logn < 0) return;

    // Global thread id
    uint32_t tid   = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start = tid * elementsPerThread;

    // Each thread computes elementsPerThread consecutive entries
    for (int e = 0; e < elementsPerThread; ++e) {
        uint32_t i = start + e;
        if (i < n) {
            perm[i] = bit_reverse_index(i, logn);
        }
    }
}

/*
__global__ void apply_bitrev_inplace_kernel(uint32_t* __restrict__ a,
                                            const uint32_t* __restrict__ perm,
                                            uint32_t n,
                                            int elements_per_thread) {
    uint64_t total = n;
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t start = tid * (uint64_t)elements_per_thread;
    for (uint64_t k = 0; k < (uint64_t)elements_per_thread; ++k) {
        uint64_t i = start + k;
        if (i >= total) break;
        uint32_t j = perm[i];
        if (i < j) {
            uint32_t tmp = a[i];
            a[i] = a[j];
            a[j] = tmp;
        }
    }
}
*/
__global__ void apply_bitrev_inplace_kernel(uint32_t* __restrict__ a,
                                            const uint32_t* __restrict__ perm,
                                            uint32_t n,
                                            uint32_t stride,
                                            uint32_t steps,
                                            int elements_per_thread) {
    uint64_t total = (uint64_t)steps * n;

    uint64_t tid   = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t start = tid * (uint64_t)elements_per_thread;

    if (start >= total) return;

    // Compute initial row and column once
    uint32_t row = start / n;
    uint32_t col = start - (row * n);

    for (uint64_t k = 0; k < (uint64_t)elements_per_thread; ++k) {
        uint64_t global_idx = start + k;
        if (global_idx >= total) break;

        // Apply permutation within the row
        uint32_t j_col = perm[col];

        // Compute absolute indices in the array
        uint64_t i = row * stride + col;
        uint64_t j = row * stride + j_col;

        if (i < j) {
            uint32_t tmp = a[i];
            a[i] = a[j];
            a[j] = tmp;
        }

        // Increment column, wrap to next row when needed
        ++col;
        if (col == n) {
            col = 0;
            ++row;
        }
    }
}

// ==============================
// Vectorized twiddle generation
// ==============================
// Build twiddle vector W[j] = wlen^j mod mod for j=0..half-1

// Kernel: precompute pow2 scalars on device
__global__ void PrecomputePow2Kernel(uint32_t* pow2,
                                     uint32_t half,
                                     uint32_t wlen,
                                     uint32_t mod) {
    // Only one thread computes pow2 (small array)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        uint32_t p = wlen % mod;
        pow2[0] = p;
        int idx = 1;
        while ((1U << idx) <= half) {
            uint64_t sq = (uint64_t)pow2[idx - 1] * (uint64_t)pow2[idx - 1];
            pow2[idx] = (uint32_t)(sq % mod);
            idx++;
        }
    }
}

// Kernel: fill selector array for bit b
__global__ void FillSelectorKernel(uint32_t* sel,
                                   uint32_t half,
                                   int b,
                                   const uint32_t* pow2,
                                   uint32_t elementsPerThread) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t start = tid * elementsPerThread;
    if (start >= half) return;

    const uint32_t end = min(start + elementsPerThread, half);
    for (uint32_t j = start; j < end; ++j) {
        sel[j] = ((j >> b) & 1U) ? pow2[b] : 1U;
    }
}

struct TwiddleVectorKey {
    uint32_t root;
    uint32_t n;
    uint32_t mod;

    bool operator==(const TwiddleVectorKey& other) const {
        return std::tie(root, n, mod) == std::tie(other.root, other.n, other.mod);
    }
};

namespace std {
    template <>
    struct hash<TwiddleVectorKey> {
        std::size_t operator()(const TwiddleVectorKey& k) const noexcept {
            return std::hash<uint32_t>()(k.root) ^ (std::hash<uint32_t>()(k.n) << 1) ^ (std::hash<uint32_t>()(k.mod) << 2);
        }
    };
}

class TwiddleVectorCache {
private:
    static TwiddleVectorCache* Get(DoContext* ctx) {
        DoContextImpl* ctximpl = reinterpret_cast<DoContextImpl*>(ctx);
        if (!ctximpl) return nullptr;
        std::shared_ptr<TwiddleVectorCache>* p_cache = ctximpl->get<std::shared_ptr<TwiddleVectorCache>>();
        if (!p_cache) {
            p_cache = ctximpl->put(std::make_shared<TwiddleVectorCache>());
        };
        return p_cache->get();
    }

public:
    static bool GetVector(DoContext* ctx, const TwiddleVectorKey& key, uint32_t** out_vec) {
        CudaVectorGraph* graph = CudaVectorGraph::Get(ctx);
        if (!graph) return false;
        TwiddleVectorCache* cache = TwiddleVectorCache::Get(ctx);
        if (!cache) return false;
        return cache->GetVector(ctx, graph, key, out_vec);
    }

    TwiddleVectorCache() {}
    ~TwiddleVectorCache() {
        std::lock_guard<std::mutex> lock(mtx_);
        // for (auto& e : cache_) { // NOTE: let the graph destructor take care of freeing each e.second
        //     _CUDA_CHECK(cudaFreeAsync(e.second, 0));
        // }
    }

private:
    bool GetVector(DoContext* ctx, CudaVectorGraph* graph, const TwiddleVectorKey& key, uint32_t** out_vec) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            *out_vec = it->second;
            return true;
        }

        uint32_t root = key.root, n = key.n, mod = key.mod;
        
        uint32_t* Ws = static_cast<uint32_t*>(graph->Malloc(sizeof(uint32_t) * n * 32));
        if (!Ws) return false;

        uint32_t* pow2 = reinterpret_cast<uint32_t*>(graph->Malloc(32 * sizeof(uint32_t)));
        if (!pow2) return false;

        uint32_t* sel = reinterpret_cast<uint32_t*>(graph->Malloc(sizeof(uint32_t) * n * 32));
        if (!sel) return false;

        bool success;

        if (!CudaFieldSetVector(ctx, Ws, 0, n * 32, 1U)) {
            return false;
        }

        int elementsPerThread = graph->GetParams().elementsPerThread;

        for (uint32_t len = 2, Wo = 0; len <= (uint32_t)n; len <<= 1U, Wo += n) {
            uint32_t half = len >> 1U;
            uint32_t wlen = mod_pow(root, ((uint32_t)n) / len, mod);

            success = graph->Compute({}, {Ws, pow2, sel}, [ctx, Ws, Wo, sel, half, wlen, pow2, mod, elementsPerThread](cudaStream_t stream) {
                PrecomputePow2Kernel<<<1, 1, 0, stream>>>(pow2, half, wlen, mod);
                if (_CUDA_CHECK(cudaGetLastError()) != cudaSuccess) {
                    return false;
                }
                // Iteratively multiply W by selector vectors
                uint64_t threadsPerBlock = threadsPerBlockFor(half, elementsPerThread);
                uint64_t blocksPerGrid   = blocksPerGridFor(half, threadsPerBlock, elementsPerThread);

                dim3 block(threadsPerBlock);
                dim3 grid(blocksPerGrid);

                for (int b = 0; (1U << b) < half; ++b) {
                    FillSelectorKernel<<<grid, block, 0, stream>>>(sel, half, b, pow2, elementsPerThread);
                    if (_CUDA_CHECK(cudaGetLastError()) != cudaSuccess) {
                        return false;
                    }
                    if (!CudaFieldMulVectors(ctx, Ws, Wo, Ws, Wo, sel, 0, half, mod)) {
                        return false;
                    }
                }
                return true;
            });
            if (!success) return false;
        }
        // NOTE - for debug printing
        // cudaDeviceSynchronize();
        // if (_CUDA_CHECK(cudaGetLastError()) != cudaSuccess) {
        //     return false;
        // }
        // uint32_t Wo = 0;
        // for (uint32_t len = 2; len <= n; len <<= 1U, Wo += n) {
        //     uint32_t half = len >> 1;
        //     std::cerr << "Stage len=" << len << " half=" << half << " W[0..half-1]:";
        //     for (uint32_t j = 0; j < std::min<uint32_t>(half, 32); ++j) {
        //         std::cerr << ' ' << Ws[Wo + j];
        //     }
        //     std::cerr << std::endl;
        // }

        success = graph->Free(sel);
        if (!success) return false;

        success = graph->Free(pow2);
        if (!success) return false;

        *out_vec = Ws;
        return true;
    }

private:
    std::mutex mtx_;
    std::unordered_map<TwiddleVectorKey, uint32_t*> cache_;
};

// ==============================
// NTT stage kernel (uses precomputed twiddle vector W)
// ==============================
/*
__global__ void ntt_stage_kernel_vec(uint32_t* __restrict__ a,
                                     uint32_t n,
                                     uint32_t len,
                                     const uint32_t* __restrict__ W,
                                     uint32_t mod,
                                     int elements_per_thread) {
    uint64_t butterflies = (uint64_t)n >> 1; // n/2 total butterflies per stage
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t start = tid * (uint64_t)elements_per_thread;

    uint32_t half = len >> 1;

    for (uint64_t t = 0; t < (uint64_t)elements_per_thread; ++t) {
        uint64_t k = start + t;
        if (k >= butterflies) break;

        uint64_t seg = k / half;
        uint32_t j = (uint32_t)(k % half);
        uint64_t i = seg * (uint64_t)len;

        uint32_t w = W[j];

        uint32_t u = a[i + j];
        uint32_t v = mod_mul(a[i + j + half], w, mod);
        a[i + j] = mod_add(u, v, mod);
        a[i + j + half] = mod_sub(u, v, mod);
    }
}
*/
__global__ void ntt_stage_kernel_vec(uint32_t* __restrict__ a,
                                     uint32_t n,
                                     uint32_t stride,
                                     uint32_t steps,
                                     uint32_t len,
                                     const uint32_t* __restrict__ W,
                                     uint32_t mod,
                                     int elements_per_thread) {
    // Total butterflies per row is n/2; across all rows:
    uint64_t butterflies_total = (uint64_t)steps * ((uint64_t)n >> 1);

    uint64_t tid   = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t start = tid * (uint64_t)elements_per_thread;

    if (start >= butterflies_total) return;

    uint32_t half = len >> 1;

    // Precompute row and k_in_row once; avoid div/mod inside the loop.
    uint64_t butterflies_per_row = (uint64_t)n >> 1;
    uint64_t row = start / butterflies_per_row;
    uint64_t k_in_row = start - (row * butterflies_per_row);

    // Derive initial segment and position within segment once
    uint64_t seg = k_in_row / half;
    uint64_t j   = k_in_row - (seg * half);

    // Number of segments per row (each segment has 'len' elements; each contributes 'half' butterflies)
    uint64_t segments_per_row = n / len;

    for (uint64_t t = 0; t < elements_per_thread; ++t) {
        uint64_t gk = start + t;
        if (gk >= butterflies_total) break;

        // Base index for this segment within the current row
        uint64_t base = row * stride + seg * len;

        uint32_t w = W[j];

        uint32_t u = a[base + j];
        uint32_t v = mod_mul(a[base + j + half], w, mod);
        a[base + j]        = mod_add(u, v, mod);
        a[base + j + half] = mod_sub(u, v, mod);

        // Advance j; wrap to next segment when j reaches 'half'
        ++j;
        if (j == half) {
            j = 0;
            ++seg;

            // If we finished all segments in this row, move to the next row
            if (seg == segments_per_row) {
                seg = 0;
                ++row;
            }
        }
    }
}

// ==============================
// Bit-reversal cache class
// ==============================
class BitrevCache {
private:
    static BitrevCache* Get(DoContext* ctx) {
        DoContextImpl* ctximpl = reinterpret_cast<DoContextImpl*>(ctx);
        if (!ctximpl) return nullptr;
        std::shared_ptr<BitrevCache>* p_cache = ctximpl->get<std::shared_ptr<BitrevCache>>();
        if (!p_cache) {
            p_cache = ctximpl->put(std::make_shared<BitrevCache>());
        };
        return p_cache->get();
    }

public:
    static bool GetPerm(DoContext* ctx, uint32_t n, uint32_t** out_perm) {
        CudaVectorGraph* graph = CudaVectorGraph::Get(ctx);
        if (!graph) return false;
        BitrevCache* cache = BitrevCache::Get(ctx);
        if (!cache) return false;
        return cache->GetPerm(graph, n, out_perm);
    }

    BitrevCache() {
        for (int i = 0; i < 32; ++i) {
            ns_[i] = 0;
            dev_perm_[i] = nullptr;
        }
    }

    ~BitrevCache() {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            for (int i = 0; i < 32; ++i) {
                if (dev_perm_[i]) {
                    //_CUDA_CHECK(cudaFreeAsync(dev_perm_[i], 0)); // NOTE: let the graph destructor take care of freeing dev_perm_[i]
                    dev_perm_[i] = nullptr;
                    ns_[i] = 0;
                }
            }
        }
        _CUDA_CHECK(cudaDeviceSynchronize());
    }

private:
    // Returns true on success, and sets *out_perm to device pointer.
    bool GetPerm(CudaVectorGraph* graph, uint32_t n, uint32_t** out_perm) {
        if (n == 0 || n > UINT32_MAX) return false;
        int logn = ilog2_power_of_two((uint32_t)n);
        if (logn < 0 || logn >= 32) return false;

        std::lock_guard<std::mutex> lock(mtx_);
        if (ns_[logn] == (uint32_t)n && dev_perm_[logn] != nullptr) {
            *out_perm = dev_perm_[logn];
            return true;
        }

        uint32_t* d_perm = reinterpret_cast<uint32_t*>(graph->Malloc(n * sizeof(uint32_t)));
        if (!d_perm) return false;

        int elementsPerThread = graph->GetParams().elementsPerThread;

        bool success;

        success = graph->Compute({}, {d_perm}, [d_perm, n, elementsPerThread](cudaStream_t stream) {
            uint64_t threadsPerBlock = threadsPerBlockFor(n, elementsPerThread);
            uint64_t blocksPerGrid   = blocksPerGridFor(n, threadsPerBlock, elementsPerThread);

            dim3 block(threadsPerBlock);
            dim3 grid(blocksPerGrid);

            compute_bitrev_perm_kernel<<<grid, block, 0, stream>>>(d_perm, n, elementsPerThread);
            if (_CUDA_CHECK(cudaGetLastError()) != cudaSuccess) {
                return false;
            }
            return true;
        });
        if (!success) return false;

        ns_[logn] = (uint32_t)n;
        dev_perm_[logn] = d_perm;
        *out_perm = d_perm;
        return true;
    }

private:
    std::mutex mtx_;
    uint32_t ns_[32];
    uint32_t* dev_perm_[32];
};

// ==============================
// NTT
// ==============================

#ifdef __cplusplus
extern "C" {
#endif

// Core NTT with explicit elements_per_thread and optional precomputed permutation.
// Returns true on success, false on invalid n or allocation failures.
bool cuda_ntt(DoContext* ctx,
              uint32_t* a,
              uint32_t ao,
              uint32_t n,
              uint32_t stride,
              uint32_t steps,
              uint32_t root,
              uint32_t mod) {
    CudaVectorGraph* graph = CudaVectorGraph::Get(ctx);
    if (!graph) return false;

    // Validate n in (0, 2^32]
    if (n == 0 || n > UINT32_MAX) return false;
    int logn = ilog2_power_of_two((uint32_t)n);
    if (logn < 0) return false;

    // Prepare permutation
    uint32_t* d_perm;
    if (!BitrevCache::GetPerm(ctx, n, &d_perm)) return false;

    int elementsPerThread = graph->GetParams().elementsPerThread;

    bool success;
    // Apply bit-reversal in-place
    success = graph->Compute({d_perm, a}, {a}, [elementsPerThread, a = a + ao, n, stride, steps, d_perm](cudaStream_t stream) {
        uint64_t threadsPerBlock = threadsPerBlockFor(n * steps, elementsPerThread);
        uint64_t blocksPerGrid   = blocksPerGridFor(n * steps, threadsPerBlock, elementsPerThread);

        dim3 block(threadsPerBlock);
        dim3 grid(blocksPerGrid);

        apply_bitrev_inplace_kernel<<<grid, block, 0, stream>>>(a, d_perm, n, stride, steps, elementsPerThread);
        if (_CUDA_CHECK(cudaGetLastError()) != cudaSuccess) {
            return false;
        }
        return true;
    });
    if (!success) return false;

    // prepare twiddle vectors
    uint32_t* Ws;
    success = TwiddleVectorCache::GetVector(ctx, {root, n, mod}, &Ws);
    if (!success) return false;

    // Iterative NTT stages with vectorized twiddles
    success = graph->Compute({Ws}, {a}, [a = a + ao, Ws, n, stride, steps, mod, elementsPerThread](cudaStream_t stream) {
        // Launch stage kernel
        uint64_t butterflies = ((uint64_t)n) >> 1;
        uint64_t threadsPerBlock = threadsPerBlockFor(butterflies * steps, elementsPerThread);
        uint64_t blocksPerGrid   = blocksPerGridFor(butterflies * steps, threadsPerBlock, elementsPerThread);

        dim3 block(threadsPerBlock);
        dim3 grid(blocksPerGrid);

        uint32_t* W = Ws;
        for (uint32_t len = 2; len <= n; len <<= 1U, W += n) {
            ntt_stage_kernel_vec<<<grid, block, 0, stream>>>(a, n, stride, steps, len, W, mod, elementsPerThread);
            if (_CUDA_CHECK(cudaGetLastError()) != cudaSuccess) {
                return false;
            }
        }
        return true;
    });
    if (!success) return false;

    return true;
}

bool cuda_intt(DoContext* ctx,
               uint32_t* a,
               uint32_t ao,
               uint32_t n,
               uint32_t stride,
               uint32_t steps,
               uint32_t root,
               uint32_t mod) {
    uint32_t inv_root = mod_inv(root, mod);

    if (!cuda_ntt(ctx, a, ao, n, stride, steps, inv_root, mod)) return false;

    uint32_t inv_n = mod_inv(n, mod);
    if (!CudaFieldMulVectorExt(ctx, a, ao, a, ao, inv_n, n, stride, steps, mod)) return false;
    return true;
}

bool cuda_ntt_convolution(DoContext* ctx,
                          const uint32_t* a,
                          uint32_t ao,
                          uint32_t as,
                          const uint32_t* b,
                          uint32_t bo,
                          uint32_t bs,
                          uint32_t* result,
                          uint32_t ro,
                          uint32_t rs,
                          uint32_t n,
                          uint32_t steps,
                          uint32_t root,
                          uint32_t mod) {
    CudaVectorGraph* graph = CudaVectorGraph::Get(ctx);
    if (!graph) {
        return false;
    }

    if ((as && as < n) || (bs && bs < n) || (rs && rs < n)) {
        return false;
    }
    size_t bytes = sizeof(uint32_t) * n * steps;

    uint32_t* fa = static_cast<uint32_t*>(graph->Malloc(bytes));
    if (!fa) {
        return false;
    }
    uint32_t* fb = static_cast<uint32_t*>(graph->Malloc(bytes));
    if (!fb) {
        return false;
    }

    bool success;
    // Copy input arrays into fa and fb (device-to-device copy)
    success = CudaPermutedExtentsAssign(ctx, fa, 0, n, 0, a, ao, as, 0, n, nullptr, 0, steps);
    if (!success) {
        return false;
    }
    success = CudaPermutedExtentsAssign(ctx, fb, 0, n, 0, b, bo, bs, 0, n, nullptr, 0, steps);
    if (!success) {
        return false;
    }

    // Forward NTT on fa and fb
    if (!cuda_ntt(ctx, fa, 0, n, n, steps, root, mod)) {
        return false;
    }
    if (!cuda_ntt(ctx, fb, 0, n, n, steps, root, mod)) {
        return false;
    }

    // Pointwise multiplication in frequency domain
    if (!CudaFieldMulVectorsExt(ctx, result, ro, rs, fa, 0, n, fb, 0, n, n, steps, mod)) {
        return false;
    }

    // Inverse NTT to get the convolution result
    if (!cuda_intt(ctx, result, ro, n, rs, steps, root, mod)) {
        return false;
    }

    return true;
}

#ifdef __cplusplus
} // extern "C"
#endif

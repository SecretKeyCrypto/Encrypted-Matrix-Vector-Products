#include "cudatdm.h"
#include <cuda_runtime.h>

#include "../dataobjects/cudacommon.h"
#include "../dataobjects/cudagraph.h"

constexpr uint64_t BLOCK_ROWS = 8;

__global__ void CudaPermutedExtentsAssignKernel(
    uint32_t* __restrict__ r,
    uint32_t ro,
    uint32_t rfs,
    uint32_t rps,
    const uint32_t* __restrict__ s,  // may be nullptr => zeros
    uint32_t so,
    uint32_t ss,
    uint32_t sc,
    uint64_t extent,
    const uint32_t* __restrict__ perm, // may be nullptr => identity
    uint32_t po,
    uint64_t length,
    int elementsPerThread)
{
    const uint64_t total = length * extent;
    const uint64_t tIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t start = tIndex * (uint64_t)elementsPerThread;
    if (start >= total) return;

    const uint64_t end = min(start + (uint64_t)elementsPerThread, total);

    // Initialize i and e once
    uint64_t i = start / extent;
    uint64_t e = start - i * extent;

    // Compute bases for this i
    uint64_t ri_base = ro + i * rfs + (perm ? perm[po + i] : (uint32_t)i) * rps;
    uint64_t si_base = so + i * ss;

    // Current indices
    uint64_t ri = ri_base + e;
    uint64_t si = si_base + e;

    for (uint64_t k = start; k < end; ++k) {
        // Write element
        const uint32_t sval = s ? s[si] : 0u;
        r[ri] = sval + sc;

        // Advance to next element
        ++e;
        ++ri;
        ++si;

        // If we’ve exhausted this extent, move to next i
        if (e == extent) {
            ++i;
            e = 0;
            ri_base = ro + i * rfs + (perm ? perm[po + i] : (uint32_t)i) * rps;
            si_base = so + i * ss;
            ri = ri_base;
            si = si_base;
        }
    }
}

__global__ void CudaCircularCopyKernel(uint32_t* __restrict__ r,
                                       const uint32_t* __restrict__ v,
                                       uint64_t length,
                                       int elementsPerThread) {
    uint64_t total = length * length; // total elements in r
    uint64_t tid   = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t start = tid * elementsPerThread;

    if (start >= total) return;

    // Compute initial row and column once
    uint64_t row = start / length;
    uint64_t col = start - (row * length);

    for (int k = 0; k < elementsPerThread; ++k) {
        uint64_t gidx = start + k;
        if (gidx >= total) break;

        // For row = t, col = j, the rotated index is (j + (length - row)) % length
        uint64_t srcIdx = col + (length - row);
        if (srcIdx >= length) srcIdx -= length;
        r[row * length + col] = v[srcIdx];

        // Advance column; wrap to next row when needed
        ++col;
        if (col == length) {
            col = 0;
            ++row;
        }
    }
}

__global__ void CudaMatrixTranspose1Kernel(uint32_t* r,
                                          const uint32_t* a,
                                          uint64_t rows, uint64_t cols,
                                          uint64_t tileRows, uint64_t tileCols,
                                          int elementsPerThread)
{
    extern __shared__ uint32_t tile[]; // flat shared memory
    // Indexing helper: tile[row][col] → tile[row * (tileCols+1) + col]
    auto tileIndex = [&](int row, int col) {
        return row * (tileCols + 1) + col;
    };

    // Block starting coordinates
    uint64_t col0 = blockIdx.x * tileCols + threadIdx.x;
    uint64_t row0 = blockIdx.y * tileRows + threadIdx.y;

    // Load tile into shared memory
    for (int row1 = 0; row1 < tileRows; row1 += BLOCK_ROWS) {
        for (int e = 0; e < elementsPerThread; ++e) {
            uint64_t rr = row0 + row1 + e * BLOCK_ROWS;
            if (col0 < cols && rr < rows) {
                tile[tileIndex(threadIdx.y + row1 + e * BLOCK_ROWS,
                               threadIdx.x)] = a[rr * cols + col0];
            }
        }
    }

    __syncthreads();

    // Write transposed tile
    col0 = blockIdx.y * tileRows + threadIdx.x;
    row0 = blockIdx.x * tileCols + threadIdx.y;

    for (int col1 = 0; col1 < tileCols; col1 += BLOCK_ROWS) {
        for (int e = 0; e < elementsPerThread; ++e) {
            uint64_t cc = row0 + col1 + e * BLOCK_ROWS;
            if (col0 < rows && cc < cols) {
                r[cc * rows + col0] =
                    tile[tileIndex(threadIdx.x,
                                   threadIdx.y + col1 + e * BLOCK_ROWS)];
            }
        }
    }
}

bool CudaPermutedExtentsAssign(
    DoContext* ctx,
    uint32_t* r,
    uint32_t ro,
    uint32_t rfs,
    uint32_t rps,
    const uint32_t* s,
    uint32_t so,
    uint32_t ss,
    uint32_t sc,
    uint64_t extent,
    const uint32_t* perm,
    uint32_t po,
    uint64_t length)
{
    CudaVectorGraph* graph = CudaVectorGraph::Get(ctx);
    if (!graph) {
        return false;
    }
    int elementsPerThread = graph->GetParams().elementsPerThread;

    if (length == 0 || extent == 0 || elementsPerThread <= 0) {
        return false;
    }

    bool success;

    std::vector<const void*> inputs;
    inputs.push_back(s);
    if (perm) inputs.push_back(perm);
    // NOTE - for debug printing
    // size_t permsize = graph->Size(perm);
    // size_t rsize = graph->Size(r);
    // size_t ssize = graph->Size(s);
    success = graph->Compute(inputs, {r}, [r, ro, rfs, rps, s, so, ss, sc, extent, perm, po, length, elementsPerThread](cudaStream_t stream) {
        const uint64_t total = length * extent;
        uint64_t threadsPerBlock = threadsPerBlockFor(total, elementsPerThread);
        uint64_t blocksPerGrid   = blocksPerGridFor(total, threadsPerBlock, elementsPerThread);

        dim3 block(threadsPerBlock);
        dim3 grid(blocksPerGrid);

        // NOTE - for debug printing
        // std::cerr << "CudaPermutedExtentsAssign: threadsPerBlock=" << threadsPerBlock << " blocksPerGrid=" << blocksPerGrid
        //     << " extent=" << extent << " length=" << length << " permsize=" << permsize << " perm=" << perm << " po=" << po
        //     << " rsize=" << rsize << " r=" << r << " ro=" << ro << " rfs=" << rfs << " rps=" << rps
        //     << " ssize=" << ssize << " s=" << s << " so=" << so << " ss=" << ss << " sc=" << sc
        //     << std::endl;

        CudaPermutedExtentsAssignKernel<<<grid, block, 0, stream>>>(
            r, ro, rfs, rps, s, so, ss, sc, extent, perm, po, length, elementsPerThread);
        if (_CUDA_CHECK(cudaGetLastError()) != cudaSuccess) {
            return false;
        }
        return true;
    });
    if (!success) {
        return false;
    }
    return true;
}

bool CudaCircularCopy(DoContext* ctx, uint32_t* r, const uint32_t* v, uint64_t length) {
    CudaVectorGraph* graph = CudaVectorGraph::Get(ctx);
    if (!graph) return false;

    int elementsPerThread = graph->GetParams().elementsPerThread;

    if (length == 0 || elementsPerThread <= 0) return false;

    return graph->Compute({v}, {r}, [r, v, length, elementsPerThread](cudaStream_t stream) {
        uint64_t total = (uint64_t)length * length;
        uint64_t threadsPerBlock = threadsPerBlockFor(total, elementsPerThread);
        uint64_t blocksPerGrid   = blocksPerGridFor(total, threadsPerBlock, elementsPerThread);

        dim3 block(threadsPerBlock);
        dim3 grid(blocksPerGrid);

        CudaCircularCopyKernel<<<grid, block, 0, stream>>>(r, v, length, elementsPerThread);
        if (_CUDA_CHECK(cudaGetLastError()) != cudaSuccess) {
            return false;
        }

        return true;
    });
}

bool CudaMatrixTranspose1(DoContext* ctx, uint32_t* r, const uint32_t* a, uint64_t rows, uint64_t cols) {
    CudaVectorGraph* graph = CudaVectorGraph::Get(ctx);
    if (!graph) return false;

    int elementsPerThread = graph->GetParams().elementsPerThread;

    if (rows == 0 || cols == 0 || elementsPerThread <= 0) return false;

    return graph->Compute({a}, {r}, [r, a, rows, cols, elementsPerThread](cudaStream_t stream) {
        // Choose tile size adaptively
        uint64_t tileCols = threadsPerBlockFor(cols, 1, defaultMaxThreads2D);
        uint64_t tileRows = threadsPerBlockFor(rows, 1, defaultMaxThreads2D);

        // Ensure tileRows is at least BLOCK_ROWS
        tileRows = std::max(tileRows, BLOCK_ROWS);

        // Grid dimensions: cover full matrix
        uint64_t gridX = blocksPerGridFor(cols, tileCols, 1);
        uint64_t gridY = blocksPerGridFor(rows, tileRows, 1);

        dim3 block(tileCols, BLOCK_ROWS);
        dim3 grid(gridX, gridY);

        size_t sharedMemSize = (tileRows + 1) * tileCols * sizeof(uint32_t);

        CudaMatrixTranspose1Kernel<<<grid, block, sharedMemSize, stream>>>(
            r, a, rows, cols,
            tileRows, tileCols,
            elementsPerThread
        );
        if (_CUDA_CHECK(cudaGetLastError()) != cudaSuccess) {
            return false;
        }
        return true;
    });
}
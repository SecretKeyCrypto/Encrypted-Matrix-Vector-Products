#include <iostream>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cassert>
#include <mutex>
#include <chrono>
#include <cstdlib>
#include <cstdio>

#include "mvp.h"
#include "../dataobjects/dataobj.h"
#include "../dataobjects/fields.h"

// -----------------------------------------------------------------------------
// Core templated implementation (no C linkage)
// -----------------------------------------------------------------------------
// K       = number of blocks grouped per row pass
// U       = inner unroll factor (columns per iteration)
// PF_DIST = prefetch distance in cache lines

template <int K, int U, int PF_DIST>
static void BlockMatVecProduct_Impl(
    const uint32_t* __restrict__ mat,    // row-major n×m
    const uint32_t* __restrict__ vec,    // length-m
    uint32_t*       __restrict__ result, // length-(s×n), row-major blocks
    uint32_t n, uint32_t m,
    uint32_t s, uint32_t p
) {
    assert(m % s == 0);
    const uint32_t b = m / s;

    // zero out result
    std::memset(result, 0, size_t(s) * n * sizeof(uint32_t));

    // process K blocks at a time
    for (uint32_t bigBlk = 0; bigBlk < s; bigBlk += K) {
        int Ki = std::min<int>(K, int(s - bigBlk));
        for (uint32_t row = 0; row < n; ++row) {
            const uint32_t* row_ptr = mat + size_t(row) * m;
            uint64_t acc[K][U] = {};

            // inner unrolled loop
            uint32_t j = 0;
            for (; j + U <= b; j += U) {
                for (int k = 0; k < Ki; ++k) {
                    size_t base = size_t(bigBlk + k) * b + j;
                    __builtin_prefetch(row_ptr + base + PF_DIST, 0, 3);
                    for (int u = 0; u < U; ++u) {
                        acc[k][u] += uint64_t(row_ptr[base + u]) * vec[base + u];
                    }
                }
            }
            // remainder columns
            for (; j < b; ++j) {
                for (int k = 0; k < Ki; ++k) {
                    size_t col = size_t(bigBlk + k) * b + j;
                    acc[k][0] += uint64_t(row_ptr[col]) * vec[col];
                }
            }
            // finalize and store
            for (int k = 0; k < Ki; ++k) {
                uint64_t sum = 0;
                for (int u = 0; u < U; ++u) sum += acc[k][u];
                result[size_t(bigBlk + k) * n + row] = uint32_t(sum % p);
            }
        }
    }
}

// Not used for now, need to implement the way of tuning the parameters.
// -----------------------------------------------------------------------------
// Non-template wrapper specifying default tuning knobs K=8,U=8,PF_DIST=16
// -----------------------------------------------------------------------------
static inline void BlockMatVecProduct_Default(
    const uint32_t* mat,
    const uint32_t* vec,
    uint32_t*       result,
    uint32_t        n,
    uint32_t        m,
    uint32_t        s,
    uint32_t        p
) {
    BlockMatVecProduct_Impl<8, 8, 16>(mat, vec, result, n, m, s, p);
}

extern "C" {

void BlockMatVecProduct_UnrolledPreload(
    const uint32_t* mat,
    const uint32_t* vec,
    uint32_t*       result,
    uint32_t        n,
    uint32_t        m,
    uint32_t        s,
    uint32_t        p
) {
    BlockMatVecProduct_Default(mat, vec, result, n, m, s, p);
}

// The most straight forward way of processing Mat Vec Product in blocks, works for flexible blocksize.
void BlockMatVecProduct_StraightForward(
    const uint32_t* __restrict__ mat,    // matrix: size n × m, row-major
    const uint32_t* __restrict__ vec,    // vector: length m
    uint32_t*       __restrict__ result, // output: size s × n, row-major blocks
    uint32_t n, uint32_t m,
    uint32_t s, uint32_t p
) {
    assert(m % s == 0);         // each block has b = m / s columns
    const uint32_t b = m / s;

    // Zero out the result buffer
    std::memset(result, 0, size_t(n) * s * sizeof(uint32_t));

    for (uint32_t row = 0; row < n; ++row) {
        const uint32_t* row_ptr = mat + size_t(row) * m;

        for (uint32_t blk = 0; blk < s; ++blk) {
            uint64_t acc = 0;
            uint32_t col_start = blk * b;

            for (uint32_t j = 0; j < b; ++j) {
                uint32_t col = col_start + j;
                acc += uint64_t(row_ptr[col]) * vec[col];
            }

            // Store result at [blk][row] in block-row-major layout
            result[size_t(blk) * n + row] = uint32_t(acc % p);
        }
    }
}

// If the matrix are already stored by blocks.
void BlockMatVecProduct(const uint32_t* __restrict__ mat,    // matrix: size n × m, block-wise, row-major
    const uint32_t* __restrict__ vec,    // vector: length m
    uint32_t*       __restrict__ result, // output: size s × n, row-major blocks
    uint32_t n, uint32_t m,
    uint32_t s, uint32_t p
) {
    assert(m % s == 0);
    uint32_t b = m / s;  // columns per block

    const uint32_t* mat_blk = mat;
    const uint32_t* vec_blk = vec;
    uint32_t* result_blk = result;
    for (uint32_t blk = 0; blk < s; ++blk, mat_blk += n * b, vec_blk += b, result_blk += n) {
        // mat_blk is n × b (row-major)
        MatVecProduct(mat_blk, vec_blk, result_blk, n, b, p);
    }
}


// M x v
void MatVecProduct(const uint32_t* mat, const uint32_t* vec, uint32_t* result, uint32_t n, uint32_t m, uint32_t p)
{
    if (USE_FAST_CODE) {
        assert(uint64_t(p-1) * uint64_t(m) < 1ULL << 31);
        memset(result, 0, sizeof(result[0]) * n);
        alignas(64) uint32_t prod[m];
        for (int row = 0; row < n; ++row) {
            const uint32_t* row_ptr = mat + row * m;

            FieldMulVectors(prod, 0, row_ptr, 0, vec, 0, m, p);
            for (int col = 0; col < m; ++col) {
                // TODO: the next line (since prod is reduced modulu p) would overflow when (p-1)*m >= (1<<31)
                result[row] += prod[col];
            }
        }
        FieldModVector(result, 0, n, p);
    } else {
        for (int row = 0; row < n; ++row) {
            const uint32_t* row_ptr = mat + row * m;

            uint64_t acc = 0;
            for (int col = 0; col < m; ++col) {
                acc = (acc + ((uint64_t)row_ptr[col] * vec[col]));
            }

            result[row] = acc % p;
        }
    }
}

void BlockVecMatProduct(const uint32_t* mat, const uint32_t* vec, uint32_t* result, uint32_t n, uint32_t m, uint32_t s, uint32_t p)
{
    assert(n % s == 0);
    int b = n / s;

    // 1) zero out the final 32-bit result buffer
    std::memset(result, 0, sizeof(uint32_t) * size_t(s) * m);

    // 2) temp 64-bit accumulators: one per column in a block
    alignas(64) uint64_t acc[m];

    for (uint32_t blk = 0; blk < s; ++blk) {
        uint32_t row_start = blk * b;
        uint32_t* res_ptr  = result + blk * size_t(m);

        // reset accumulators to zero
        memset(acc, 0, sizeof(acc));

        // accumulate products for all b rows in this block
        for (uint32_t i = 0; i < b; ++i) {
            uint32_t row = row_start + i;
            uint32_t v   = vec[row];
            const uint32_t* row_ptr = mat + size_t(row) * m;

            for (uint32_t col = 0; col < m; ++col) {
                acc[col] += uint64_t(row_ptr[col]) * v;
            }
        }

        // reduce each accumulator exactly once
        for (uint32_t col = 0; col < m; ++col) {
            res_ptr[col] = uint32_t(acc[col] % p);
        }
    }
}

}
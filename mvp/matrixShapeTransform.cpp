// transform.cpp
#include <cstdint>
#include <cstring>
#include "matrixShapeTransform.h"
#include <cassert>

// FIXME - make fast on GPU
void TransformRowMajorToBlockRowMajor(
    const uint32_t* mat,        // input: size n × m, row-major
    uint32_t* matBlocked,       // output: size n × m, block-row-major
    uint32_t n, uint32_t m, uint32_t s
) {
    assert(m % s == 0);
    uint32_t b = m / s;

    for (uint32_t row = 0; row < n; ++row) {
        for (uint32_t blk = 0; blk < s; ++blk) {
            for (uint32_t j = 0; j < b; ++j) {
                uint32_t orig_col = blk * b + j;
                uint32_t val = mat[row * m + orig_col];  // from row-major

                // write into block-row-major
                size_t dest_idx = ((blk * n + row) * b) + j;
                matBlocked[dest_idx] = val;
            }
        }
    }
}

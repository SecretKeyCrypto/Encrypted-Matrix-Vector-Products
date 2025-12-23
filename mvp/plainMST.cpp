#include <cstdint>
#include <cstring>
#include "plainMST.h"
#include <cassert>

#ifdef __cplusplus
extern "C" {
#endif

bool PlainTransformRowMajorToBlockRowMajor(
    DoContext* doctx,
    const uint32_t* mat,        // input: size n × m, row-major
    uint32_t mo, uint32_t ms,   // offset and stride for mat
    uint32_t* matBlocked,       // output: size n × m, block-row-major
    uint32_t bo, uint32_t bs,   // offset and stride for matBlocked
    uint32_t n, uint32_t m, uint32_t s
) {
    assert(m % s == 0);
    uint32_t b = m / s;

    for (uint32_t row = 0; row < n; ++row) {
        for (uint32_t blk = 0; blk < s; ++blk) {
            for (uint32_t j = 0; j < b; ++j) {
                uint32_t orig_col = blk * b + j;
                uint32_t val = mat[mo + row * ms + orig_col];  // from row-major

                // write into block-row-major
                size_t dest_idx = ((bo + blk * bs + row) * b) + j;
                matBlocked[dest_idx] = val;
            }
        }
    }
    return true;
}

#ifdef __cplusplus
} // extern "C"
#endif

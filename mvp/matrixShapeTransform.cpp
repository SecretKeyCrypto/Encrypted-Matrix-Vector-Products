#ifdef USE_FAST_CODE_WITH_CUDA

#include "cudaMST.h"
#include "plainMST.h"
#include "../dataobjects/common.h"

#ifdef __cplusplus
extern "C" {
#endif

bool TransformRowMajorToBlockRowMajor(
    DoContext* doctx,
    const uint32_t* mat,        // input: size n × m, row-major
    uint32_t mo, uint32_t ms,   // offset and stride for mat
    uint32_t* matBlocked,       // output: size n × m, block-row-major
    uint32_t bo, uint32_t bs,   // offset and stride for matBlocked
    uint32_t n, uint32_t m, uint32_t s
) {
    return CUDA_CALL(TransformRowMajorToBlockRowMajor(doctx, mat, mo, ms, matBlocked, bo, bs, n, m, s));
}

#ifdef __cplusplus
} // extern "C"
#endif

#else /* USE_FAST_CODE_WITH_CUDA */

#include <cstdint>
#include <cstring>
#include "plainMST.h"
#include <cassert>

#ifdef __cplusplus
extern "C" {
#endif

bool TransformRowMajorToBlockRowMajor(
    DoContext* doctx,
    const uint32_t* mat,        // input: size n × m, row-major
    uint32_t mo, uint32_t ms,   // offset and stride for mat
    uint32_t* matBlocked,       // output: size n × m, block-row-major
    uint32_t bo, uint32_t bs,   // offset and stride for matBlocked
    uint32_t n, uint32_t m, uint32_t s
) {
    return PlainTransformRowMajorToBlockRowMajor(doctx, mat, mo, ms, matBlocked, bo, bs, n, m, s);
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif

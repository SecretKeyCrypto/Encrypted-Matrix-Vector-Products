#include <cstdint>
constexpr uint64_t TILE_DIM = 32;

#ifdef USE_FAST_CODE_WITH_CUDA

#include "cudatdm.h"
#include "plaintdm.h"
#include "../dataobjects/common.h"

#ifdef __cplusplus
extern "C" {
#endif

bool PermutedExtentsAssign(
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
    return CUDA_CALL(PermutedExtentsAssign(ctx, r, ro, rfs, rps, s, so, ss, sc, extent, perm, po, length));
}

bool CircularCopy(DoContext* ctx, uint32_t* r, const uint32_t* v, uint64_t length) {
    return CUDA_CALL(CircularCopy(ctx, r, v, length));
}

#ifdef __cplusplus
} // extern "C"
#endif

#else /* USE_FAST_CODE_WITH_CUDA */

#include "plaintdm.h"

#ifdef __cplusplus
extern "C" {
#endif

bool PermutedExtentsAssign(
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
    return PlainPermutedExtentsAssign(ctx, r, ro, rfs, rps, s, so, ss, sc, extent, perm, po, length);
}

bool CircularCopy(DoContext* ctx, uint32_t* r, const uint32_t* v, uint64_t length) {
    return PlainCircularCopy(ctx, r, v, length);
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* USE_FAST_CODE_WITH_CUDA */
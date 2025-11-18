#ifdef USE_FAST_CODE_WITH_CUDA

#include "cudamvp.h"
#include "plainmvp.h"
#include "../dataobjects/common.h"

#ifdef __cplusplus
extern "C" {
#endif

bool BlockMatVecProduct(DoContext* ctx, const uint32_t* mat, const uint32_t* vec, uint32_t* result, uint32_t n, uint32_t m, uint32_t s, uint32_t p) {
    return CUDA_CALL(BlockMatVecProduct(ctx, mat, vec, result, n, m, s, p));
}

bool MatVecProduct(DoContext* ctx, const uint32_t* mat, const uint32_t* vec, uint32_t* result, uint32_t n, uint32_t m, uint32_t p) {
    return CUDA_CALL(MatVecProduct(ctx, mat, vec, result, n, m, p));
}

bool MatVecProductExt(
    DoContext* ctx,
    const uint32_t* mat, uint32_t mo, uint32_t ms,
    const uint32_t* vec, uint32_t vo, uint32_t vs,
    uint32_t* result, uint32_t ro, uint32_t rs,
    uint32_t n, uint32_t m, uint32_t steps, uint32_t p)
{
    return CUDA_CALL(MatVecProductExt(ctx, mat, mo, ms, vec, vo, vs, result, ro, rs, n, m, steps, p));
}

bool BlockVecMatProduct(DoContext* ctx, const uint32_t* mat, const uint32_t* vec, uint32_t* result, uint32_t n, uint32_t m, uint32_t s, uint32_t p) {
    return CUDA_CALL(BlockVecMatProduct(ctx, mat, vec, result, n, m, s, p));
}

#ifdef __cplusplus
}
#endif

#else /* USE_FAST_CODE_WITH_CUDA */

#include "plainmvp.h"

#ifdef __cplusplus
extern "C" {
#endif

bool BlockMatVecProduct(DoContext* ctx, const uint32_t* mat, const uint32_t* vec, uint32_t* result, uint32_t n, uint32_t m, uint32_t s, uint32_t p) {
    return PlainBlockMatVecProduct(ctx, mat, vec, result, n, m, s, p);
}

bool MatVecProduct(DoContext* ctx, const uint32_t* mat, const uint32_t* vec, uint32_t* result, uint32_t n, uint32_t m, uint32_t p) {
    return PlainMatVecProduct(ctx, mat, vec, result, n, m, p);
}

bool MatVecProductExt(
    DoContext* ctx,
    const uint32_t* mat, uint32_t mo, uint32_t ms,
    const uint32_t* vec, uint32_t vo, uint32_t vs,
    uint32_t* result, uint32_t ro, uint32_t rs,
    uint32_t n, uint32_t m, uint32_t steps, uint32_t p)
{
    return PlainMatVecProductExt(ctx, mat, mo, ms, vec, vo, vs, result, ro, rs, n, m, steps, p);
}

bool BlockVecMatProduct(DoContext* ctx, const uint32_t* mat, const uint32_t* vec, uint32_t* result, uint32_t n, uint32_t m, uint32_t s, uint32_t p) {
    return PlainBlockVecMatProduct(ctx, mat, vec, result, n, m, s, p);
}

#ifdef __cplusplus
} //extern "C"
#endif

#endif /* USE_FAST_CODE_WITH_CUDA */

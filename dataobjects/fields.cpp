#ifdef USE_FAST_CODE_WITH_CUDA

#include "cudafields.h"
#include "plainfields.h"
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

bool FieldRangeVector(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    uint32_t start, uint64_t length
) {
    return CUDA_CALL(FieldRangeVector(ctx, r, ro, start, length));
}

bool FieldCopyVector(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    uint64_t length
) {
    return CUDA_CALL(FieldCopyVector(ctx, r, ro, a, ao, length));
}

bool FieldSetVector(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    uint64_t length, uint32_t v
) {
    return CUDA_CALL(FieldSetVector(ctx, r, ro, length, v));
}

bool FieldAddToVector(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    uint32_t v,
    uint64_t length
) {
    return CUDA_CALL(FieldAddToVector(ctx, r, ro, v, length));
}

bool FieldAddVectors(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    const uint32_t* b, uint64_t bo,
    uint64_t length, uint32_t p
) {
    return CUDA_CALL(FieldAddVectors(ctx, r, ro, a, ao, b, bo, length, p));
}

bool FieldMulVector(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    uint32_t b,
    uint64_t length, uint32_t p
) {
    return CUDA_CALL(FieldMulVector(ctx, r, ro, a, ao, b, length, p));
}

bool FieldMulVectors(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    const uint32_t* b, uint64_t bo,
    uint64_t length, uint32_t p
) {
    return CUDA_CALL(FieldMulVectors(ctx, r, ro, a, ao, b, bo, length, p));
}

bool FieldSubVectors(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    const uint32_t* b, uint64_t bo,
    uint64_t length, uint32_t p
) {
    return CUDA_CALL(FieldSubVectors(ctx, r, ro, a, ao , b, bo, length, p));
}

bool FieldNegVector(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    uint64_t length, uint32_t p
) {
    return CUDA_CALL(FieldNegVector(ctx, r, ro, length, p));
}

bool FieldAddVectorIfNonZero(
    DoContext* ctx,
    bool* t, uint64_t t_index,
    uint32_t* r, uint64_t ro,
    const uint32_t* e, uint64_t eo,
    uint64_t length, uint32_t p
) {
    return CUDA_CALL(FieldAddVectorIfNonZero(ctx, t, t_index, r, ro, e, eo, length, p));
}

bool FieldAddVectorIfNonZeroExt(
    DoContext* ctx,
    bool* t, uint64_t to, uint64_t ts,
    uint32_t* r, uint64_t ro, uint64_t rs,
    const uint32_t* e, uint64_t eo, uint64_t es,
    uint64_t length, uint64_t steps, uint32_t p
) {
    return CUDA_CALL(FieldAddVectorIfNonZeroExt(ctx, t, to, ts, r, ro, rs, e, eo, es, length, steps, p));
}

bool FieldInvVector(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    uint64_t length, uint32_t p
) {
    return CUDA_CALL(FieldInvVector(ctx, r, ro, a, ao, length, p));
}

bool FieldAddVectorsExt(
    DoContext* ctx,
    uint32_t* r, uint64_t ro, uint64_t rs,
    const uint32_t* a, uint64_t ao, uint64_t as,
    const uint32_t* b, uint64_t bo, uint64_t bs,
    uint64_t length, uint64_t steps, uint32_t p
) {
    return CUDA_CALL(FieldAddVectorsExt(ctx, r, ro, rs, a, ao, as, b, bo, bs, length, steps, p));
}

bool FieldMulVectorsExt(
    DoContext* ctx,
    uint32_t* r, uint64_t ro, uint64_t rs,
    const uint32_t* a, uint64_t ao, uint64_t as,
    const uint32_t* b, uint64_t bo, uint64_t bs,
    uint64_t length, uint64_t steps, uint32_t p
) {
    return CUDA_CALL(FieldMulVectorsExt(ctx, r, ro, rs, a, ao, as, b, bo, bs, length, steps, p));
}

bool FieldNegVectorExt(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    uint64_t length, uint64_t stride, uint64_t steps, uint32_t p
) {
    return CUDA_CALL(FieldNegVectorExt(ctx, r, ro, length, stride, steps, p));
}

#ifdef __cplusplus
} /* extern "C" */
#endif


#else /* USE_FAST_CODE_WITH_CUDA */

#include "plainfields.h"

#ifdef __cplusplus
extern "C" {
#endif

bool FieldRangeVector(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    uint32_t start, uint64_t length
) {
    return PlainFieldRangeVector(ctx, r, ro, start, length);
}

bool FieldCopyVector(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    uint64_t length
) {
    return PlainFieldCopyVector(ctx, r, ro, a, ao, length);
}

bool FieldSetVector(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    uint64_t length, uint32_t v
) {
    return PlainFieldSetVector(ctx, r, ro, length, v);
}

bool FieldAddToVector(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    uint32_t v,
    uint64_t length
) {
    return PlainFieldAddToVector(ctx, r, ro, v, length);
}

bool FieldAddVectors(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    const uint32_t* b, uint64_t bo,
    uint64_t length, uint32_t p
) {
    return PlainFieldAddVectors(ctx, r, ro, a, ao, b, bo, length, p);
}

bool FieldMulVector(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    uint32_t b,
    uint64_t length, uint32_t p
) {
    return PlainFieldMulVector(ctx, r, ro, a, ao, b, length, p);
}

bool FieldMulVectors(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    const uint32_t* b, uint64_t bo,
    uint64_t length, uint32_t p
) {
    return PlainFieldMulVectors(ctx, r, ro, a, ao, b, bo, length, p);
}

bool FieldSubVectors(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    const uint32_t* b, uint64_t bo,
    uint64_t length, uint32_t p
) {
    return PlainFieldSubVectors(ctx, r, ro, a, ao , b, bo, length, p);
}

bool FieldNegVector(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    uint64_t length, uint32_t p
) {
    return PlainFieldNegVector(ctx, r, ro, length, p);
}

bool FieldAddVectorIfNonZero(
    DoContext* ctx,
    bool* t, uint64_t t_index,
    uint32_t* r, uint64_t ro,
    const uint32_t* e, uint64_t eo,
    uint64_t length, uint32_t p
) {
    return PlainFieldAddVectorIfNonZero(ctx, t, t_index, r, ro, e, eo, length, p);
}

bool FieldInvVector(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    uint64_t length, uint32_t p
) {
    return PlainFieldInvVector(ctx, r, ro, a, ao, length, p);
}

bool FieldAddVectorsExt(
    DoContext* ctx,
    uint32_t* r, uint64_t ro, uint64_t rs,
    const uint32_t* a, uint64_t ao, uint64_t as,
    const uint32_t* b, uint64_t bo, uint64_t bs,
    uint64_t length, uint64_t steps, uint32_t p
) {
    return PlainFieldAddVectorsExt(ctx, r, ro, rs, a, ao, as, b, bo, bs, length, steps, p);
}

bool FieldMulVectorsExt(
    DoContext* ctx,
    uint32_t* r, uint64_t ro, uint64_t rs,
    const uint32_t* a, uint64_t ao, uint64_t as,
    const uint32_t* b, uint64_t bo, uint64_t bs,
    uint64_t length, uint64_t steps, uint32_t p
) {
    return PlainFieldMulVectorsExt(ctx, r, ro, rs, a, ao, as, b, bo, bs, length, steps, p);
}

bool FieldNegVectorExt(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    uint64_t length, uint64_t stride, uint64_t steps, uint32_t p
) {
    return PlainFieldNegVectorExt(ctx, r, ro, length, stride, steps, p);
}

bool FieldAddVectorIfNonZeroExt(
    DoContext* ctx,
    bool* t, uint64_t to, uint64_t ts,
    uint32_t* r, uint64_t ro, uint64_t rs,
    const uint32_t* e, uint64_t eo, uint64_t es,
    uint64_t length, uint64_t steps, uint32_t p
) {
    return PlainFieldAddVectorIfNonZeroExt(ctx, t, to, ts, r, ro, rs, e, eo, es, length, steps, p);
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* USE_FAST_CODE_WITH_CUDA */
#ifndef _CUDAFIELDS_H
#define _CUDAFIELDS_H

#include <cstdint>

#include "docontext.h"

#ifdef __cplusplus
extern "C" {
#endif

bool CudaFieldRangeVector(DoContext* ctx, uint32_t* r, uint64_t ro, uint32_t start, uint64_t length);
bool CudaFieldCopyVector(DoContext* ctx, uint32_t* r, uint64_t ro, const uint32_t* a, uint64_t ao, uint64_t length);
bool CudaFieldSetVector(DoContext* ctx, uint32_t* r, uint64_t ro, uint64_t length, uint32_t v);
bool CudaFieldAddToVector(DoContext* ctx, uint32_t* r, uint64_t ro, uint32_t v, uint64_t length);
bool CudaFieldAddVectors(DoContext* ctx, uint32_t* r, uint64_t ro, const uint32_t* a, uint64_t ao, const uint32_t* b, uint64_t bo, uint64_t length, uint32_t p);
bool CudaFieldMulVector(DoContext* ctx, uint32_t* r, uint64_t ro, const uint32_t* a, uint64_t ao, uint32_t b, uint64_t length, uint32_t p);
bool CudaFieldMulVectors(DoContext* ctx, uint32_t* r, uint64_t ro, const uint32_t* a, uint64_t ao, const uint32_t* b, uint64_t bo, uint64_t length, uint32_t p);
bool CudaFieldSubVectors(DoContext* ctx, uint32_t* r, uint64_t ro, const uint32_t* a, uint64_t ao, const uint32_t* b, uint64_t bo, uint64_t length, uint32_t p);
bool CudaFieldNegVector(DoContext* ctx, uint32_t* r, uint64_t ro, uint64_t length, uint32_t p);
bool CudaFieldIsZeroVector(DoContext* ctx, bool *t, const uint32_t* e, uint64_t length);
bool CudaFieldAddVectorIfNonZero(DoContext* ctx, bool* t, uint64_t t_index, uint32_t* r, uint64_t ro, const uint32_t* e, uint64_t eo, uint64_t length, uint32_t p);
bool CudaFieldInvVector(DoContext* ctx, uint32_t* r, uint64_t ro, const uint32_t* a, uint64_t ao, uint64_t length, uint32_t p);

bool CudaFieldAddVectorsExt(
    DoContext* ctx,
    uint32_t* r, uint64_t ro, uint32_t rs,
    const uint32_t* a, uint64_t ao, uint32_t as,
    const uint32_t* b, uint64_t bo, uint32_t bs,
    uint64_t length, uint64_t steps, uint32_t p
);
bool CudaFieldMulVectorExt(
    DoContext* ctx, uint32_t* r, uint64_t ro, const uint32_t* a, uint64_t ao, uint32_t b, uint64_t length, uint64_t stride, uint64_t steps, uint32_t p
);
bool CudaFieldMulVectorsExt(
    DoContext* ctx,
    uint32_t* r, uint64_t ro, uint32_t rs,
    const uint32_t* a, uint64_t ao, uint32_t as,
    const uint32_t* b, uint64_t bo, uint32_t bs,
    uint64_t length, uint64_t steps, uint32_t p
);
bool CudaFieldNegVectorExt(DoContext* ctx, uint32_t* r, uint64_t ro, uint64_t length, uint64_t stride, uint64_t steps, uint32_t p);
bool CudaFieldAddVectorIfNonZeroExt(
    DoContext* ctx,
    bool* t, uint64_t to, uint64_t ts,
    uint32_t* r, uint64_t ro, uint64_t rs,
    const uint32_t* e, uint64_t eo, uint64_t es,
    uint64_t length, uint64_t steps, uint32_t p
);

#ifdef __cplusplus
}
#endif

#endif /* _CUDAFIELDS_H */
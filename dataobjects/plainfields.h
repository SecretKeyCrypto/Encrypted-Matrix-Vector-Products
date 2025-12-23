#ifndef _PLAINFIELDS_H
#define _PLAINFIELDS_H

#include <cstdint>
#include "docontext.h"

#ifdef __cplusplus
extern "C" {
#endif

bool PlainFieldRangeVector(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    uint32_t start, uint64_t length
);

bool PlainFieldCopyVector(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    uint64_t length
);

bool PlainFieldSetVector(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    uint64_t length, uint32_t v
);

bool PlainFieldAddToVector(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    uint32_t v,
    uint64_t length
);

bool PlainFieldAddVectors(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    const uint32_t* b, uint64_t bo,
    uint64_t length, uint32_t p
);

bool PlainFieldAddVectorsExt(
    DoContext* ctx,
    uint32_t* r, uint64_t ro, uint64_t rs,
    const uint32_t* a, uint64_t ao, uint64_t as,
    const uint32_t* b, uint64_t bo, uint64_t bs,
    uint64_t length, uint64_t steps, uint32_t p
);

bool PlainFieldMulVector(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    uint32_t b,
    uint64_t length, uint32_t p
);

bool PlainFieldMulVectors(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    const uint32_t* b, uint64_t bo,
    uint64_t length, uint32_t p
);

bool PlainFieldMulVectorsExt(
    DoContext* ctx,
    uint32_t* r, uint64_t ro, uint64_t rs,
    const uint32_t* a, uint64_t ao, uint64_t as,
    const uint32_t* b, uint64_t bo, uint64_t bs,
    uint64_t length, uint64_t steps, uint32_t p
);

bool PlainFieldSubVectors(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    const uint32_t* b, uint64_t bo,
    uint64_t length, uint32_t p
);

bool PlainFieldNegVector(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    uint64_t length, uint32_t p
);

bool PlainFieldNegVectorExt(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    uint64_t length, uint64_t stride, uint64_t steps, uint32_t p
);

bool PlainFieldIsZeroVector(
    DoContext* ctx,
    bool *t, uint64_t t_index,
    const uint32_t* e, uint64_t eo, uint64_t length
);

bool PlainFieldAddVectorIfNonZero(
    DoContext* ctx,
    bool* t, uint64_t t_index,
    uint32_t* r, uint64_t ro,
    const uint32_t* e, uint64_t eo,
    uint64_t length, uint32_t p
);

bool PlainFieldAddVectorIfNonZeroExt(
    DoContext* ctx,
    bool* t, uint64_t to, uint64_t ts,
    uint32_t* r, uint64_t ro, uint64_t rs,
    const uint32_t* e, uint64_t eo, uint64_t es,
    uint64_t length, uint64_t steps, uint32_t p
);

bool PlainFieldInvVector(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    uint64_t length, uint32_t p
);

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* _PLAINFIELDS_H */
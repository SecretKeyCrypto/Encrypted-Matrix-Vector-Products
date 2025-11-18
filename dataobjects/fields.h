#ifndef _FIELDS_H_
#define _FIELDS_H_

#include <stdint.h>
#include <stdbool.h>
#include "docontext.h"

#ifdef __cplusplus
extern "C" {
#endif

bool FieldRangeVector(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    uint32_t start, uint64_t length
);

bool FieldCopyVector(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    uint64_t length
);

bool FieldSetVector(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    uint64_t length, uint32_t v
);

bool FieldAddToVector(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    uint32_t v,
    uint64_t length
);

bool FieldAddVectors(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    const uint32_t* b, uint64_t bo,
    uint64_t length, uint32_t p
);

bool FieldAddVectorsExt(
    DoContext* ctx,
    uint32_t* r, uint64_t ro, uint64_t rs,
    const uint32_t* a, uint64_t ao, uint64_t as,
    const uint32_t* b, uint64_t bo, uint64_t bs,
    uint64_t length, uint64_t steps, uint32_t p
);

bool FieldMulVector(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    uint32_t b,
    uint64_t length, uint32_t p
);

bool FieldMulVectors(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    const uint32_t* b, uint64_t bo,
    uint64_t length, uint32_t p
);

bool FieldMulVectorsExt(
    DoContext* ctx,
    uint32_t* r, uint64_t ro, uint64_t rs,
    const uint32_t* a, uint64_t ao, uint64_t as,
    const uint32_t* b, uint64_t bo, uint64_t bs,
    uint64_t length, uint64_t steps, uint32_t p
);

bool FieldSubVectors(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    const uint32_t* b, uint64_t bo,
    uint64_t length, uint32_t p
);

bool FieldNegVector(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    uint64_t length, uint32_t p
);

bool FieldNegVectorExt(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    uint64_t length, uint64_t stride, uint64_t steps, uint32_t p
);

bool FieldIsZeroVector(
    DoContext* ctx,
    bool *t, uint64_t t_index,
    const uint32_t* e, uint64_t eo, uint64_t length
);

bool FieldAddVectorIfNonZero(
    DoContext* ctx,
    bool* t, uint64_t t_index,
    uint32_t* r, uint64_t ro,
    const uint32_t* e, uint64_t eo,
    uint64_t length, uint32_t p
);

bool FieldAddVectorIfNonZeroExt(
    DoContext* ctx,
    bool* t, uint64_t to, uint64_t ts,
    uint32_t* r, uint64_t ro, uint64_t rs,
    const uint32_t* e, uint64_t eo, uint64_t es,
    uint64_t length, uint64_t steps, uint32_t p
);

bool FieldInvVector(
    DoContext* ctx,
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    uint64_t length, uint32_t p
);

#ifdef __cplusplus
}
#endif

#endif /* _FIELDS_H_ */
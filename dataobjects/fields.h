#ifndef _FIELDS_H
#define _FIELDS_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

int FieldRangeVector(
    uint32_t* r, uint64_t ro,
    uint32_t start, uint64_t length
);

int FieldCopyVector(
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    uint64_t length
);

int FieldSetVector(
    uint32_t* r, uint64_t ro,
    uint64_t length, uint32_t v
);

int FieldAddToVector(
    uint32_t* r, uint64_t ro,
    uint32_t v,
    uint64_t length
);

int FieldAddVectors(
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    const uint32_t* b, uint64_t bo,
    uint64_t length, uint32_t p
);

int FieldMulVector(
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    uint32_t b,
    uint64_t length, uint32_t p
);

int FieldMulVectors(
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    const uint32_t* b, uint64_t bo,
    uint64_t length, uint32_t p
);

int FieldSubVectors(
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    const uint32_t* b, uint64_t bo,
    uint64_t length, uint32_t p
);

int FieldNegVector(
    uint32_t* r, uint64_t ro,
    uint64_t length, uint32_t p
);

int FieldIsZeroVector(
    bool *t, uint64_t t_index,
    const uint32_t* e, uint64_t eo, uint64_t length
);

int FieldAddVectorIfNonZero(
    bool* t, uint64_t t_index,
    uint32_t* r, uint64_t ro,
    const uint32_t* e, uint64_t eo,
    uint64_t length, uint32_t p
);

#ifdef __cplusplus
}
#endif

#endif /* _FIELDS_H */
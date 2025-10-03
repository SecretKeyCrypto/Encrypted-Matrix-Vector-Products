#ifndef _FIELDS_H
#define _FIELDS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void FieldModVector(
    uint32_t* r, uint64_t ro,
    uint64_t length, uint32_t p
);

void FieldAddVectors(
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    const uint32_t* b, uint64_t bo,
    uint64_t length, uint32_t p
);

void FieldMulVector(
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    uint32_t b,
    uint64_t length, uint32_t p
);

void FieldMulVectors(
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    const uint32_t* b, uint64_t bo,
    uint64_t length, uint32_t p
);

void FieldSubVectors(
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    const uint32_t* b, uint64_t bo,
    uint64_t length, uint32_t p
);

void FieldNegVector(
    uint32_t* r, uint64_t ro,
    uint64_t length, uint32_t p
);

#ifdef __cplusplus
}
#endif

#endif /* _FIELDS_H */
#ifndef _CUDAFIELDS_H
#define _CUDAFIELDS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int CudaFieldAddVectors(
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    const uint32_t* b, uint64_t bo,
    uint64_t length, uint32_t p
);

int CudaFieldMulVector(
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    uint32_t b,
    uint64_t length, uint32_t p
);

int CudaFieldMulVectors(
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    const uint32_t* b, uint64_t bo,
    uint64_t length, uint32_t p
);

int CudaFieldSubVectors(
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    const uint32_t* b, uint64_t bo,
    uint64_t length, uint32_t p
);

int CudaFieldNegVector(
    uint32_t* r, uint64_t ro,
    uint64_t length, uint32_t p
);

#ifdef __cplusplus
}
#endif

#endif /* _CUDAFIELDS_H */
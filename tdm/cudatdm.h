#ifndef _CUDATDM_H_
#define _CUDATDM_H_

#include <stdint.h>
#include "../dataobjects/docontext.h"

#ifdef __cplusplus
extern "C" {
#endif

bool CudaPermutedExtentsAssign(
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
    uint64_t length);

bool CudaCircularCopy(DoContext* ctx, uint32_t* r, const uint32_t* v, uint64_t length);

bool CudaMatrixTranspose1(DoContext* ctx, uint32_t* r, const uint32_t* a, uint64_t rows, uint64_t cols);

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* _CUDATDM_H_ */
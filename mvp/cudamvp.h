#ifndef _CUDAMVP_H_
#define _CUDAMVP_H_

#include <stdint.h>
#include "../dataobjects/docontext.h"

#ifdef __cplusplus
extern "C" {
#endif

bool CudaBlockMatVecProduct(DoContext* ctx, const uint32_t* mat, const uint32_t* vec, uint32_t* result, uint32_t n, uint32_t m, uint32_t s, uint32_t p);

bool CudaMatVecProduct(DoContext* ctx, const uint32_t* mat, const uint32_t* vec, uint32_t* result, uint32_t n, uint32_t m, uint32_t p);

bool CudaMatVecProductExt(
    DoContext* ctx,
    const uint32_t* mat, uint32_t mo, uint32_t ms,
    const uint32_t* vec, uint32_t vo, uint32_t vs,
    uint32_t* result, uint32_t ro, uint32_t rs,
    uint32_t n, uint32_t m, uint32_t steps, uint32_t p
);

bool CudaBlockVecMatProduct(DoContext* ctx, const uint32_t* mat, const uint32_t* vec, uint32_t* result, uint32_t n, uint32_t m, uint32_t s, uint32_t p);

#ifdef __cplusplus
}
#endif

#endif /* _CUDAMVP_H_ */
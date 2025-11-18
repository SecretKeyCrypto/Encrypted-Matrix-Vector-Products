#ifndef _PLAINMVP_H_
#define _PLAINMVP_H_

#include <stdint.h>
#include <stdbool.h>
#include "../dataobjects/docontext.h"

#ifdef __cplusplus
extern "C" {
#endif

bool PlainBlockMatVecProduct(DoContext* ctx, const uint32_t* mat, const uint32_t* vec, uint32_t* result, uint32_t n, uint32_t m, uint32_t s, uint32_t p);

bool PlainMatVecProduct(DoContext* ctx, const uint32_t* mat, const uint32_t* vec, uint32_t* result, uint32_t n, uint32_t m, uint32_t p);

bool PlainMatVecProductExt(
    DoContext* ctx,
    const uint32_t* mat, uint32_t mo, uint32_t ms,
    const uint32_t* vec, uint32_t vo, uint32_t vs,
    uint32_t* result, uint32_t ro, uint32_t rs,
    uint32_t n, uint32_t m, uint32_t steps, uint32_t p
);

bool PlainBlockVecMatProduct(DoContext* ctx, const uint32_t* mat, const uint32_t* vec, uint32_t* result, uint32_t n, uint32_t m, uint32_t s, uint32_t p);

#ifdef __cplusplus
}
#endif

#endif /* _PLAINMVP_H_ */
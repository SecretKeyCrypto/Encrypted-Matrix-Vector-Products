// transform.h
#ifndef TRANSFORM_H
#define TRANSFORM_H

#include <stdint.h>
#include "../dataobjects/docontext.h"

#ifdef __cplusplus
extern "C" {
#endif

bool TransformRowMajorToBlockRowMajor(
    DoContext* doctx,
    const uint32_t* mat,
    uint32_t mo, uint32_t ms,
    uint32_t* matBlocked,
    uint32_t bo, uint32_t bs,
    uint32_t n, uint32_t m, uint32_t s
);

#ifdef __cplusplus
}
#endif

#endif  // TRANSFORM_H

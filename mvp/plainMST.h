#ifndef _PLAINMST_H_
#define _PLAINMST_H_

#include <stdint.h>
#include "../dataobjects/docontext.h"

#ifdef __cplusplus
extern "C" {
#endif

bool PlainTransformRowMajorToBlockRowMajor(
    DoContext* ctx,
    const uint32_t* d_mat,
    uint32_t mo, uint32_t ms,
    uint32_t* d_matBlocked,
    uint32_t bo, uint32_t bs,
    uint32_t n, uint32_t m, uint32_t s
);

#ifdef __cplusplus
}
#endif

#endif /* _PLAINMST_H_ */

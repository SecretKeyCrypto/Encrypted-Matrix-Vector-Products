#ifndef _MATRICES_H_
#define _MATRICES_H_

#include <stdint.h>
#include "docontext.h"

#ifdef __cplusplus
extern "C" {
#endif

bool MatrixTranspose(DoContext* ctx, uint32_t* result, uint32_t ro, const uint32_t* matrix, uint32_t mo, uint32_t M, uint32_t N);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* _MATRICES_H_ */
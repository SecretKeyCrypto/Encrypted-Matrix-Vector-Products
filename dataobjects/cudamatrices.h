#ifndef _CUDAMATRICES_H
#define _CUDAMATRICES_H

#include <stdint.h>
#include "docontext.h"

#ifdef __cplusplus
extern "C" {
#endif

bool CudaMatrixTranspose(DoContext* ctx, uint32_t* result, uint32_t ro, const uint32_t* matrix, uint32_t mo, uint32_t M, uint32_t N);

#ifdef __cplusplus
}
#endif

#endif /* _CUDAMATRICES_H */
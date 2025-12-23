#ifdef USE_FAST_CODE_WITH_CUDA

#include "cudamatrices.h"
#include "plainmatrices.h"
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

bool MatrixTranspose(DoContext* ctx, uint32_t* result, uint32_t ro, const uint32_t* matrix, uint32_t mo, uint32_t M, uint32_t N) {
    return CUDA_CALL(MatrixTranspose(ctx, result, ro, matrix, mo, M, N));
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#else /* USE_FAST_CODE_WITH_CUDA */

#include "plainmatrices.h"

#ifdef __cplusplus
extern "C" {
#endif

bool MatrixTranspose(DoContext* ctx, uint32_t* result, uint32_t ro, const uint32_t* matrix, uint32_t mo, uint32_t M, uint32_t N) {
    return PlainMatrixTranspose(ctx, result, ro, matrix, mo, M, N);
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* USE_FAST_CODE_WITH_CUDA */
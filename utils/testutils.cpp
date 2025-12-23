#ifdef USE_FAST_CODE_WITH_CUDA

#include "cudatestutils.h"
#include "plaintestutils.h"
#include "../dataobjects/common.h"

#ifdef __cplusplus
extern "C" {
#endif

bool FieldVectorsAreEqual(
    const uint32_t* a, uint32_t ao, uint32_t as,
    const uint32_t* b, uint32_t bo, uint32_t bs,
    uint32_t length, uint32_t steps
) {
    return CUDA_CALL(FieldVectorsAreEqual(a, ao, as, b, bo, bs, length, steps));
}

#ifdef __cplusplus
} // extern "C"
#endif

#else /* USE_FAST_CODE_WITH_CUDA */

#include "plaintestutils.h"

#ifdef __cplusplus
extern "C" {
#endif

bool FieldVectorsAreEqual(
    const uint32_t* a, uint32_t ao, uint32_t as,
    const uint32_t* b, uint32_t bo, uint32_t bs,
    uint32_t length, uint32_t steps
) {
    return PlainFieldVectorsAreEqual(a, ao, as, b, bo, bs, length, steps);
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* USE_FAST_CODE_WITH_CUDA */

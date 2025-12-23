#ifdef USE_FAST_CODE_WITH_CUDA

#include "cudalpn.h"
#include "plainlpn.h"
#include "../dataobjects/common.h"

#ifdef __cplusplus
extern "C" {
#endif

bool LpnEncode(
    DoContext* ctx,
    const uint32_t* input,            // [M * L]
    const uint32_t* rlcMatrix,        // [K * L]
    const uint32_t* generatorMatrix,  // [ECCLength * M_1]
    uint32_t* encoded,                // [ECCLength * rowPerSlice * N]
    uint32_t M, uint32_t L, uint32_t K,
    uint32_t M_1, uint32_t ECCLength,
    uint32_t P
) {
    return CUDA_CALL(LpnEncode(ctx, input, rlcMatrix, generatorMatrix, encoded, M, L, K, M_1, ECCLength, P));
}

#ifdef __cplusplus
}
#endif

#else /* USE_FAST_CODE_WITH_CUDA */

#include "plainlpn.h"

#ifdef __cplusplus
extern "C" {
#endif

bool LpnEncode(
    DoContext* ctx,
    const uint32_t* input,            // [M * L]
    const uint32_t* rlcMatrix,        // [K * L]
    const uint32_t* generatorMatrix,  // [ECCLength * M_1]
    uint32_t* encoded,                // [ECCLength * M / M_1 * N]
    uint32_t M, uint32_t L, uint32_t K,
    uint32_t M_1, uint32_t ECCLength,
    uint32_t P
) {
    return PlainLpnEncode(ctx, input, rlcMatrix, generatorMatrix, encoded, M, L, K, M_1, ECCLength, P);
}

#ifdef __cplusplus
}
#endif

#endif

#include "ReedSolomon.h"

#ifdef USE_FAST_CODE_WITH_CUDA

#include "cudaRS.h"
#include "plainRS.h"
#include "../dataobjects/common.h"
#include <stdexcept>

#ifdef __cplusplus
extern "C" {
#endif

bool GenerateSystematicRSMatrix(DoContext* doctx, uint32_t n, uint32_t m, uint32_t Q, const uint32_t* alphas_in, uint32_t* output) {
    return CUDA_CALL(GenerateSystematicRSMatrix(doctx, n, m, Q, alphas_in, output));
}

bool LagrangeInterpEval(DoContext* doctx,uint32_t* result, const uint32_t* x_in, const uint32_t* y_in, uint32_t k, uint32_t eval_point, uint32_t q) {
    // return CUDA_CALL(LagrangeInterpEval(result, x_in, y_in, k, eval_point, q)); // FiXME - remove code path
    throw new std::runtime_error("not implemented - use ReedSolomonDecode");
}

bool ReedSolomonDecode(
    DoContext* doctx,
    uint32_t *code, uint64_t co, uint64_t cs,
    const bool* noisyQuery,
    uint32_t ecc_len, uint32_t ecc_k, uint32_t q,
    uint32_t* success,
    uint64_t steps
) {
    return CUDA_CALL(ReedSolomonDecode(doctx, code, co, cs, noisyQuery, ecc_len, ecc_k, q, success, steps));
}

#ifdef __cplusplus
} // extern "C"
#endif

#else /* USE_FAST_CODE_WITH_CUDA */

#include "plainRS.h"
#include <stdexcept>

#ifdef __cplusplus
extern "C" {
#endif

bool GenerateSystematicRSMatrix(DoContext* doctx, uint32_t n, uint32_t m, uint32_t Q, const uint32_t* alphas_in, uint32_t* output) {
    return PlainGenerateSystematicRSMatrix(doctx, n, m, Q, alphas_in, output);
}

bool LagrangeInterpEval(DoContext* doctx, uint32_t* result, const uint32_t* x_in, const uint32_t* y_in, uint32_t k, uint32_t eval_point, uint32_t q) {
    // return PlainLagrangeInterpEval(result, x_in, y_in, k, eval_point, q); // FiXME - remove code path
    throw new std::runtime_error("not implemented - use ReedSolomonDecode");
}

bool ReedSolomonDecode(
    DoContext* doctx,
    uint32_t *code, uint64_t co, uint64_t cs,
    const bool* noisyQuery,
    uint32_t ecc_len, uint32_t ecc_k, uint32_t q,
    uint32_t* success,
    uint64_t steps
) {
    return PlainReedSolomonDecode(doctx, code, co, cs, noisyQuery, ecc_len, ecc_k, q, success, steps);
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif
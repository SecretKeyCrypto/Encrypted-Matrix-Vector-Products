#ifndef __CUDARS_H_
#define __CUDARS_H_

#include <stdint.h>
#include "../dataobjects/docontext.h"

#ifdef __cplusplus
extern "C" {
#endif

bool CudaGenerateSystematicRSMatrix(DoContext* ctx, uint32_t n, uint32_t m, uint32_t Q, const uint32_t* alphas_in, uint32_t* output);

// bool CudaLagrangeInterpEval(DoContext* ctx, uint32_t* result, const uint32_t* x_in, const uint32_t* y_in, uint32_t k, uint32_t eval_point, uint32_t q);

bool CudaReedSolomonDecode(
    DoContext* ctx,
    uint32_t *code, uint64_t co, uint64_t cs,
    const bool* noisyQuery,
    uint32_t ecc_len, uint32_t ecc_k, uint32_t q,
    uint32_t* success,
    uint64_t steps
);

#ifdef __cplusplus
}
#endif

#endif /* __CUDARS_H_ */
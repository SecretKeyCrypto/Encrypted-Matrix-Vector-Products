#ifndef WRAPPER_H
#define WRAPPER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void GenerateSystematicRSMatrix_uint32(uint32_t n, uint32_t m, uint32_t Q, uint32_t m_0, const uint32_t* alphas_in, uint32_t* output);

uint32_t LagrangeInterpEval(const uint32_t* x_in, const uint32_t* y_in, uint32_t k, uint32_t eval_point, uint32_t q);

#ifdef __cplusplus
}
#endif

#endif
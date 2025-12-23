#ifndef _PLAINLPN_H_
#define _PLAINLPN_H_

#include <stdint.h>
#include <stdbool.h>
#include "../dataobjects/docontext.h"

#ifdef __cplusplus
extern "C" {
#endif

bool PlainLpnEncode(
    DoContext* ctx,
    const uint32_t* input,            // [M * L]
    const uint32_t* rlcMatrix,        // [K * L]
    const uint32_t* generatorMatrix,  // [ECCLength * M_1]
    uint32_t* encoded,                // [ECCLength * M / M_1 * N]
    uint32_t M, uint32_t L, uint32_t K,
    uint32_t M_1, uint32_t ECCLength,
    uint32_t P
);

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* _PLANILPN_H_ */
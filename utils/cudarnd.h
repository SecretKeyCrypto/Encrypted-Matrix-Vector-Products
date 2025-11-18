#ifndef _UTIL_CUDARND_H
#define _UTIL_CUDARND_H

#include <cstdint>
#include "../dataobjects/docontext.h"

#ifdef __cplusplus
extern "C" {
#endif

bool cuda_randomize_vector(DoContext* ctx, uint32_t* data, uint32_t M, uint32_t N, bool transpose, bool circulant);
bool cuda_randomize_vector_with_seed(DoContext* ctx, uint32_t* data, uint32_t M, uint32_t N, bool transpose, bool circulant, int64_t seed, int64_t offset);
bool cuda_randomize_vector_with_modulus(DoContext* ctx, uint32_t* data, uint32_t M, uint32_t N, bool transpose, bool circulant, uint32_t modulus);
bool cuda_randomize_vector_with_modulus_and_seed(DoContext* ctx, uint32_t* data, uint32_t M, uint32_t N, bool transpose, bool circulant, uint32_t modulus, int64_t seed, int64_t offset);

bool cuda_lpn_noise_vector(DoContext* ctx, uint32_t* r, uint64_t ro, uint64_t length, double epsi, uint32_t p, int64_t seed, int64_t offset);

bool cuda_random_permutation(DoContext* ctx, uint32_t* d_perm, uint32_t n, int64_t seed, int64_t offset);

#ifdef __cplusplus
}
#endif

#endif /* _UTIL_CUDARND_H */
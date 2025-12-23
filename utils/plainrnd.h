#ifndef _UTIL_PLAINRND_H
#define _UTIL_PLAINRND_H

#include <cstdint>
#include "../dataobjects/docontext.h"

#ifdef __cplusplus
extern "C" {
#endif

bool plain_randomize_vector(DoContext* ctx, uint32_t* data, uint32_t M, uint32_t N, bool transpose, bool circulant);
bool plain_randomize_vector_with_seed(DoContext* ctx, uint32_t* data, uint32_t M, uint32_t N, bool transpose, bool circulant, int64_t seed, int64_t offset);
bool plain_randomize_vector_with_modulus(DoContext* ctx, uint32_t* data, uint32_t M, uint32_t N, bool transpose, bool circulant, uint32_t modulus);
bool plain_randomize_vector_with_modulus_and_seed(DoContext* ctx, uint32_t* data, uint32_t M, uint32_t N, bool transpose, bool circulant, uint32_t modulus, int64_t seed, int64_t offset);

bool plain_lpn_noise_vector(DoContext* ctx, uint32_t* r, uint64_t ro, uint64_t length, double epsi, uint32_t modulus, int64_t seed, int64_t offset);

bool plain_random_permutation(DoContext* ctx, uint32_t* perm, uint32_t n, int64_t seed, int64_t offset);

#ifdef __cplusplus
}
#endif

#endif /* _UTIL_PLAINRND_H */

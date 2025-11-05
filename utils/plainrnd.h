#ifndef _UTIL_PLAINRND_H
#define _UTIL_PLAINRND_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void plain_randomize_vector(uint32_t* data, uint32_t M, uint32_t N, bool transpose);
void plain_randomize_vector_with_seed(uint32_t* data, uint32_t M, uint32_t N, bool transpose, int64_t seed);
void plain_randomize_vector_with_modulus(uint32_t* data, uint32_t M, uint32_t N, bool transpose, uint32_t modulus);
void plain_randomize_vector_with_modulus_and_seed(uint32_t* data, uint32_t M, uint32_t N, bool transpose, uint32_t modulus, int64_t seed);

void plain_lpn_noise_vector(uint32_t* data, uint64_t length, double epsi, uint32_t modulus);

#ifdef __cplusplus
}
#endif

#endif /* _UTIL_PLAINRND_H */

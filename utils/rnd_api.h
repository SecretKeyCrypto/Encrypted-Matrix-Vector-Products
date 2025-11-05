#ifndef _UTIL_RND_API_H
#define _UTIL_RND_API_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// NOTE: using a (non-power-of-2) modulus introduces a small bias

void randomize_vector(uint32_t* data, uint32_t M, uint32_t N, bool transpose);
void randomize_vector_with_seed(uint32_t* data, uint32_t M, uint32_t N, bool transpose, int64_t seed);
void randomize_vector_with_modulus(uint32_t* data, uint32_t M, uint32_t N, bool transpose, uint32_t modulus);
void randomize_vector_with_modulus_and_seed(uint32_t* data, uint32_t M, uint32_t N, bool transpose, uint32_t modulus, int64_t seed);

void lpn_noise_vector(uint32_t* data, uint64_t length, double epsi, uint32_t modulus);

#ifdef __cplusplus
}
#endif

#endif /* _UTIL_RND_API_H */

#ifndef _UTIL_CUDARND_H
#define _UTIL_CUDARND_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void cuda_randomize_vector(uint32_t* data, uint32_t M, uint32_t N, bool transpose);
void cuda_randomize_vector_with_seed(uint32_t* data, uint32_t M, uint32_t N, bool transpose, int64_t seed);
void cuda_randomize_vector_with_modulus(uint32_t* data, uint32_t M, uint32_t N, bool transpose, uint32_t modulus);
void cuda_randomize_vector_with_modulus_and_seed(uint32_t* data, uint32_t M, uint32_t N, bool transpose, uint32_t modulus, int64_t seed);

void cuda_lpn_noise_vector(uint32_t* r, uint64_t length, double epsi, uint32_t p);

#ifdef __cplusplus
}
#endif

#endif /* _UTIL_CUDARND_H */
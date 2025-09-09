#ifndef _UTIL_RND_API_H
#define _UTIL_RND_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void randomize_vector(uint32_t* data, uint32_t length);
void randomize_vector_with_seed(uint32_t* data, uint32_t length, int64_t seed);
void randomize_vector_with_modulus(uint32_t* data, uint32_t length, uint32_t modulus);
void randomize_vector_with_modulus_and_seed(uint32_t* data, uint32_t length, uint32_t modulus, int64_t seed);

#ifdef __cplusplus
}
#endif

#endif /* _UTIL_RND_API_H */

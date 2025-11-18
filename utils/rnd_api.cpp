#ifdef USE_FAST_CODE_WITH_CUDA

#include "../dataobjects/common.h"

#include "cudarnd.h"
#include "plainrnd.h"

#ifdef __cplusplus
extern "C" {
#endif

bool randomize_vector(DoContext* ctx, uint32_t* data, uint32_t M, uint32_t N, bool transpose, bool circulant) {
    return cuda_call(randomize_vector(ctx, data, M, N, transpose, circulant));
}

bool randomize_vector_with_seed(DoContext* ctx, uint32_t* data, uint32_t M, uint32_t N, bool transpose, bool circulant, int64_t seed, int64_t offset) {
    return cuda_call(randomize_vector_with_seed(ctx, data, M, N, transpose, circulant, seed, offset));
}

bool randomize_vector_with_modulus(DoContext* ctx, uint32_t* data, uint32_t M, uint32_t N, bool transpose, bool circulant, uint32_t modulus) {
    return cuda_call(randomize_vector_with_modulus(ctx, data, M, N, transpose, circulant, modulus));
}

bool randomize_vector_with_modulus_and_seed(DoContext* ctx, uint32_t* data, uint32_t M, uint32_t N, bool transpose, bool circulant, uint32_t modulus, int64_t seed, int64_t offset) {
    return cuda_call(randomize_vector_with_modulus_and_seed(ctx, data, M, N, transpose, circulant, modulus, seed, offset));
}

bool lpn_noise_vector(DoContext* ctx, uint32_t* r, uint64_t ro, uint64_t length, double epsi, uint32_t modulus, int64_t seed, int64_t offset) {
    return cuda_call(lpn_noise_vector(ctx, r, ro, length, epsi, modulus, seed, offset));
}

bool random_permutation(DoContext* ctx, uint32_t* perm, uint32_t n, int64_t seed, int64_t offset) {
    return cuda_call(random_permutation(ctx, perm, n, seed, offset));
}

#ifdef __cplusplus
} // extern "C"
#endif

#else /* USE_FAST_CODE_WITH_CUDA */

#include "plainrnd.h"

#ifdef __cplusplus
extern "C" {
#endif

bool randomize_vector(DoContext* ctx, uint32_t* data, uint32_t M, uint32_t N, bool transpose, bool circulant) {
    return plain_randomize_vector(ctx, data, M, N, transpose, circulant);
}

bool randomize_vector_with_seed(DoContext* ctx, uint32_t* data, uint32_t M, uint32_t N, bool transpose, bool circulant, int64_t seed, int64_t offset) {
    return plain_randomize_vector_with_seed(ctx, data, M, N, transpose, circulant, seed, offset);
}

bool randomize_vector_with_modulus(DoContext* ctx, uint32_t* data, uint32_t M, uint32_t N, bool transpose, bool circulant, uint32_t modulus) {
    return plain_randomize_vector_with_modulus(ctx, data, M, N, transpose, circulant, modulus);
}

bool randomize_vector_with_modulus_and_seed(DoContext* ctx, uint32_t* data, uint32_t M, uint32_t N, bool transpose, bool circulant, uint32_t modulus, int64_t seed, int64_t offset) {
    return plain_randomize_vector_with_modulus_and_seed(ctx, data, M, N, transpose, circulant, modulus, seed, offset);
}

bool lpn_noise_vector(DoContext* ctx, uint32_t* r, uint64_t ro, uint64_t length, double epsi, uint32_t modulus, int64_t seed, int64_t offset) {
    return plain_lpn_noise_vector(ctx, r, ro, length, epsi, modulus, seed, offset);
}

bool random_permutation(DoContext* ctx, uint32_t* perm, uint32_t n, int64_t seed, int64_t offset) {
    return plain_random_permutation(ctx, perm, n, seed, offset);
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* USE_FAST_CODE_WITH_CUDA */

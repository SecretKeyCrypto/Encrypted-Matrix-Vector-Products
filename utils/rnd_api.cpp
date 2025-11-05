#include "rnd_api.h"

#ifdef USE_FAST_CODE_WITH_CUDA

#include "cudarnd.h"

void randomize_vector(uint32_t* data, uint32_t M, uint32_t N, bool transpose) {
    cuda_randomize_vector(data, M, N, transpose);
}

void randomize_vector_with_seed(uint32_t* data, uint32_t M, uint32_t N, bool transpose, int64_t seed) {
    cuda_randomize_vector_with_seed(data, M, N, transpose, seed);
}

void randomize_vector_with_modulus(uint32_t* data, uint32_t M, uint32_t N, bool transpose, uint32_t modulus) {
    cuda_randomize_vector_with_modulus(data, M, N, transpose, modulus);
}

void randomize_vector_with_modulus_and_seed(uint32_t* data, uint32_t M, uint32_t N, bool transpose, uint32_t modulus, int64_t seed) {
    cuda_randomize_vector_with_modulus_and_seed(data, M, N, transpose, modulus, seed);
}

void lpn_noise_vector(uint32_t* data, uint64_t length, double epsi, uint32_t modulus) {
    cuda_lpn_noise_vector(data, length, epsi, modulus);
}

#else

#include "plainrnd.h"

void randomize_vector(uint32_t* data, uint32_t M, uint32_t N, bool transpose) {
    plain_randomize_vector(data, M, N, transpose);
}

void randomize_vector_with_seed(uint32_t* data, uint32_t M, uint32_t N, bool transpose, int64_t seed) {
    plain_randomize_vector_with_seed(data, M, N, transpose, seed);
}

void randomize_vector_with_modulus(uint32_t* data, uint32_t M, uint32_t N, bool transpose, uint32_t modulus) {
    plain_randomize_vector_with_modulus(data, M, N, transpose, modulus);
}

void randomize_vector_with_modulus_and_seed(uint32_t* data, uint32_t M, uint32_t N, bool transpose, uint32_t modulus, int64_t seed) {
    plain_randomize_vector_with_modulus_and_seed(data, M, N, transpose, modulus, seed);
}

void lpn_noise_vector(uint32_t* data, uint64_t length, double epsi, uint32_t modulus) {
    plain_lpn_noise_vector(data, length, epsi, modulus);
}

#endif

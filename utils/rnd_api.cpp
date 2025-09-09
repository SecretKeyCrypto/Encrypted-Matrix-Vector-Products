#include "rnd_api.h"
#include "aes_rnd.h"
#include "../dataobjects/mod_simd.h"

#include <string.h>

static thread_local AES_Random aesrnd;

void randomize_vector(uint32_t* data, uint32_t length) {
    if (!data) return;

    uint32_t bytelength = length * sizeof(uint32_t);
    uint8_t* bytedata = (uint8_t*)data;
    uint32_t i = 0;

    for (; i + 16 <= bytelength; i += 16, bytedata += 16) {
        aesrnd.random_bytes(bytedata);
    }

    if (i < bytelength) {
        uint8_t bytes[16];
        aesrnd.random_bytes(bytes);
        memcpy(bytedata, bytes, bytelength - i);
    }
}

void randomize_vector_with_seed(uint32_t* data, uint32_t length, int64_t seed) {
    uint8_t seed8_16[16];
    int64_t* seed2_64 = (int64_t*)seed8_16;
    seed2_64[0] = seed2_64[1] = seed;
    aesrnd.reseed(seed8_16);
    randomize_vector(data, length);
}

void randomize_vector_with_modulus(uint32_t* data, uint32_t length, uint32_t modulus) {
    randomize_vector(data, length);
    vector_mod_op(data, data, modulus, length);
}

void randomize_vector_with_modulus_and_seed(uint32_t* data, uint32_t length, uint32_t modulus, int64_t seed) {
    randomize_vector_with_seed(data, length, seed);
    vector_mod_op(data, data, modulus, length);
}

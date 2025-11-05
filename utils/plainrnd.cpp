#include "plainrnd.h"

#ifdef USE_FAST_CODE_WITH_CUDA

#include "fields.h"
#include "cudafields.h"

#ifdef __cplusplus
extern "C" {
#endif

int FieldLPNNoiseVector(
    uint32_t* r, uint64_t ro,
    uint64_t length, double epsi, uint32_t p
) {
    CudaLPNNoiseVector(r + ro, length, epsi, p);
    return 2;
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#else

#include "aes_rnd.h"
#include "../dataobjects/mod_simd.h"

#include <string.h>

static thread_local AES_Random aesrnd;

void plain_randomize_vector_1d(uint32_t* data, uint64_t length) {
    if (!data) return;

    uint64_t bytelength = length * sizeof(uint32_t);
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

void plain_randomize_vector(uint32_t* data, uint32_t M, uint32_t N, bool transpose) {
    if (!data) return;
    if (M == 1 || N == 1) {
        plain_randomize_vector_1d(data, (uint64_t)M*N);
        return;
    }

    const uint64_t total_elems = (uint64_t)M * (uint64_t)N;
    if (total_elems == 0) return;

    const uint64_t elemsPerChunk = 4;
    union Chunk {
        uint8_t bytes[elemsPerChunk*sizeof(uint32_t)];
        uint32_t vals[elemsPerChunk];
    } chunk;

    uint64_t start_linear = 0;
    uint32_t row = 0, col = 0;

    while (start_linear < total_elems) {
        aesrnd.random_bytes(chunk.bytes);

        int produced = 0;
        for (uint32_t r = row; r < M && produced < elemsPerChunk; ++r) {
            for (uint32_t c = (r == row ? col : 0); c < N && produced < elemsPerChunk; ++c) {
                uint64_t linear = start_linear + (uint64_t)produced;
                if (linear >= total_elems) break;

                uint32_t v = chunk.vals[produced];
                if (!transpose) {
                    data[linear] = v;
                } else {
                    uint64_t t_idx = (uint64_t)c * (uint64_t)M + (uint64_t)r;
                    data[t_idx] = v;
                }
                ++produced;
            }
        }

        start_linear += elemsPerChunk;
        col += elemsPerChunk;
        while (col >= N) {
            col -= N;
            ++row;
        }
    }
}

void plain_randomize_vector_with_seed(uint32_t* data, uint32_t M, uint32_t N, bool transpose, int64_t seed) {
    uint8_t seed8_16[16];
    int64_t* seed2_64 = (int64_t*)seed8_16;
    seed2_64[0] = seed2_64[1] = seed;
    aesrnd.reseed(seed8_16);
    plain_randomize_vector(data, M, N, transpose);
}

void plain_randomize_vector_with_modulus(uint32_t* data, uint32_t M, uint32_t N, bool transpose, uint32_t modulus) {
    plain_randomize_vector(data, M, N, transpose);
    vector_mod_op(data, data, modulus, (size_t)M * N);
}

void plain_randomize_vector_with_modulus_and_seed(uint32_t* data, uint32_t M, uint32_t N, bool transpose, uint32_t modulus, int64_t seed) {
    plain_randomize_vector_with_seed(data, M, N, transpose, seed);
    vector_mod_op(data, data, modulus, (size_t)M * N);
}

inline void NoSimdLPNNoiseVector(uint32_t* r, uint64_t length, double epsi, uint32_t p) {
    uint32_t pmask = bitmask_for(p);
    AES_Random rnd;
    rnd.reseed();
    const int bits = 53;
    const uint64_t bits_power = 1ULL << bits;
    uint64_t u64a[2];
    uint32_t *u32a = reinterpret_cast<uint32_t *>(u64a);

    for (uint64_t i = 0; i < length; i++) {
        rnd.random_bytes(reinterpret_cast<uint8_t*>(u64a));
        double f = (u64a[0] & (bits_power - 1)) / double(bits_power);
        if (f <= epsi) {
            int j = 2;
            uint32_t u;
            do {
                ++j;
                if (j == 4) {
                    rnd.random_bytes(reinterpret_cast<uint8_t*>(u64a));
                    j = 0;
                }
                u = u32a[j] & pmask;
            } while (u >= p - 1);
            r[i] = u + 1;
        }
    }
}

void plain_lpn_noise_vector(uint32_t* data, uint64_t length, double epsi, uint32_t modulus) {
    NoSimdLPNNoiseVector(data, length, epsi, modulus);
}

#endif /* USE_FAST_CODE_WITH_CUDA */
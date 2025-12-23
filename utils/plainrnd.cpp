#include "plainrnd.h"

#include "plainrnd.h"
#include "aes_rnd.h"
#include "../dataobjects/fields.h"
#include "../dataobjects/mod_simd.h"

#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

static thread_local AES_Random aesrnd;

bool plain_randomize_vector_1d(uint32_t* data, uint64_t length, bool circulant) {
    if (!data || length == 0) return true;

    if (!circulant) {
        // --- original fast path ---
        uint64_t bytelength = length * sizeof(uint32_t);
        uint8_t* bytedata = (uint8_t*)data;
        uint64_t i = 0;

        for (; i + 16 <= bytelength; i += 16, bytedata += 16) {
            aesrnd.random_bytes(bytedata);
        }

        if (i < bytelength) {
            uint8_t bytes[16];
            aesrnd.random_bytes(bytes);
            memcpy(bytedata, bytes, bytelength - i);
        }
        return true;
    }

    // --- circulant path ---
    const uint32_t elemsPerChunk = 4;
    union Chunk {
        uint8_t bytes[elemsPerChunk * sizeof(uint32_t)];
        uint32_t vals[elemsPerChunk];
    } chunk;

    uint64_t produced = 0;
    while (produced < length) {
        aesrnd.random_bytes(chunk.bytes);

        uint32_t count = (uint32_t)((length - produced) < elemsPerChunk ? (length - produced) : elemsPerChunk);

        for (uint32_t k = 0; k < count; ++k) {
            uint64_t c = produced + k;
            uint64_t idx = (c == 0 ? 0 : (length - c));
            data[idx] = chunk.vals[k];
        }

        produced += count;
    }
    return true;
}

bool plain_randomize_vector(DoContext* ctx, uint32_t* data, uint32_t M, uint32_t N, bool transpose, bool circulant) {
    if (!data) return true;
    if (M == 1 || N == 1) {
        bool circulant_1d = (M == 1) ? circulant : false;
        plain_randomize_vector_1d(data, (uint64_t)M * (uint64_t)N, circulant_1d);
        return true;
    }

    const uint64_t total_elems = (uint64_t)M * (uint64_t)N;
    if (total_elems == 0) return true;

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

                uint32_t cc = circulant ? (c == 0 ? 0 : N - c) : c;
                uint64_t idx = transpose ? (uint64_t)cc * M + r : (uint64_t)r * N + cc;
                if (idx >= total_elems) break;
                uint32_t v = chunk.vals[produced];
                data[idx] = v;
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
    return true;
}

bool plain_randomize_vector_with_seed(DoContext* ctx, uint32_t* data, uint32_t M, uint32_t N, bool transpose, bool circulant, int64_t seed, int64_t offset) {
    uint8_t seed8_16[16];
    int64_t* seed2_64 = (int64_t*)seed8_16;
    seed2_64[0] = seed2_64[1] = seed;
    aesrnd.reseed(seed8_16);
    if (offset >= 0) aesrnd.seek(offset);
    plain_randomize_vector(ctx, data, M, N, transpose, circulant);
    return true;
}

bool plain_randomize_vector_with_modulus(DoContext* ctx, uint32_t* data, uint32_t M, uint32_t N, bool transpose, bool circulant, uint32_t modulus) {
    plain_randomize_vector(ctx, data, M, N, transpose, circulant);
    vector_mod_op(data, data, modulus, (size_t)M * N);
    return true;
}

bool plain_randomize_vector_with_modulus_and_seed(DoContext* ctx, uint32_t* data, uint32_t M, uint32_t N, bool transpose, bool circulant, uint32_t modulus, int64_t seed, int64_t offset) {
    plain_randomize_vector_with_seed(ctx, data, M, N, transpose, circulant, seed, offset);
    vector_mod_op(data, data, modulus, (size_t)M * N);
    return true;
}

inline void NoSimdLPNNoiseVector(DoContext* ctx, uint32_t* r, uint64_t length, double epsi, uint32_t p, int64_t seed, int64_t offset) {
    uint32_t pmask = bitmask_for(p);
    AES_Random rnd;
    rnd.reseed(seed);
    rnd.seek(offset);
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

bool plain_lpn_noise_vector(DoContext* ctx, uint32_t* r, uint64_t ro, uint64_t length, double epsi, uint32_t modulus, int64_t seed, int64_t offset) {
    NoSimdLPNNoiseVector(ctx, r + ro, length, epsi, modulus, seed, offset);
    return true;
}

bool plain_random_permutation(DoContext* ctx, uint32_t* perm, uint32_t n, int64_t seed, int64_t offset) {
    FieldRangeVector(ctx, perm, 0, 0, n);

    // Seed and shuffle
    uint8_t seed8_16[16];
    int64_t* seed2_64 = (int64_t*)seed8_16;
    seed2_64[0] = seed2_64[1] = seed;
    aesrnd.reseed(seed8_16);
    if (offset >= 0) aesrnd.seek(offset);

    uint8_t bytes[16 + 4];
    int width = n >= (1 << 16) ? 4 : n >= (1 << 8) ? 2 : 1;
    int count = n >= (1 << 16) ? 4 : n >= (1 << 8) ? 8 : 16;
    uint32_t mask = (1 << width) - 1;
    uint32_t b = 0;
    for (uint32_t i = n - 1; i > 0; --i) {
        if (b == 0) {
            aesrnd.random_bytes(bytes);
        }
        uint32_t j = (*(reinterpret_cast<uint32_t*>(bytes + b)) & mask) % (i + 1);
        uint32_t temp = perm[i];
        perm[i] = perm[j];
        perm[j] = temp;
        b += width;
        if (b == count) {
            b = 0;
        }
    }
    return true;
}

#ifdef __cplusplus
}
#endif

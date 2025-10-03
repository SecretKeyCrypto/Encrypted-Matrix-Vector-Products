#ifndef _MOD_SIMD_H
#define _MOD_SIMD_H

#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>

#define _MOD_SIMD_ALIGNED_LOADSTORE

const uint32_t _not_prime_power = ~0U;

inline uint32_t _mersenne_prime_power(uint32_t modulus) {
    if (modulus && ((modulus & (modulus + 1)) != 0)) return _not_prime_power;
    uint32_t power = 0;
    if (0xFFFF0000U & modulus) power += 16;
    if (0xFF00FF00U & modulus) power += 8;
    if (0xF0F0F0F0U & modulus) power += 4;
    if (0xCCCCCCCCU & modulus) power += 2;
    if (0xAAAAAAAAU & modulus) power += 1;
    switch (power) {
        case 2:
        case 3:
        case 5:
        case 7:
        case 13:
        case 17:
        case 19:
        case 31:
            return power;
    }
    return _not_prime_power;
}

inline uint32_t _fermat_prime_power(uint32_t modulus) {
    switch (modulus) {
        case 3: return 1;
        case 5: return 2;
        case 17: return 4;
        case 257: return 8;
        case 65537: return 16;
        default: return _not_prime_power;
    }
}

inline bool _nosimd_mod_op(uint32_t* in, uint32_t* out, uint32_t modulus, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        out[i] = in[i] % modulus;
    }
    return true;
}

// works for a fermat prime modulus of the form 2^k+1
//
// For modulus m of k+1 bits, number n of 2k+2 bits with T as high k+1 bits and U as low k+1 bits:
// m = 2^k + 1
// n = 2^(k+1) * T + U
// n = (2^k + 1 + 2^k + 1) * T - 2 * T + U
// n = U - 2 * T (mod m)

// For a Fermat modulus 2^k + 1, the maximum value 2^k leads to a maximum multiplication 2^(2k)
// which has only 2k+1 bits, not 2k+2. We could take the lower k bits for U and the higher k+1 bits
// for T. Then, it would be enough to subtract once and check r<0 once but no need to check r>=m
// since U<m:
//
// m = 2^k + 1
// n = 2^k * T + U
// n = (2^k + 1) * T - T + U
// n = U - T (mod m)
#define _MOD_SIMD_FASTER

#ifdef __SSE2__
class _sse2_fermat_prime {
    uint32_t k;
    uint32_t modulus;
    uint32_t shift;
    uint32_t bitmask;

    __m128i mask;
    __m128i two;
    __m128i mod;
    __m128i mod_minus_1;
    __m128i zero;
public:
    inline _sse2_fermat_prime(uint32_t modulus) : modulus(modulus) {
        k = _fermat_prime_power(modulus);
    }
    inline bool valid() const {
        return ~k != 0;
    }
    inline uint32_t get_shift() const {
        return shift;
    }
    inline __m128i get_mask() const {
        return mask;
    }
    inline bool init() {
        if (!valid()) return false;

#ifdef _MOD_SIMD_FASTER
        shift = k;
#else
        shift = k + 1;
#endif
        bitmask = (1U << shift) - 1;

        mask = _mm_set1_epi32(bitmask);
        two = _mm_set1_epi32(2);
        mod = _mm_set1_epi32(modulus);
        mod_minus_1 = _mm_set1_epi32(modulus - 1);
        zero = _mm_setzero_si128();

        return true;
    }
    inline __m128i compute(__m128i T, __m128i U) const {
        __m128i r = U;

        // Step 1: r -= T
        r = _mm_sub_epi32(r, T);
        __m128i is_neg1 = _mm_cmplt_epi32(r, zero);
        __m128i corr1 = _mm_and_si128(is_neg1, mod);
        r = _mm_add_epi32(r, corr1);

#ifndef _MOD_SIMD_FASTER
        // Step 2: r -= T again
        r = _mm_sub_epi32(r, T);
        __m128i is_neg2 = _mm_cmplt_epi32(r, zero);
        __m128i corr2 = _mm_and_si128(is_neg2, mod);
        r = _mm_add_epi32(r, corr2);

        // Step 3: if r >= m, subtract m
        __m128i is_over = _mm_cmpgt_epi32(r, mod_minus_1); // r >= m
        __m128i corr3 = _mm_and_si128(is_over, mod);
        r = _mm_sub_epi32(r, corr3);
#endif

        return r;
    }
    inline uint32_t compute(uint32_t T, uint32_t U) const {
        int32_t r = (int32_t)U - (int32_t)T;
        if (r < 0) r += modulus;
#ifndef _MOD_SIMD_FASTER
        r -= T;
        if (r < 0) r += modulus;
        if ((uint32_t)r >= modulus) r -= modulus;
#endif
        return (uint32_t)r;
    }
    inline void apply(__m128i* out, __m128i* in) const {
#ifdef _MOD_SIMD_ALIGNED_LOADSTORE
        __m128i x = _mm_load_si128(in);
#else
        __m128i x = _mm_loadu_si128(in);
#endif

        __m128i T = _mm_srli_epi32(x, shift);
        __m128i U = _mm_and_si128(x, mask);

        __m128i r = compute(T, U);
#ifdef _MOD_SIMD_ALIGNED_LOADSTORE
        _mm_store_si128(out, r);
#else
        _mm_storeu_si128(out, r);
#endif
    }
    inline void apply(uint32_t* out, uint32_t* in) const {
        uint32_t T = *in >> shift;
        uint32_t U = *in & bitmask;
        uint32_t r = compute(T, U);
        *out = r;
    }
};

inline bool _sse2_fermat_prime_mod_op(uint32_t* in, uint32_t* out, uint32_t modulus, size_t length) {
    _sse2_fermat_prime op(modulus);
    if (!op.init()) return false;

    size_t i = 0;
    for (; i + 4 <= length; i += 4) {
        op.apply((__m128i*)&out[i], (__m128i*)&in[i]);
    }
    for (; i < length; ++i) {
        op.apply(out + i, in + i);
    }
    return true;
}
#endif

#ifdef __AVX2__
class _avx2_fermat_prime {
    uint32_t k;
    uint32_t modulus;
    uint32_t shift;
    uint32_t bitmask;

    __m256i mask;
    __m256i mod;
    __m256i mod_minus_1;
    __m256i zero;
public:
    _avx2_fermat_prime(uint32_t modulus) : modulus(modulus) {
        k = _fermat_prime_power(modulus);
    }
    inline bool valid() const {
        return ~k != 0;
    }
    inline uint32_t get_shift() const {
        return shift;
    }
    inline __m256i get_mask() const {
        return mask;
    }
    inline bool init() {
        if (!valid()) return false;

#ifdef _MOD_SIMD_FASTER
        shift = k;
#else
        shift = k + 1;
#endif
        bitmask = (1U << shift) - 1;

        mask = _mm256_set1_epi32(bitmask);
        mod = _mm256_set1_epi32(modulus);
        mod_minus_1 = _mm256_set1_epi32(modulus - 1);
        zero = _mm256_setzero_si256();

        return true;
    }
    inline __m256i compute(__m256i T, __m256i U) const {
        __m256i r = U;

        // r -= T
        r = _mm256_sub_epi32(r, T);
        __m256i is_neg1 = _mm256_cmpgt_epi32(zero, r);
        __m256i corr1 = _mm256_and_si256(is_neg1, mod);
        r = _mm256_add_epi32(r, corr1);

#ifndef _MOD_SIMD_FASTER
        // r -= T again
        r = _mm256_sub_epi32(r, T);
        __m256i is_neg2 = _mm256_cmpgt_epi32(zero, r);
        __m256i corr2 = _mm256_and_si256(is_neg2, mod);
        r = _mm256_add_epi32(r, corr2);

        // if r >= m, subtract m
        __m256i is_over = _mm256_cmpgt_epi32(r, mod_minus_1); // r >= m
        __m256i corr3 = _mm256_and_si256(is_over, mod);
        r = _mm256_sub_epi32(r, corr3);
#endif

        return r;
    }
    inline uint32_t compute(uint32_t T, uint32_t U) const {
        int32_t r = (int32_t)U - (int32_t)T;
        if (r < 0) r += modulus;
#ifndef _MOD_SIMD_FASTER
        r -= T;
        if (r < 0) r += modulus;
        if ((uint32_t)r >= modulus) r -= modulus;
#endif
        return (uint32_t)r;
    }
    inline void apply(__m256i* out, __m256i* in) const {
#ifdef _MOD_SIMD_ALIGNED_LOADSTORE
        __m256i x = _mm256_load_si256(in);
#else
        __m256i x = _mm256_loadu_si256(in);
#endif

        __m256i T = _mm256_srli_epi32(x, shift);
        __m256i U = _mm256_and_si256(x, mask);

        __m256i r = compute(T, U);
#ifdef _MOD_SIMD_ALIGNED_LOADSTORE
        _mm256_store_si256(out, r);
#else
        _mm256_storeu_si256(out, r);
#endif
    }
    inline void apply(uint32_t* out, uint32_t* in) const {
        uint32_t T = *in >> shift;
        uint32_t U = *in & bitmask;
        uint32_t r = compute(T, U);
        *out = (uint32_t)r;
    }
};

inline bool _avx2_fermat_prime_mod_op(uint32_t* in, uint32_t* out, uint32_t modulus, size_t length) {
    _avx2_fermat_prime op(modulus);
    if (!op.init()) return false;

    size_t i = 0;
    for (; i + 8 <= length; i += 8) {
        op.apply((__m256i*)&out[i], (__m256i*)&in[i]);
    }
    for (; i < length; ++i) {
        op.apply(out + i, in + i);
    }
    return true;
}
#endif

inline bool vector_mod_op(uint32_t* in, uint32_t* out, uint32_t modulus, size_t length) {
    if (~_fermat_prime_power(modulus) != 0) {
#if defined(__AVX2__)
        return _avx2_fermat_prime_mod_op(in, out, modulus, length);
#elif defined(__SSE2__)
        return _sse2_fermat_prime_mod_op(in, out, modulus, length);
#endif
    }
    return _nosimd_mod_op(in, out, modulus, length);
}

#endif /* _MOD_SIMD_H */
#include "fields.h"
#include "mod_simd.h"
#include <iostream>
#include <emmintrin.h>
#include <immintrin.h>
#include <stdint.h>

inline void NoSimdAddVectors(uint32_t* r, const uint32_t* a, const uint32_t* b, uint64_t length, uint32_t p) {
    for (uint64_t i = 0; i < length; ++i) {
        r[i] = uint32_t((uint64_t(a[i]) + uint64_t(b[i])) % uint64_t(p));
    }
}

inline void NoSimdMulVector(uint32_t* r, const uint32_t* a, uint32_t b, uint64_t length, uint32_t p) {
    for (uint64_t i = 0; i < length; ++i) {
        r[i] = uint32_t((uint64_t(a[i]) * uint64_t(b)) % uint64_t(p));
    }
}

inline void NoSimdMulVectors(uint32_t* r, const uint32_t* a, const uint32_t* b, uint64_t length, uint32_t p) {
    for (uint64_t i = 0; i < length; ++i) {
        r[i] = uint32_t((uint64_t(a[i]) * uint64_t(b[i])) % uint64_t(p));
    }
}

inline void NoSimdSubVectors(uint32_t* r, const uint32_t* a, const uint32_t* b, uint64_t length, uint32_t p) {
    for (uint64_t i = 0; i < length; ++i) {
        r[i] = uint32_t((uint64_t(a[i]) + uint64_t(p) - uint64_t(b[i])) % uint64_t(p));
    }
}

inline void NoSimdNegVector(uint32_t* r, uint64_t length, uint32_t p) {
    for (uint64_t i = 0; i < length; ++i) {
        r[i] = uint32_t((uint64_t(p) - uint64_t(r[i])) % uint64_t(p));
    }
}

#ifdef __SSE2__
inline void SSE2AddVectors(uint32_t* r, const uint32_t* a, const uint32_t* b, uint64_t length, uint32_t p) {
    uint64_t i = 0;

    // Process 4 elements at a time
    for (; i + 4 <= length; i += 4) {
        __m128i va = _mm_loadu_si128((__m128i*)(a + i));
        __m128i vb = _mm_loadu_si128((__m128i*)(b + i));
        __m128i vr = _mm_add_epi32(va, vb);
        _mm_storeu_si128((__m128i*)(r + i), vr);
    }

    // Handle remaining elements
    for (; i < length; ++i) {
        r[i] = a[i] + b[i];
    }

    vector_mod_op(r, r, p, length);
}

inline void SSE2MulVector(uint32_t* r, const uint32_t* a, uint32_t b, uint64_t length, uint32_t p) {
    uint64_t i = 0;

    // std::cerr << "=====================================SSE2 " << p << std::endl;
    // Process 4 elements at a time
    _sse2_fermat_prime op(p);
    if (!op.valid()) {
        // std::cerr << "========================================= " << p << std::endl;
        return;
    }
    __m128i mask = op.get_mask();
    uint32_t shift = op.get_shift();
    __m128i vb = _mm_set1_epi32(b);
    for (; i + 4 <= length; i += 4) {
        __m128i va0 = _mm_loadu_si128((__m128i*)(a + i));
        __m128i va1 = _mm_srli_si128(va0, 4);
        __m128i vr0 = _mm_mul_epu32(va0, vb);
        __m128i vr1 = _mm_mul_epu32(va1, vb);
        __m128i U = _mm_and_si128(vr0, mask);
        __m128i T0 = _mm_srli_epi32(vr0, shift);
        __m128i T1 = _mm_slli_epi32(vr1, 32U - shift);
        __m128i T = _mm_or_si128(T0, T1);
        __m128i vr = op.compute(T, U);
        _mm_storeu_si128((__m128i*)(r + i), vr);
    }

    // Handle remaining elements
    for (; i < length; ++i) {
        r[i] = uint32_t((uint64_t(a[i]) * uint64_t(b)) % uint64_t(p));
    }
}

inline void SSE2MulVectors(uint32_t* r, const uint32_t* a, const uint32_t* b, uint64_t length, uint32_t p) {
    uint64_t i = 0;

    // std::cerr << "=====================================SSE2 " << p << std::endl;
    // Process 4 elements at a time
    _sse2_fermat_prime op(p);
    if (!op.valid()) {
        // std::cerr << "========================================= " << p << std::endl;
        return;
    }
    __m128i mask = op.get_mask();
    uint32_t shift = op.get_shift();
    for (; i + 4 <= length; i += 4) {
        __m128i va0 = _mm_loadu_si128((__m128i*)(a + i));
        __m128i va1 = _mm_srli_si128(va0, 4);
        __m128i vb = _mm_loadu_si128((__m128i*)(b + i));
        __m128i vr0 = _mm_mul_epu32(va0, vb);
        __m128i vr1 = _mm_mul_epu32(va1, vb);
        __m128i U = _mm_and_si128(vr0, mask);
        __m128i T0 = _mm_srli_epi32(vr0, shift);
        __m128i T1 = _mm_slli_epi32(vr1, 32U - shift);
        __m128i T = _mm_or_si128(T0, T1);
        __m128i vr = op.compute(T, U);
        _mm_storeu_si128((__m128i*)(r + i), vr);
    }

    // Handle remaining elements
    for (; i < length; ++i) {
        r[i] = uint32_t((uint64_t(a[i]) * uint64_t(b[i])) % uint64_t(p));
    }
}

inline void SSE2SubVectors(uint32_t* r, const uint32_t* a, const uint32_t* b, uint64_t length, uint32_t p) {
    uint64_t i = 0;

    // Subtract 4 elements at a time
    __m128i vp = _mm_set1_epi32(p);
    for (; i + 4 <= length; i += 4) {
        __m128i va = _mm_loadu_si128((__m128i*)(a + i));
        __m128i vb = _mm_loadu_si128((__m128i*)(b + i));
        __m128i vt = _mm_add_epi32(va, vp);
        __m128i vr = _mm_sub_epi32(vt, vb);
        _mm_storeu_si128((__m128i*)(r + i), vr);
    }

    // Handle remaining scalars
    for (; i < length; ++i) {
        r[i] = a[i] + p - b[i];
    }

    vector_mod_op(r, r, p, length);
}

inline void SSE2NegVector(uint32_t* r, uint64_t length, uint32_t p) {
    uint64_t i = 0;

    // Subtract 8 elements at a time
    __m128i vp = _mm_set1_epi32(p);
    for (; i + 8 <= length; i += 8) {
        __m128i vr = _mm_loadu_si128((__m128i*)(r + i));
        __m128i vt = _mm_sub_epi32(vp, vr);
        _mm_storeu_si128((__m128i*)(r + i), vt);
    }

    // Handle remaining scalars
    for (; i < length; ++i) {
        r[i] = p - r[i];
    }

    vector_mod_op(r, r, p, length);
}
#endif

#ifdef __AVX2__
inline void AVX2AddVectors(uint32_t* r, const uint32_t* a, const uint32_t* b, uint64_t length, uint32_t p) {
    uint64_t i = 0;

    // Process 8 elements at a time
    for (; i + 8 <= length; i += 8) {
        __m256i va = _mm256_loadu_si256((__m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((__m256i*)(b + i));
        __m256i vr = _mm256_add_epi32(va, vb);
        _mm256_storeu_si256((__m256i*)(r + i), vr);
    }

    // Handle remaining elements
    for (; i < length; ++i) {
        r[i] = a[i] + b[i];
    }

    vector_mod_op(r, r, p, length);
}

inline void AVX2MulVector(uint32_t* r, const uint32_t* a, uint32_t b, uint64_t length, uint32_t p) {
    if (!r || !a || !b) {
        return;
    }

    uint64_t i = 0;

    // std::cerr << "=====================================AVX2 " << p << std::endl;
    // Process 8 elements at a time
    _avx2_fermat_prime op(p);
    if (!op.init()) {
        // std::cerr << "========================================= " << p << std::endl;
        return;
    }
    __m256i mask = op.get_mask();
    uint32_t shift = op.get_shift();
    __m256i vb = _mm256_set1_epi32(b);
    __m256i mask32 = _mm256_set1_epi64x((1ULL << 32) - 1);
    for (; i + 8 <= length; i += 8) {
        __m256i va0 = _mm256_loadu_si256((__m256i*)(a + i));
        __m256i va1 = _mm256_srli_si256(va0, 4);
        __m256i vx0 = _mm256_mul_epu32(va0, vb);
        __m256i vx1 = _mm256_mul_epu32(va1, vb);
        // __m256i vr0a = _mm256_and_si256(vx0, mask32);
        // __m256i vr0b = _mm256_and_si256(vx1, mask32);
        // __m256i vr0c = _mm256_slli_epi64(_mm256_and_si256(vx1, mask32), 32);
        __m256i vr0 = _mm256_or_si256(_mm256_and_si256(vx0, mask32), _mm256_slli_epi64(_mm256_and_si256(vx1, mask32), 32));
        __m256i vy0 = _mm256_srli_si256(vx0, 4);
        __m256i vy1 = _mm256_srli_si256(vx1, 4);
        __m256i vr1 = _mm256_or_si256(_mm256_and_si256(vy0, mask32), _mm256_slli_epi64(_mm256_and_si256(vy1, mask32), 32));
        __m256i U = _mm256_and_si256(vr0, mask);
        __m256i T0 = _mm256_srli_epi32(vr0, shift);
        __m256i T1 = _mm256_slli_epi32(vr1, 32U - shift);
        __m256i T = _mm256_or_si256(T0, T1);
        __m256i vr = op.compute(T, U);
        // if (i == 0) {
        //     std::cerr << std::hex << "va0=" << *(uint32_t*)(&va0) << " va1=" << *(uint32_t*)(&va1)
        //         << " vx0=" << *(uint32_t*)(&vx0) << " vx1=" << *(uint32_t*)(&vx1)
        //         << " vr0a=" << *(uint32_t*)(&vr0a) << " vr0b=" << *(uint32_t*)(&vr0b) << " vr0c=" << *(uint32_t*)(&vr0c)
        //         << " vr0=" << *(uint32_t*)(&vr0) << " vr1=" << *(uint32_t*)(&vr1)
        //         << " T0=" << *(uint32_t*)(&T0) << " T1=" << *(uint32_t*)(&T1)
        //         << " U=" << *(uint32_t*)(&U) << " T=" << *(uint32_t*)(&T)
        //         << " mask=" << *(uint32_t*)(&mask) << " shift=" << shift << " mask32=" << *(uint32_t*)(&mask32)
        //         << std::endl;
        // }
        _mm256_storeu_si256((__m256i*)(r + i), vr);
    }

    // Handle remaining elements
    for (; i < length; ++i) {
        r[i] = uint32_t((uint64_t(a[i]) * uint64_t(b)) % uint64_t(p));
    }
}

inline void AVX2MulVectors(uint32_t* r, const uint32_t* a, const uint32_t* b, uint64_t length, uint32_t p) {
    if (!r || !a || !b) {
        return;
    }

    uint64_t i = 0;

    // std::cerr << "=====================================AVX2 " << p << std::endl;
    // Process 8 elements at a time
    _avx2_fermat_prime op(p);
    if (!op.init()) {
        // std::cerr << "========================================= " << p << std::endl;
        return;
    }
    __m256i mask = op.get_mask();
    uint32_t shift = op.get_shift();
    __m256i mask32 = _mm256_set1_epi64x((1ULL << 32) - 1);
    for (; i + 8 <= length; i += 8) {
        __m256i va0 = _mm256_loadu_si256((__m256i*)(a + i));
        __m256i va1 = _mm256_srli_si256(va0, 4);
        __m256i vb0 = _mm256_loadu_si256((__m256i*)(b + i));
        __m256i vb1 = _mm256_srli_si256(vb0, 4);
        __m256i vx0 = _mm256_mul_epu32(va0, vb0);
        __m256i vx1 = _mm256_mul_epu32(va1, vb1);
        // __m256i vr0a = _mm256_and_si256(vx0, mask32);
        // __m256i vr0b = _mm256_and_si256(vx1, mask32);
        // __m256i vr0c = _mm256_slli_epi64(_mm256_and_si256(vx1, mask32), 32);
        __m256i vr0 = _mm256_or_si256(_mm256_and_si256(vx0, mask32), _mm256_slli_epi64(_mm256_and_si256(vx1, mask32), 32));
        __m256i vy0 = _mm256_srli_si256(vx0, 4);
        __m256i vy1 = _mm256_srli_si256(vx1, 4);
        __m256i vr1 = _mm256_or_si256(_mm256_and_si256(vy0, mask32), _mm256_slli_epi64(_mm256_and_si256(vy1, mask32), 32));
        __m256i U = _mm256_and_si256(vr0, mask);
        __m256i T0 = _mm256_srli_epi32(vr0, shift);
        __m256i T1 = _mm256_slli_epi32(vr1, 32U - shift);
        __m256i T = _mm256_or_si256(T0, T1);
        __m256i vr = op.compute(T, U);
        // if (i == 0) {
        //     std::cerr << std::hex << "a=" << *a << " b=" << *b
        //         << " va0=" << *(uint32_t*)(&va0) << " va1=" << *(uint32_t*)(&va1)
        //         << " vx0=" << *(uint32_t*)(&vx0) << " vx1=" << *(uint32_t*)(&vx1)
        //         << " vr0a=" << *(uint32_t*)(&vr0a) << " vr0b=" << *(uint32_t*)(&vr0b) << " vr0c=" << *(uint32_t*)(&vr0c)
        //         << " vr0=" << *(uint32_t*)(&vr0) << " vr1=" << *(uint32_t*)(&vr1)
        //         << " T0=" << *(uint32_t*)(&T0) << " T1=" << *(uint32_t*)(&T1)
        //         << " U=" << *(uint32_t*)(&U) << " T=" << *(uint32_t*)(&T) << " vr=" << *(uint32_t*)(&vr)
        //         << " mask=" << *(uint32_t*)(&mask) << " shift=" << shift << " mask32=" << *(uint32_t*)(&mask32)
        //         << std::endl;
        // }
        _mm256_storeu_si256((__m256i*)(r + i), vr);
    }

    // Handle remaining elements
    for (; i < length; ++i) {
        r[i] = uint32_t((uint64_t(a[i]) * uint64_t(b[i])) % uint64_t(p));
    }
}

inline void AVX2SubVectors(uint32_t* r, const uint32_t* a, const uint32_t* b, uint64_t length, uint32_t p) {
    uint64_t i = 0;

    // Subtract 8 elements at a time
    __m256i vp = _mm256_set1_epi32(p);
    for (; i + 8 <= length; i += 8) {
        __m256i va = _mm256_loadu_si256((__m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((__m256i*)(b + i));
        __m256i vt = _mm256_add_epi32(va, vp);
        __m256i vr = _mm256_sub_epi32(vt, vb);
        _mm256_storeu_si256((__m256i*)(r + i), vr);
    }

    // Handle remaining scalars
    for (; i < length; ++i) {
        r[i] = a[i] + p - b[i];
    }

    vector_mod_op(r, r, p, length);
}

inline void AVX2NegVector(uint32_t* r, uint64_t length, uint32_t p) {
    uint64_t i = 0;

    // Subtract 8 elements at a time
    __m256i vp = _mm256_set1_epi32(p);
    for (; i + 8 <= length; i += 8) {
        __m256i vr = _mm256_loadu_si256((__m256i*)(r + i));
        __m256i vt = _mm256_sub_epi32(vp, vr);
        _mm256_storeu_si256((__m256i*)(r + i), vt);
    }

    // Handle remaining scalars
    for (; i < length; ++i) {
        r[i] = p - r[i];
    }

    vector_mod_op(r, r, p, length);
}
#endif

extern "C" {

void FieldAddVectors(
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    const uint32_t* b, uint64_t bo,
    uint64_t length, uint32_t p
) {
#if defined(__AVX2__)
    AVX2AddVectors(r + ro, a + ao, b + bo, length, p);
#elif defined(__SSE2__)
    SSE2AddVectors(r + ro, a + ao, b + bo, length, p);
#else
    NoSimdAddVectors(r + ro, a + ao, b + bo, length, p);
#endif
}

void FieldMulVector(
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    uint32_t b,
    uint64_t length, uint32_t p
) {
#if defined(__AVX2__)
    AVX2MulVector(r + ro, a + ao, b, length, p);
#elif defined(__SSE2__)
    SSE2MulVector(r + ro, a + ao, b, length, p);
#else
    NoSimdMulVector(r + ro, a + ao, b, length, p);
#endif
}

void FieldMulVectors(
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    const uint32_t* b, uint64_t bo,
    uint64_t length, uint32_t p
) {
#if defined(__AVX2__)
    AVX2MulVectors(r + ro, a + ao, b + bo, length, p);
#elif defined(__SSE2__)
    SSE2MulVectors(r + ro, a + ao, b + bo, length, p);
#else
    NoSimdMulVectors(r + ro, a + ao, b + bo, length, p);
#endif
}

void FieldSubVectors(
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    const uint32_t* b, uint64_t bo,
    uint64_t length, uint32_t p
) {
#if defined(__AVX2__)
    AVX2SubVectors(r + ro, a + ao, b + bo, length, p);
#elif defined(__SSE2__)
    SSE2SubVectors(r + ro, a + ao, b + bo, length, p);
#else
    NoSimdSubVectors(r + ro, a + ao, b + bo, length, p);
#endif
}

void FieldNegVector(
    uint32_t* r, uint64_t ro,
    uint64_t length, uint32_t p
) {
#if defined(__AVX2__)
    AVX2NegVector(r + ro, length, p);
#elif defined(__SSE2__)
    SSE2NegVector(r + ro, length, p);
#else
    NoSimdNegVector(r + ro, a + ao, b + bo, length, p);
#endif
}

}
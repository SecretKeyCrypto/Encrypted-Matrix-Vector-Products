#ifdef USE_FAST_CODE_WITH_CUDA

#include "fields.h"
#include "cudafields.h"

#ifdef __cplusplus
extern "C" {
#endif

int FieldRangeVector(
    uint32_t* r, uint64_t ro,
    uint32_t start, uint64_t length
) {
    CudaRangeVector(r + ro, start, length);
    return 2;
}

int FieldCopyVector(
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    uint64_t length
) {
    CudaCopyVector(r + ro, a + ao, length);
    return 2;
}

int FieldSetVector(
    uint32_t* r, uint64_t ro,
    uint64_t length, uint32_t v
) {
    CudaSetVector(r + ro, length, v);
    return 2;
}

int FieldAddToVector(
    uint32_t* r, uint64_t ro,
    uint32_t v,
    uint64_t length
) {
    CudaAddToVector(r + ro, v, length);
    return 2;
}

int FieldAddVectors(
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    const uint32_t* b, uint64_t bo,
    uint64_t length, uint32_t p
) {
    CudaAddVectors(r + ro, a + ao, b + bo, length, p);
    return 2;
}

int FieldMulVector(
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    uint32_t b,
    uint64_t length, uint32_t p
) {
    CudaMulVector(r + ro, a + ao, b, length, p);
    return 2;
}

int FieldMulVectors(
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    const uint32_t* b, uint64_t bo,
    uint64_t length, uint32_t p
) {
    CudaMulVectors(r + ro, a + ao, b + bo, length, p);
    return 2;
}

int FieldSubVectors(
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    const uint32_t* b, uint64_t bo,
    uint64_t length, uint32_t p
) {
    CudaSubVectors(r + ro, a + ao , b + bo, length, p);
    return 2;
}

int FieldNegVector(
    uint32_t* r, uint64_t ro,
    uint64_t length, uint32_t p
) {
    CudaNegVector(r + ro, length, p);
    return 2;
}

int FieldAddVectorIfNonZero(
    bool* t, uint64_t t_index,
    uint32_t* r, uint64_t ro,
    const uint32_t* e, uint64_t eo,
    uint64_t length, uint32_t p
) {
    CudaAddVectorIfNonZero(t + t_index, r + ro, e + eo, length, p);
    return 2;
}

#ifdef __cplusplus
} /* extern "C" */
#endif


#else /* USE_FAST_CODE_WITH_CUDA */

#include <memory.h>
#include "common.h"
#include "fields.h"
#include "mod_simd.h"
#include "../utils/aes_rnd.h"

inline void NoSimdRangeVector(uint32_t* r, uint32_t start, uint64_t length) {
    for (uint64_t i = 0; i < length; i++) {
        r[i] = start++;
    }
}

inline void NoSimdSetVector(uint32_t* r, uint64_t length, uint32_t v) {
    for (uint64_t i = 0; i < length; ++i) {
        r[i] = v;
    }
}

inline void NoSimdAddToVector(uint32_t* r, uint32_t v, uint64_t length) {
    for (uint64_t i = 0; i < length; ++i) {
        r[i] += v;
    }
}

inline void NoSimdAddVectors(uint32_t* r, const uint32_t* a, const uint32_t* b, uint64_t length, uint32_t p) {
    for (uint64_t i = 0; i < length; ++i) {
        uint64_t t = uint64_t(a[i]) + uint64_t(b[i]);
        r[i] = uint32_t(t >= p ? t - p : p);
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
        
        r[i] = (a[i] >= b[i]) ? (a[i] - b[i]) : (p - (b[i] - a[i]));
    }
}

inline void NoSimdNegVector(uint32_t* r, uint64_t length, uint32_t p) {
    for (uint64_t i = 0; i < length; ++i) {
        r[i] = (r[i] == 0) ? 0 : (p - r[i]);
    }
}

void NoSimdIsZeroVector(bool *t, const uint32_t* e, uint64_t e_length) {
    for (uint64_t i = 0; i < e_length; i++) {
        if (e[i] != 0) {
            *t = false;
            return;
        }
    }
    *t = true;
}

#ifdef __SSE2__
inline void SSE2RangeVector(uint32_t* r, uint32_t start, uint64_t length) {
    __m128i vec = _mm_set_epi32(start + 3, start + 2, start + 1, start);
    __m128i inc = _mm_set1_epi32(4);

    uint64_t i = 0;
    for (; i + 4 <= length; i += 4) {
        _mm_storeu_si128((__m128i*)(r + i), vec);
        vec = _mm_add_epi32(vec, inc);
    }

    start += i;
    for (; i < length; i++) {
        r[i] = start++;
    }
}

inline void SSE2SetVector(uint32_t* r, uint64_t length, uint32_t v) {
    uint64_t i = 0;

    // Process 4 elements at a time
    for (; i + 4 <= length; i += 4) {
        __m128i vr = _mm_set1_epi32(v);
        _mm_storeu_si128((__m128i*)(r + i), vr);
    }

    // Handle remaining elements
    for (; i < length; ++i) {
        r[i] = v;
    }
}

inline void SSE2AddVectors(uint32_t* r, uint32_t v, uint64_t length) {
    uint64_t i = 0;
    __m128i add_val = _mm_set1_epi32(v);

    // Process 4 elements at a time
    for (; i + 4 <= length; i += 4) {
        __m128i vec = _mm_loadu_si128((__m128i*)&r[i]);
        vec = _mm_add_epi32(vec, add_val);
        _mm_storeu_si128((__m128i*)&r[i], vec);
    }

    // Handle remaining elements
    for (; i < length; ++i) {
        r[i] += v;
    }
}

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

inline void SSE2IsZeroVector(bool *t, const uint32_t* e, uint64_t e_length) {
    const __m128i zero = _mm_setzero_si128();
    uint64_t i = 0;

    // Process 4 elements at a time
    for (; i + 4 <= e_length; i += 4) {
        __m128i vec = _mm_loadu_si128((__m128i*)&e[i]);
        __m128i cmp = _mm_cmpeq_epi32(vec, zero);
        int mask = _mm_movemask_epi8(cmp);
        if (mask != 0xFFFF) {
            *t = false;
            return;
        }
    }

    // Handle remaining elements
    for (; i < e_length; ++i) {
        if (e[i] != 0) {
            *t = false;
            return;
        }
    }

    *t = true;
}
#endif

#ifdef __AVX2__
inline void AVX2RangeVector(uint32_t* r, uint32_t start, uint64_t length) {
    __m256i vec = _mm256_set_epi32(start + 7, start + 6, start + 5, start + 4,
                                   start + 3, start + 2, start + 1, start);
    __m256i inc = _mm256_set1_epi32(8);

    uint64_t i = 0;
    for (; i + 8 <= length; i += 8) {
        _mm256_storeu_si256((__m256i*)(r + i), vec);
        vec = _mm256_add_epi32(vec, inc);
    }

    start += i;
    for (; i < length; i++) {
        r[i] = start++;
    }
}

inline void AVX2SetVector(uint32_t* r, uint64_t length, uint32_t v) {
    uint64_t i = 0;

    // Process 4 elements at a time
    for (; i + 8 <= length; i += 8) {
        __m256i vr = _mm256_set1_epi32(v);
        _mm256_storeu_si256((__m256i*)(r + i), vr);
    }

    // Handle remaining elements
    for (; i < length; ++i) {
        r[i] = v;
    }
}

inline void AVX2AddToVector(uint32_t* r, uint32_t v, uint64_t length) {
    uint64_t i = 0;
    __m256i add_val = _mm256_set1_epi32(v);

    // Process 8 elements at a time
    for (; i + 8 <= length; i += 8) {
        __m256i vec = _mm256_loadu_si256((__m256i*)&r[i]);
        vec = _mm256_add_epi32(vec, add_val);
        _mm256_storeu_si256((__m256i*)&r[i], vec);
    }

    // Handle remaining elements
    for (; i < length; ++i) {
        r[i] += v;
    }
}

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

inline void AVX2IsZeroVector(bool *t, const uint32_t* e, uint64_t e_length) {
    const __m256i zero = _mm256_setzero_si256();
    uint64_t i = 0;

    // Process 8 elements at a time
    for (; i + 8 <= e_length; i += 8) {
        __m256i vec = _mm256_loadu_si256((__m256i*)&e[i]);
        __m256i cmp = _mm256_cmpeq_epi32(vec, zero);
        int mask = _mm256_movemask_epi8(cmp);
        if (mask != -1) { // -1 == all bits set
            *t = false;
            return;
        }
    }

    // Handle remaining elements
    for (; i < e_length; ++i) {
        if (e[i] != 0) {
            *t = false;
            return;
        }
    }

    *t = true;
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

int FieldRangeVector(
    uint32_t* r, uint64_t ro,
    uint32_t start, uint64_t length
) {
#if defined(__AVX2__)
    AVX2RangeVector(r + ro, start, length);
#elif defined(__SSE2__)
    SSE2RangeVector(r + ro, start, length);
#else
    NoSimdRangeVector(r + ro, start, length);
#endif
    return 1;
}

int FieldCopyVector(
    uint32_t* r, uint64_t ro,
    const uint32_t* a, uint64_t ao,
    uint64_t length
) {
    memcpy(r + ro, a + ao, sizeof(uint32_t) * length);
    return 1;
}

int FieldSetVector(
    uint32_t* r, uint64_t ro,
    uint64_t length, uint32_t v
) {
#if defined(__AVX2__)
    AVX2SetVector(r + ro, length, v);
#elif defined(__SSE2__)
    SSE2SetVector(r + ro, length, v);
#else
    NoSimdSetVector(r + ro, length, v);
#endif
    return 1;
}

int FieldAddToVector(
    uint32_t* r, uint64_t ro,
    uint32_t v,
    uint64_t length
) {
#if defined(__AVX2__)
    AVX2AddToVector(r + ro, v, length);
#elif defined(__SSE2__)
    SSE2AddToVector(r + ro, v, length);
#else
    NoSimdAddToVector(r + ro, v, length);
#endif
    return 1;
}

int FieldAddVectors(
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
    return 1;
}

int FieldMulVector(
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
    return 1;
}

int FieldMulVectors(
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
    return 1;
}

int FieldSubVectors(
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
    return 1;
}

int FieldNegVector(
    uint32_t* r, uint64_t ro,
    uint64_t length, uint32_t p
) {
#if defined(__AVX2__)
    AVX2NegVector(r + ro, length, p);
#elif defined(__SSE2__)
    SSE2NegVector(r + ro, length, p);
#else
    NoSimdNegVector(r + ro, length, p);
#endif
    return 1;
}

int FieldIsZeroVector(
    bool *t, uint64_t t_index,
    const uint32_t* e, uint64_t eo, uint64_t e_length
) {
#if defined(__AVX2__)
    AVX2IsZeroVector(t + t_index, e + eo, e_length);
#elif defined(__SSE2__)
    SSE2IsZeroVector(t + t_index, e + eo, e_length);
#else
    NoSimdIsZeroVector(t + t_index, e + eo, e_length);
#endif
    return 1;
}

int FieldAddVectorIfNonZero(
    bool* t, uint64_t t_index,
    uint32_t* r, uint64_t ro,
    const uint32_t* e, uint64_t eo,
    uint64_t length, uint32_t p
) {
    FieldIsZeroVector(t, t_index, e, eo, length);
    t[t_index] = !t[t_index];
    if (t[t_index]) {
        FieldAddVectors(r, ro, r, ro, e, eo, length, p);
    }
    return 1;
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* USE_FAST_CODE_WITH_CUDA */
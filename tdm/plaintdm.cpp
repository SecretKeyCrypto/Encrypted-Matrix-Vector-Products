#include "plaintdm.h"
#include <cstdint>
#include <algorithm>
#include "../dataobjects/plainfields.h"

void NoSimdPermutedExtentsAssign(
    uint32_t* r,
    uint32_t ro,
    uint32_t rfs,
    uint32_t rps,
    const uint32_t* s, // may be nullptr => zero source
    uint32_t so,
    uint32_t ss,
    uint32_t sc,
    uint64_t extent,
    const uint32_t* perm,    // may be nullptr => identity
    uint32_t po,
    uint64_t length)
{
    if (length == 0 || extent == 0) return;

    for (uint64_t i = 0; i < length; ++i) {
        const uint32_t p = perm ? perm[po + i] : static_cast<uint32_t>(i);
        const uint64_t ri_base = ro + i * rfs + static_cast<uint64_t>(p) * rps;
        const uint64_t si_base = so + i * ss;

        if (s) {
            for (uint64_t e = 0; e < extent; ++e) {
                r[ri_base + e] = s[si_base + e] + sc;
            }
        } else {
            // Source is logically zero; just write sc to r
            for (uint64_t e = 0; e < extent; ++e) {
                r[ri_base + e] = sc;
            }
        }
    }
}

#ifdef __SSE2__

#include <emmintrin.h>

static void SSE2PermutedExtentsAssign(
    uint32_t* r, uint32_t ro, uint32_t rfs, uint32_t rps,
    const uint32_t* s, uint32_t so, uint32_t ss, uint32_t sc,
    uint64_t extent, const uint32_t* perm, uint32_t po, uint64_t length)
{
    if (length == 0 || extent == 0) return;

    const __m128i vsc = _mm_set1_epi32(static_cast<int>(sc));

    for (uint64_t i = 0; i < length; ++i) {
        const uint32_t p = perm ? perm[po + i] : static_cast<uint32_t>(i);
        const uint64_t ri_base = ro + i * rfs + static_cast<uint64_t>(p) * rps;
        const uint64_t si_base = so + i * ss;

        uint64_t e = 0;
        if (s) {
            for (; e + 4 <= extent; e += 4) {
                __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s + si_base + e));
                v = _mm_add_epi32(v, vsc);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(r + ri_base + e), v);
            }
        } else {
            for (; e + 4 <= extent; e += 4) {
                _mm_storeu_si128(reinterpret_cast<__m128i*>(r + ri_base + e), vsc);
            }
        }
        // Tail
        for (; e < extent; ++e) {
            r[ri_base + e] = s ? (s[si_base + e] + sc) : sc;
        }
    }
}

#endif

#ifdef __AVX2__

#include <immintrin.h>

static void AVX2PermutedExtentsAssign(
    uint32_t* r, uint32_t ro, uint32_t rfs, uint32_t rps,
    const uint32_t* s, uint32_t so, uint32_t ss, uint32_t sc,
    uint64_t extent, const uint32_t* perm, uint32_t po, uint64_t length)
{
    if (length == 0 || extent == 0) return;

    const __m256i vsc = _mm256_set1_epi32(static_cast<int>(sc));

    for (uint64_t i = 0; i < length; ++i) {
        const uint32_t p = perm ? perm[po + i] : static_cast<uint32_t>(i);
        const uint64_t ri_base = ro + i * rfs + static_cast<uint64_t>(p) * rps;
        const uint64_t si_base = so + i * ss;

        uint64_t e = 0;
        if (s) {
            for (; e + 8 <= extent; e += 8) {
                __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + si_base + e));
                v = _mm256_add_epi32(v, vsc);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(r + ri_base + e), v);
            }
        } else {
            for (; e + 8 <= extent; e += 8) {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(r + ri_base + e), vsc);
            }
        }
        // Tail
        for (; e < extent; ++e) {
            r[ri_base + e] = s ? (s[si_base + e] + sc) : sc;
        }
    }
}

#endif

#ifdef __cplusplus
extern "C" {
#endif

bool PlainPermutedExtentsAssign(
    DoContext* ctx,
    uint32_t* r,
    uint32_t ro,
    uint32_t rfs,
    uint32_t rps,
    const uint32_t* s,
    uint32_t so,
    uint32_t ss,
    uint32_t sc,
    uint64_t extent,
    const uint32_t* perm,
    uint32_t po,
    uint64_t length)
{
#if defined(__AVX2__)
    AVX2PermutedExtentsAssign(r, ro, rfs, rps, s, so, ss, sc, extent, perm, po, length);
#elif defined(__SSE2__)
    SSE2PermutedExtentsAssign(r, ro, rfs, rps, s, so, ss, sc, extent, perm, po, length);
#else
    NoSimdPermutedExtentsAssign(r, ro, rfs, rps, s, so, ss, sc, extent, perm, po, length);
#endif
    return true;
}

bool PlainCircularCopy(DoContext* ctx, uint32_t* r, const uint32_t* v, uint64_t length) {
    uint64_t k = length;
    for (uint32_t t = 0; t < k; t++, r += k) {
        if (!PlainFieldCopyVector(ctx, r, t, v, 0, k - t)) return false;
        if (!PlainFieldCopyVector(ctx, r, 0, v, k - t, t)) return false;
    }
    return true;
}

#ifdef __cplusplus
} // extern "C"
#endif

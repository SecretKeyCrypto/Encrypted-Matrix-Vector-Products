#include "plainntt.h"
#include <iostream>
#include <cstdint>
#include <string.h>

#include "../dataobjects/dataobj.h"
#include "../dataobjects/fields.h"
#include "../dataobjects/plainfields.h"

inline u32 _mod_op(u64 n, u32 mod) {
    //return (u32)(n < mod ? n : (n % mod));
    return (u32)(n % mod);
}

// BIT REVERSAL PERMUTATION

static void _slower_bit_reversal_permutation(u32* perm, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        perm[i] = i;
    }
    size_t j = 0;
    for (size_t i = 1; i < n; ++i) {
        size_t bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            std::swap(perm[i], perm[j]);
        }
    }
}

static void _faster_bit_reversal_permutation(u32* perm, size_t n, size_t bits) {
    for (size_t i = 0; i < n; ++i) {
        size_t rev = 0;
        size_t x = i;
        for (size_t b = 0; b < bits; ++b) {
            rev = (rev << 1) | (x & 1);
            x >>= 1;
        }
        perm[i] = rev;
    }
}

u32* get_bit_reversal_permutation(size_t n) {
    static u32* cache[32] = {0};

    if (n && ((n & (n - 1)) != 0)) return nullptr;
    uint32_t index = 0;
    if (0xFFFF0000U & n) index += 16;
    if (0xFF00FF00U & n) index += 8;
    if (0xF0F0F0F0U & n) index += 4;
    if (0xCCCCCCCCU & n) index += 2;
    if (0xAAAAAAAAU & n) index += 1;

    if (cache[index] == nullptr) {
        cache[index] = (u32*)malloc(n * sizeof(u32));
        _faster_bit_reversal_permutation(cache[index], n, index);
    }
    return cache[index];
}

void apply_bit_reversal_permutation(u32* a, size_t n) {
    u32* perm = get_bit_reversal_permutation(n);
    u32 a0[n];
    memcpy(a0, a, n * sizeof(u32));
    for (size_t i = 0; i < n; ++i) {
        a[i] = a0[perm[i]];
    }
}

static const bool USE_FASTER_NTT = true;

// #define USE_FASTERNTT

#ifdef USE_FASTERNTT
#include "../FasterNTT/include/Rq.hpp"

const size_t qNum = 1;
using FNTT_R = fntt::Rq_d<qNum>;
#endif

inline u32 mod_pow(u32 base, u32 exp, u32 mod) {
    u64 result = 1;
    u64 b = base;
    while (exp > 0) {
        if (exp & 1) result = _mod_op(result * b, mod);
        b = _mod_op(b * b, mod);
        exp >>= 1;
    }
    return (u32)result;
}

inline u32 mod_inv(u32 a, u32 mod) {
    return mod_pow(a, mod - 2, mod); // Fermat's little theorem, assuming mod is prime
}

inline void print_array(const char* name, const u32* arr, size_t n) {
    std::cout << name << ": [ ";
    for (size_t i = 0; i < n; ++i) {
        std::cout << arr[i];
        if (i + 1 < n) std::cout << ", ";
    }
    std::cout << " ]" << std::endl;
}


#ifdef USE_FASTERNTT
void fntt_ntt(FNTT_R& rq, u32* a, size_t n) {
    u32 a0[n], a1[n];
    memcpy(a0, a, n * sizeof(u32));

    for (size_t i = 0; i < n; i++) {
        rq(0, i) = a[i];
    }
    rq.ntt();
    for (size_t i = 0; i < n; i++) {
        a[i] = rq(0, i);
    }
    
    // NOTE - for debug checking
    // memcpy(a1, a, n * sizeof(u32));
    // for (size_t i = 0; i < n; i++) {
    //     rq(0, i) = a1[i];
    // }
    //
    // rq.intt();
    // for (size_t i = 0; i < n; i++) {
    //     a1[i] = rq(0, i);
    // }
    // for (size_t i=0; i<n; i++) {
    //     if (a0[i] != a1[i]) {
    //         std::cerr << "==========================" << std::endl;
    //         std::cerr << "n=" << n << " i=" << i;
    //         std::cerr << " a0";
    //         for (size_t j=std::max(i-i, i-3); j<std::min(n, i+3); j++) {
    //             std::cerr << "=" << a0[j];
    //         }
    //         std::cerr << " a1";
    //         for (size_t j=std::max(i-i, i-3); j<std::min(n, i+3); j++) {
    //             std::cerr << "=" << a1[j];
    //         }
    //         std::cerr << std::endl;
    //         std::cerr << "==========================" << std::endl;
    //         break;
    //     }
    // }
}
void fntt_intt(FNTT_R& rq, u32* a, size_t n) {
    for (size_t i = 0; i < n; i++) {
        rq(0, i) = a[i];
    }
    rq.intt();
    for (size_t i = 0; i < n; i++) {
        a[i] = rq(0, i);
    }
}

void fntt_ntt(u32* a, size_t n, u32 root, u32 mod) {
    constexpr size_t qNum = 1;
    uint64_t p[qNum] = {mod};
    FNTT_R rq(n, p, 0);
    fntt_ntt(rq, a, n);
}

void fntt_intt(u32* a, size_t n, u32 root, u32 mod) {
    constexpr size_t qNum = 1;
    uint64_t p[qNum] = {mod};
    FNTT_R rq(n, p, 0);
    fntt_intt(rq, a, n);
}
#endif

bool faster_ntt(u32* a, size_t n, u32 root, u32 mod) {
    apply_bit_reversal_permutation(a, n);

    size_t k = 0;
    for (size_t len = 2; len <= n; len <<= 1, k++);
    u32 wlens[k];
    if (k > 0) {
        wlens[--k] = root;
        while (--k != -1) {
            wlens[k] = _mod_op((u64)wlens[k + 1] * (u64)wlens[k + 1], mod);
        }
        ++k;
    }

    for (size_t len = 2; len <= n; len <<= 1, k++) {
        size_t half_len = len >> 1;
        u32 wlen = wlens[k];
        u32 ws[half_len];
        ws[0] = 1;
        for (size_t j = 1; j < half_len; ++j) {
            ws[j] = _mod_op((u64)ws[j - 1] * wlen, mod);
        }
        // NOTE - for debug printing
        // std::cerr << "Stage len=" << len << " half=" << half_len << " W[0..half-1]:";
        // for (size_t j = 0; j < std::min<size_t>(half_len, 32); ++j) {
        //     std::cerr << ' ' << ws[j];
        // }
        // std::cerr << std::endl;

        for (size_t i = 0; i < n; i += len) {
            for (size_t j = 0; j < half_len; ++j) {
                u32 w = ws[j];
                u32 u = a[i + j];
                u32 v = _mod_op((u64)a[i + j + half_len] * w, mod);
                u32 sum = u + v;
                a[i + j] = sum < mod ? sum : sum - mod;
                u32 sub = mod + u - v;
                a[i + j + half_len] = sub < mod ? sub : sub - mod;
            }
        }
    }
    return true;
}

bool faster_intt(DoContext* ctx, u32* a, u32 ao, size_t n, u32 root, u32 mod) {
    u32 inv_root = mod_inv(root, mod);

    faster_ntt(a + ao, n, inv_root, mod);

    u32 inv_n = mod_inv(n, mod);
    if (USE_FAST_CODE) {
        if (!PlainFieldMulVector(ctx, a, ao, a, ao, inv_n, n, mod)) {
            return false;
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            a[ao + i] = _mod_op((u64)a[ao + i] * inv_n, mod);
        }
    }
    return true;
}


bool orig_ntt(u32* a, size_t n, u32 root, u32 mod) {
    // Bit-reversal permutation
    size_t j = 0;
    for (size_t i = 1; i < n; ++i) {
        size_t bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            std::swap(a[i], a[j]);
        }
    }

    for (size_t len = 2; len <= n; len <<= 1) {
        u32 wlen = mod_pow(root, n / len, mod);

        for (size_t i = 0; i < n; i += len) {
            u32 w = 1;
            for (size_t j = 0; j < len / 2; ++j) {
                u32 u = a[i + j];
                u32 v = _mod_op((u64)a[i + j + len / 2] * w, mod);
                a[i + j] = _mod_op(u + v, mod);
                a[i + j + len / 2] = _mod_op(mod + u - v, mod);
                w = _mod_op((u64)w * wlen, mod);
            }
        }
    }
    return false;
}

bool orig_intt(DoContext* ctx, u32* a, u32 ao, size_t n, u32 root, u32 mod) {
    u32 inv_root = mod_inv(root, mod);

    orig_ntt(a + ao, n, inv_root, mod);

    u32 inv_n = mod_inv(n, mod);
    if (USE_FAST_CODE) {
        if (!PlainFieldMulVector(ctx, a, ao, a, ao, inv_n, n, mod)) {
            return false;
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            a[ao + i] = _mod_op((u64)a[ao + i] * inv_n, mod);
        }
    }
    return true;
}

#ifdef __cplusplus
extern "C" {
#endif

bool plain_ntt_1(DoContext* ctx, u32* a, u32 n, u32 root, u32 mod) {
#ifdef USE_FASTERNTT
    fntt_ntt(a, n, root, mod);
#else
    if (USE_FASTER_NTT) {
        if (!faster_ntt(a, n, root, mod)) {
            return false;
        }
    } else {
        if (!orig_ntt(a, n, root, mod)) {
            return false;
        }
    }
#endif
    return true;
}

bool plain_ntt(DoContext* ctx, u32* a, u32 ao, u32 n, u32 stride, u32 steps, u32 root, u32 mod) {
    for (uint32_t i = 0; i < steps; i++, a += stride) {
        if (!plain_ntt_1(ctx, a + ao, n, root, mod)) {
            return false;
        }
    }
    return true;
}

bool plain_intt_1(DoContext* ctx, u32* a, u32 ao, u32 n, u32 root, u32 mod) {
#ifdef USE_FASTERNTT
    fntt_intt(a, n, root, mod);
#else
    if (USE_FASTER_NTT) {
        if (!faster_intt(ctx, a, ao, n, root, mod)) {
            return false;
        }
    } else {
        if (!orig_intt(ctx, a, ao, n, root, mod)) {
            return false;
        }
    }
#endif
    return true;
}

bool plain_intt(DoContext* ctx, u32* a, u32 ao, u32 n, u32 stride, u32 steps, u32 root, u32 mod) {
    for (uint32_t i = 0; i < steps; i++, a += stride) {
        if (!plain_intt_1(ctx, a, ao, n, root, mod)) {
            return false;
        }
    }
    return true;
}

// Convolution using NTT
bool plain_ntt_convolution_1(DoContext* ctx, const u32* a, u32 ao, const u32* b, u32 bo, u32* result, u32 ro, u32 n, u32 root, u32 mod) {
    alignas(ALIGNMENT) u32 fa[n];
    alignas(ALIGNMENT) u32 fb[n];
    if (USE_FAST_CODE) {
        memcpy(fa, a + ao, n * sizeof(u32));
        memcpy(fb, b + bo, n * sizeof(u32));
    } else {
        for (size_t i = 0; i < n; ++i) {
            fa[i] = a[ao + i];
            fb[i] = b[bo + i];
        }
    }

#ifdef USE_FASTERNTT
    constexpr size_t qNum = 1;
    uint64_t p[qNum] = {mod};
    FNTT_R rq(n, p, 0);
    fntt_ntt(rq, fa, n);
    fntt_ntt(rq, fb, n);
#else
    plain_ntt_1(ctx, fa, n, root, mod);
    plain_ntt_1(ctx, fb, n, root, mod);
#endif

    if (USE_FAST_CODE) {
        // FieldMulVectors(fa, 0, fa, 0, fb, 0, n, mod);
        if (!PlainFieldMulVectors(ctx, result, ro, fa, 0, fb, 0, n, mod)) { // NOTE: plain since fa, fb are stack-allocated
            return false;
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            result[ro + i] = _mod_op((u64)fa[i] * fb[i], mod);
        }
    }

#ifdef USE_FASTERNTT
    if (!plain_intt(rq, result, n)) {
        return false;
    }
#else
    if (!plain_intt_1(ctx, result, ro, n, root, mod)) {
        return false;
    }
#endif
    return true;
}

bool plain_ntt_convolution(DoContext* ctx, const u32* a, u32 ao, u32 as, const u32* b, u32 bo, u32 bs, u32* result, u32 ro, u32 rs, u32 n, u32 steps, u32 root, u32 mod) {
    for (uint32_t i = 0; i < steps; i++, ao += as, bo += bs, ro += rs) {
        if (!plain_ntt_convolution_1(ctx, a, ao, b, bo, result, ro, n, root, mod)) {
            return false;
        }
    }
    return true;
}

/*
inline void poly_mod_xt_minus_1(const u32* a, u32* b, size_t n, size_t t, u32 mod) {
    // Initialize output buffer
    for (size_t i = 0; i < t; ++i) b[i] = 0;

    // Perform modulo reduction
    for (size_t i = 0, j = 0; i < n; ++i, j++) {
        if (j == t) j = 0;
        b[j] = _mod_op(b[j] + a[i], mod);
    }
}
*/

#ifdef __cplusplus
} // extern "C"
#endif

#include "NTT.h"
#include <cassert>
#include <cstdlib>  // for rand()

inline u32 _mod_op(u64 n, u32 mod) {
    //return (u32)(n < mod ? n : (n % mod));
    return (u32)(n % mod);
}

#ifdef USE_FAST_CODE_WITH_CUDA

#include "cudantt.h"
#include "plainntt.h"
#include "../dataobjects/common.h"

#ifdef __cplusplus
extern "C" {
#endif

bool ntt(DoContext* ctx, u32* a, u32 ao, u32 n, u32 stride, u32 steps, u32 root, u32 mod) {
    return cuda_call(ntt(ctx, a, ao, n, stride, steps, root, mod));
}

bool intt(DoContext* ctx, u32* a, u32 ao, u32 n, u32 stride, u32 steps, u32 root, u32 mod) {
    return cuda_call(intt(ctx, a, ao, n, stride, steps, root, mod));
}

bool ntt_convolution(DoContext* ctx, const u32* a, u32 ao, u32 as, const u32* b, u32 bo, u32 bs, u32* result, u32 ro, u32 rs, u32 n, u32 steps, u32 root, u32 mod) {
    return cuda_call(ntt_convolution(ctx, a, ao, as, b, bo, bs, result, ro, rs, n, steps, root, mod));
}

#ifdef __cplusplus
} // extern "C"
#endif

#else /* USE_FAST_CODE_WITH_CUDA */

#include "plainntt.h"

#ifdef __cplusplus
extern "C" {
#endif

bool ntt(DoContext* ctx, u32* a, u32 ao, u32 n, u32 stride, u32 steps, u32 root, u32 mod) {
    return plain_ntt(ctx, a, ao, n, stride, steps, root, mod);
}

bool intt(DoContext* ctx, u32* a, u32 ao, u32 n, u32 stride, u32 steps, u32 root, u32 mod) {
    return plain_intt(ctx, a, ao, n, stride, steps, root, mod);
}

bool ntt_convolution(DoContext* ctx, const u32* a, u32 ao, u32 as, const u32* b, u32 bo, u32 bs, u32* result, u32 ro, u32 rs, u32 n, u32 steps, u32 root, u32 mod) {
    return plain_ntt_convolution(ctx, a, ao, as, b, bo, bs, result, ro, rs, n, steps, root, mod);
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* USE_FAST_CODE_WITH_CUDA */

// Modular exponentiation: (base^exp) % mod
inline uint32_t modExponent(uint32_t base, uint32_t exp, uint32_t mod) {
    uint64_t result = 1;
    uint64_t b = _mod_op(base, mod);
    while (exp > 0) {
        if (exp & 1)
            result = _mod_op(result * b, mod);
        b = _mod_op(b * b, mod);
        exp >>= 1;
    }
    return (uint32_t)result;
}

// Check if beta^k == 1 mod M for some 2 ≤ k < N
inline bool existSmallN(uint32_t beta, uint32_t M, uint32_t N) {
    uint64_t b = beta;
    for (uint32_t k = 2; k < N; ++k) {
        b = _mod_op(b * beta, M);  // b is now beta^k
        if (b == 1)
            return true;
    }
    return false;
}

#ifdef __cplusplus
extern "C" {
#endif

// Return a primitive N-th root of unity modulo M
uint32_t NthRootOfUnity(uint32_t M, uint32_t N) {
    assert(M > 1);
    assert(_mod_op(M - 1, N) == 0);  // Ensure N divides M-1
    uint32_t phi = M - 1;

    while (true) {
        uint32_t alpha = 1 + _mod_op((u32)rand(), M - 1); // pick alpha ∈ [1, M-1]
        uint32_t beta = modExponent(alpha, phi / N, M);
        if (!existSmallN(beta, M, N)) {
            return beta;
        }
    }
}

#ifdef __cplusplus
} // extern "C"
#endif

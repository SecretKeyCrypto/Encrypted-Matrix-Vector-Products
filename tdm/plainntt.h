#ifndef _PLAINNTT_H_
#define _PLAINNTT_H_

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "../dataobjects/docontext.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t u32;
typedef uint64_t u64;

// Performs in-place NTT on the array `a` of length `n` with root of unity `root` modulo `mod`
bool plain_ntt(DoContext* ctx, u32* a, u32 ao, u32 n, u32 stride, u32 steps, u32 root, u32 mod);

bool plain_intt(DoContext* ctx, u32* a, u32 ao, u32 n, u32 stride, u32 steps, u32 root, u32 mod);

bool plain_ntt_convolution(DoContext* ctx, const u32* a, u32 ao, u32 as, const u32* b, u32 bo, u32 bs, u32* result, u32 ro, u32 rs, u32 n, u32 steps, u32 root, u32 mod);

#ifdef __cplusplus
} // exterm "C"
#endif

#endif /* _PLAINNTT_H_ */
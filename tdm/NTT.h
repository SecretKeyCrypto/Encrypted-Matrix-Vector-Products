#ifndef WRAPPER_H
#define WRAPPER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t u32;
typedef uint64_t u64;

// Performs in-place NTT on the array `a` of length `n` with root of unity `root` modulo `mod`
void ntt(u32* a, size_t n, u32 root, u32 mod);

// Performs in-place inverse NTT
//FIXME void intt(u32* a, size_t n, u32 root, u32 mod);

void ntt_convolution(const u32* a, const u32* b, u32* result, size_t n, u32 root, u32 mod);

uint32_t NthRootOfUnity(u32 M, u32 N);

//FIXME u32 mod_pow(u32 base, u32 exp, u32 mod);
//FIXME u32 mod_inv(u32 a, u32 mod);

#ifdef __cplusplus
}
#endif

#endif
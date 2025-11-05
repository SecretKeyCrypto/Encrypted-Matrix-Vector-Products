#ifndef _EMVP_COMMON_H_
#define _EMVP_COMMON_H_

//#ifdef USE_FAST_CODE_WITH_CUDA

#include <inttypes.h>

constexpr uint32_t threadsPerBlock = 256;
constexpr uint64_t blocksPerGridFor(uint64_t length) {
    return (length + threadsPerBlock - 1) / threadsPerBlock;
}

//#endif

inline uint32_t bitmask_for(uint32_t v) {
    if (v == 0) return 1;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v;
}

#endif /* _EMVP_COMMON_H_ */
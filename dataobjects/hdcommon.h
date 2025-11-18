#ifndef _EMVP_COMMON_H_
#define _EMVP_COMMON_H_

#include <cstdint>

constexpr int defaultMaxThreads = 256;
constexpr int defaultMaxThreads2D = 16;
constexpr inline uint64_t threadsPerBlockFor(uint64_t width, int elementsPerThread, int maxThreads = defaultMaxThreads) {
    uint64_t totalThreadsX = (width + elementsPerThread - 1) / elementsPerThread;
    return totalThreadsX < maxThreads ? totalThreadsX : maxThreads;
}
constexpr inline uint64_t blocksPerGridFor(uint64_t width, uint64_t threadsPerBlockWidth, int elementsPerThread) {
    uint64_t totalThreadsX = (width + elementsPerThread - 1) / elementsPerThread;
    return (totalThreadsX + threadsPerBlockWidth - 1) / threadsPerBlockWidth;
}

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
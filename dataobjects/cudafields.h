#ifndef _CUDAFIELDS_H
#define _CUDAFIELDS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void CudaRangeVector(uint32_t* r, uint32_t start, uint64_t length);
void CudaCopyVector(uint32_t* r, const uint32_t* a, uint64_t length);
void CudaSetVector(uint32_t* r, uint64_t length, uint32_t v);
void CudaAddToVector(uint32_t* r, uint32_t v, uint64_t length);
void CudaAddVectors(uint32_t* r, const uint32_t* a, const uint32_t* b, uint64_t length, uint32_t p);
void CudaMulVector(uint32_t* r, const uint32_t* a, uint32_t b, uint64_t length, uint32_t p);
void CudaMulVectors(uint32_t* r, const uint32_t* a, const uint32_t* b, uint64_t length, uint32_t p);
void CudaSubVectors(uint32_t* r, const uint32_t* a, const uint32_t* b, uint64_t length, uint32_t p);
void CudaNegVector(uint32_t* r, uint64_t length, uint32_t p);
void CudaIsZeroVector(bool *t, const uint32_t* e, uint64_t length);
void CudaAddVectorIfNonZero(bool* t, uint32_t* r, const uint32_t* e, uint64_t length, uint32_t p);

#ifdef __cplusplus
}
#endif

#endif /* _CUDAFIELDS_H */
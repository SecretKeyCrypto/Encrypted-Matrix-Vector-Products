#ifndef __CUDANTT_H_
#define __CUDANTT_H_

#include <cstdint>
#include <cstddef>
#include "../dataobjects/docontext.h"

#ifdef __cplusplus
extern "C" {
#endif

bool cuda_ntt(DoContext* ctx,
              uint32_t* a,
              uint32_t ao,
              uint32_t n,
              uint32_t stride,
              uint32_t steps,
              uint32_t root,
              uint32_t mod);

bool cuda_intt(DoContext* ctx,
               uint32_t* a,
               uint32_t ao,
               uint32_t n,
               uint32_t stride,
               uint32_t steps,
               uint32_t root,
               uint32_t mod);

bool cuda_ntt_convolution(DoContext* ctx,
                          const uint32_t* a,
                          uint32_t ao,
                          uint32_t as,
                          const uint32_t* b,
                          uint32_t bo,
                          uint32_t bs,
                          uint32_t* result,
                          uint32_t ro,
                          uint32_t rs,
                          uint32_t n,
                          uint32_t steps,
                          uint32_t root,
                          uint32_t mod);

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* __CUDANTT_H_ */
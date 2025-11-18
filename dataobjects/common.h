#ifndef _COMMON_H_
#define _COMMON_H_

#ifdef USE_CUDA_TESTING

#include "cudaobj.h"

#ifdef __cplusplus
extern "C" {
#endif

bool _cuda_call(bool value);

#ifdef __cplusplus
} // extern "C"
#endif

#define CUDA_CALL(call) _cuda_call(Cuda ## call)
#define cuda_call(call) _cuda_call(cuda_ ## call)
#define CUDA_CALL_BYPASS(call) Plain ## call
#define cuda_call_bypass(call) plain_ ## call

#else /* USE_CUDA_TESTING */

#define CUDA_CALL(call) Cuda ## call
#define cuda_call(call) cuda_ ## call
#define CUDA_CALL_BYPASS(call) Cuda ## call
#define cuda_call_bypass(call) cuda_ ## call

#endif /* USE_CUDA_TESTING */

#ifdef __cplusplus
extern "C" {
#endif

void Setup();

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* _COMMON_H_ */

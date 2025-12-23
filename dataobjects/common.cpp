#include "common.h"
#include <iostream>

// #define USE_COMMON_DEBUG

#ifdef USE_FAST_CODE_WITH_CUDA

#ifdef USE_CUDA_TESTING

#ifdef __cplusplus
extern "C" {
#endif

bool _cuda_call(bool value) {
    cuda_sync();
    return value;
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* USE_CUDA_TESTING */

#ifdef __cplusplus
extern "C" {
#endif

void _CudaSetup();

void Setup() {
#ifdef USE_COMMON_DEBUG
    std::cerr << "Cuda setup" << std::endl;
#endif /* USE_COMMON_DEBUG */
    _CudaSetup();
}

#ifdef __cplusplus
} // extern "C"
#endif

#else /* USE_FAST_CODE_WITH_CUDA */

#ifdef __cplusplus
extern "C" {
#endif

void Setup() {
#ifdef USE_COMMON_DEBUG
    std::cerr << "Plain setup" << std::endl;
#endif /* USE_COMMON_DEBUG */
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* USE_FAST_CODE_WITH_CUDA */
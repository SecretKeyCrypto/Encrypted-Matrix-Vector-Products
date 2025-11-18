#include "cudaobj.h"
#include "cudagraph.h"
#include <cuda_runtime.h>

#ifdef USE_CUDA_TESTING
#include <string.h>
#endif /* USE_CUDA_TESTING */

//uncomment for debugging
#define _CUDAOBJ_DEBUG_CUDA

#ifdef _CUDAOBJ_DEBUG_CUDA

#include <iostream>

static cudaError_t _CudaCheck(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << " : " << cudaGetErrorName(err) << " : " << cudaGetErrorString(err) << std::endl;
    }
    return err;
}

#define STRINGIZE_DETAIL(x) #x
#define STRINGIZE(x) STRINGIZE_DETAIL(x)
#define _CUDA_CHECK(cudaCall) _CudaCheck(cudaCall, __FILE__ ":" STRINGIZE(__LINE__) " : " #cudaCall)

#else /* _CUDAOBJ_DEBUG_CUDA */

#define _CUDA_CHECK(cudaCall) (cudaCall)

#endif /* _CUDAOBJ_DEBUG_CUDA */

#ifdef __cplusplus
extern "C" {
#endif

bool cuda_doreset(DoContext* doctx) {
    CudaVectorGraph* graph = CudaVectorGraph::Get(doctx);
    if (!graph) return false;
    return graph->Reset();
}

void* cuda_doalloc(DoContext* doctx, size_t size) {
    CudaVectorGraph* graph = CudaVectorGraph::Get(doctx);
    if (!graph) return NULL;
	return graph->Malloc(size);
}

bool cuda_dofree(DoContext* doctx, void* ptr) {
    CudaVectorGraph* graph = CudaVectorGraph::Get(doctx);
    if (!graph) return false;
	return graph->Free(ptr);
}

bool cuda_dosync(DoContext* doctx) {
    CudaVectorGraph* graph = CudaVectorGraph::Get(doctx);
    if (!graph) return false;
    if (!graph->Build() ) return false;
    if (!graph->Execute()) return false;
    if (_CUDA_CHECK(cudaDeviceSynchronize()) != cudaSuccess) return false;
    return true;
}

void* cuda_alloc(size_t size) {
    void* ptr;
    _CUDA_CHECK(cudaMallocManaged(&ptr, size, cudaMemAttachGlobal));
	_CUDA_CHECK(cudaMemset(ptr, 0, size));
#ifdef _CUDAOBJ_DEBUG_CUDA
    std::cerr << "cuda_alloc: ptr=" << ptr << std::endl;
#endif /* _CUDAOBJ_DEBUG_CUDA */
    return ptr;
}

void cuda_free(void* ptr) {
#ifdef _CUDAOBJ_DEBUG_CUDA
    std::cerr << "cuda_free: ptr=" << ptr << std::endl;
#endif /* _CUDAOBJ_DEBUG_CUDA */
    _CUDA_CHECK(cudaFree(ptr));
}

void cuda_sync() {
	_CUDA_CHECK(cudaDeviceSynchronize());
}

#ifdef __cplusplus
} // extern "C"
#endif

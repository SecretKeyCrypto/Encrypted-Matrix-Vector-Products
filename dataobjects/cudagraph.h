#pragma once

#include <mutex>
#include <vector>
#include <functional>

#include "docontext.h"

class CudaVectorGraphImpl;

class CudaVectorGraph {
public:

    static CudaVectorGraph* Get(DoContext* ctx);

    struct Params {
        int elementsPerThread;
        int dimElementsPerThread;
        int randElementsPerThread;
        int mvpElementsPerThread;
    };

    CudaVectorGraph();
    ~CudaVectorGraph();

    const Params& GetParams() const;

    // Graph lifecycle
    bool Reset();

    // Vector operations
    void* Malloc(size_t bytes);
    bool Free(const void* ptr);
    size_t Size(const void *ptr) const;

    // Copy operations
    bool CopyToDevice(void* ptr, const void* hostSrc,
                      size_t bytes, size_t start = 0);
    bool CopyToHost(void* hostDst, const void* ptr,
                    size_t bytes, size_t start = 0);

    // Compute operation
    bool Compute(const std::vector<const void*>& inputs,
                 const std::vector<void*>& outputs,
                 const std::function<bool(cudaStream_t)>& op);

    // Build graph
    bool Build();

    // Execute built graph
    bool Execute(cudaStream_t stream = 0);

private:
    CudaVectorGraphImpl* impl_;
    mutable std::recursive_mutex mutex_;
};

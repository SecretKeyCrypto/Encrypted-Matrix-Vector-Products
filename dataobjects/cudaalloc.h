#ifndef _CUDAALLOC_H
#define _CUDAALLOC_H

#include <cuda_runtime.h>
#include "cudagraph.h"

template <typename T>
struct GraphAllocator {
    using value_type = T;
    CudaVectorGraph* graph;

    // Constructor binding to a graph
    explicit GraphAllocator(CudaVectorGraph* graph) : graph(graph) {}

    // Copy constructor
    GraphAllocator(const GraphAllocator& other) : graph(other.graph) {}

    // Assignment operator
    GraphAllocator& operator=(const GraphAllocator& other) {
        if (this != &other) {
            graph = other.graph;
        }
        return *this;
    }

    // Allocate/deallocate
    T* allocate(std::size_t n) {
        return static_cast<T*>(graph->Malloc(n));
    }

    bool deallocate(T* ptr) {
        return graph->Free(ptr);
    }
};


#endif /* _CUDAALLOC_H */
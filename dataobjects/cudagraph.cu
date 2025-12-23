#include "cudagraph.h"
#include <cuda_runtime.h>
#include <set>
#include <memory>
#include <iostream>
#include <functional>
#include <thread>
#include <unordered_set>

#include "docontextimpl.h"
#include "cudacommon.h"

// -----------------------------------------------------------------------------
// Usage Contract for CudaVectorGraphImpl
// -----------------------------------------------------------------------------
// 1. Vector IDs:
//    - The `void*` returned by Malloc is associated with an internal vector.
//    - Methods validate that the vector is currently allocated and not freed.
//    - Freed or unknown pointers are rejected safely.
//
// 2. Graph State:
//    - Each successful method call adds at most one node to the CUDA graph.
//    - Dependencies are tracked via vector producer nodes, ensuring a correct DAG.
//    - Any sequence of successful calls leaves the graph in a valid state.
//
// 3. Error Handling:
//    - If a method fails, temporary resources are cleaned up automatically (RAII).
//    - No vector is inserted into the active set unless the method succeeds.
//    - The graph remains unchanged if the method fails before node insertion.
//    - There is no support for rolling back a successful addition to the graph.
//    - To discard the entire graph after an error, call Reset().
//
// -----------------------------------------------------------------------------

static bool printGraphNodes(cudaGraph_t graph, int depth = 0) {
    size_t numNodes = 0;
    cudaGraphGetNodes(graph, nullptr, &numNodes);
    std::vector<cudaGraphNode_t> nodes(numNodes);
    cudaGraphGetNodes(graph, nodes.data(), &numNodes);

    for (auto n : nodes) {
        cudaGraphNodeType type;
        cudaGraphNodeGetType(n, &type);

        for (int i = 0; i < depth; i++) std::cerr << "  "; // indent

        std::cerr << "Node type=" << type << std::endl;

        if (type == cudaGraphNodeTypeKernel) {
            cudaKernelNodeParams kp{};
            if (_CUDA_CHECK(cudaGraphKernelNodeGetParams(n, &kp)) != cudaSuccess) {
                return false;
            }
            for (int i = 0; i < depth; i++) std::cerr << "  ";
            std::cerr << "  Kernel func=" << kp.func
                << " grid=(" << kp.gridDim.x << "," << kp.gridDim.y << "," << kp.gridDim.z << ")"
                << " block=(" << kp.blockDim.x << "," << kp.blockDim.y << "," << kp.blockDim.z << ")"
                << std::endl;
        }
        else if (type == cudaGraphNodeTypeGraph) {
            cudaGraph_t child;
            if (_CUDA_CHECK(cudaGraphChildGraphNodeGetGraph(n, &child)) != cudaSuccess) {
                return false;
            }
            if (!printGraphNodes(child, depth + 1)) { // recurse
                return false;
            }
        }
    }
    return true;
}

/*
static bool dotPrintGraph(cudaGraph_t graph, const char* filename = nullptr) {
    if (!filename) {
        filename = "graph.dot";
    }
    unsigned int flags =
        cudaGraphDebugDotFlagsVerbose |
        cudaGraphDebugDotFlagsKernelNodeParams |
        cudaGraphDebugDotFlagsMemcpyNodeParams |
        cudaGraphDebugDotFlagsMemsetNodeParams |
        cudaGraphDebugDotFlagsHostNodeParams |
        cudaGraphDebugDotFlagsEventNodeParams;
    if (_CUDA_CHECK(cudaGraphDebugDotPrint(graph, filename, flags)) != cudaSuccess) {
        return false;
    }
    return true;
}

static bool dotPrintGraphInterim(cudaGraph_t graph, const char* filename = nullptr) {
    size_t numNodes = 0;
    cudaGraphGetNodes(graph, nullptr, &numNodes);
    if (numNodes < 50) {
        return dotPrintGraph(graph, filename);
    }
    return true;
}
*/

thread_local struct {
    bool recursingImmediate = false;
    bool recursingCapture = false;
} _graphTLS;

class CudaVectorGraphImpl {
    friend class CudaVectorGraph;

#ifdef USE_CUDA_TESTING
    static constexpr bool immediateGraphOp = true;
    static constexpr bool noCaptureGraphOp = true;
#else /* USE_CUDA_TESTING */
    // TODO - make it work with immediateGraphOp and noCaptureGraphOp set to false
    static constexpr bool immediateGraphOp = true;
    static constexpr bool noCaptureGraphOp = true;
#endif /* USE_CUDA_TESTING */
private:
    struct VectorInfo;

    static constexpr cudaStreamCaptureMode streamCaptureMode = cudaStreamCaptureModeGlobal;

    template <typename T>
    static void Uniqify(std::vector<T>& deps) {
        std::unordered_set<T> seen;
        std::vector<T> uniq;
        uniq.reserve(deps.size());
        for (auto n : deps) if (seen.insert(n).second) uniq.push_back(n);
        deps.swap(uniq);
    }

public:
    CudaVectorGraphImpl() : params_(), graph_(nullptr), graphExec_(nullptr) {
        params_.elementsPerThread = 8;
        params_.dimElementsPerThread = 4;
        params_.randElementsPerThread = 8;
        params_.mvpElementsPerThread = 1;
        Reset();
    }
    ~CudaVectorGraphImpl() { Cleanup(true); }

    const CudaVectorGraph::Params& GetParams() const {
        return params_;
    }

    bool Valid() const {
        return !graph_;
    }

    bool Reset() {
        Cleanup();
        return Valid();
    }

private:
    // Register readers (parallel allowed)
    void RegisterReaders(cudaGraphNode_t node, const std::vector<const void*>& inputs)
    {
        for (auto ip : inputs) {
            if (!ip) continue; // ignore nullptr inputs
            void* key = const_cast<void*>(ip);
            auto it = vectors_.find(key);
            if (it == vectors_.end()) continue; // inputs must exist, but defensive

            VectorInfo* vi = it->second;
            vi->activeReaders.push_back(node);
        }
    }

    // Register writers (clear readers, update lastWriter)
    void RegisterWriters(cudaGraphNode_t node, const std::vector<void*>& outputs)
    {
        for (auto op : outputs) {
            auto it = vectors_.find(op);
            if (it == vectors_.end()) continue; // must exist, but defensive

            VectorInfo* vi = it->second;
            vi->lastWriter = node;
            vi->activeReaders.clear(); // buffer contents changed
        }
    }

    template <typename PtrVec>
    bool CollectDeps(std::vector<cudaGraphNode_t>& deps,
                     bool forOutputs,
                     const PtrVec& ptrs)
    {
        for (auto ptr : ptrs) {
            // ignore nullptr inputs
            if (!forOutputs && !ptr) continue;
            auto it = vectors_.find(ptr);
            // output too must already exist by calling Malloc first
            if (it == vectors_.end()) {
                return false;
            }

            VectorInfo* vi = it->second;
            if (vi->lastWriter) {
                deps.push_back(vi->lastWriter);
            }
            if (forOutputs) {
                for (auto rn : vi->activeReaders) {
                    deps.push_back(rn);
                }
            }
        }
        return true;
    }

    bool Deps(std::vector<cudaGraphNode_t>& deps,
              const std::vector<const void*>& inputs,
              const std::vector<void*>& outputs)
    {
        deps.clear();

        // Collect input dependencies - must wait for input nodes
        if (!CollectDeps(deps, false, inputs)) return false;

        // Collect output dependencies - must wait for existing output nodes (like malloc)
        if (!CollectDeps(deps, true, outputs)) return false;

        Uniqify(deps);

        return true;
    }

public:
    // -------------------------------
    // 1. Malloc
    // -------------------------------
    void* Malloc(size_t bytes) {
        auto vi = std::make_unique<VectorInfo>();
        vi->ptr = nullptr;
        vi->bytes = bytes;
        vi->node = nullptr;
        // bytes += sizeof(uint32_t); // NOTE - avoids incorrect memcheck errors at allocation boundary

#ifdef USE_CUDA_TESTING
        if (_CUDA_CHECK(cudaMallocManaged(&vi->ptr, bytes)) != cudaSuccess) {
            return nullptr;
        }
        // NOTE - does not work in WSL2
        // int device;
        // if (_CUDA_CHECK(cudaGetDevice(&device)) != cudaSuccess) {
        //     return nullptr;
        // }
        // cudaMemLocation loc;
        // loc.type = cudaMemLocationTypeDevice;
        // loc.id = device;
        // int supported = 0;
        // cudaDeviceGetAttribute(&supported, cudaDevAttrConcurrentManagedAccess, 0);
        // if (supported) {
        //     if (_CUDA_CHECK(cudaMemPrefetchAsync(vi->ptr, bytes, loc, 0, 0)) != cudaSuccess) {
        //         return nullptr;
        //     }
        //     if (_CUDA_CHECK(cudaStreamSynchronize(0)) != cudaSuccess) {
        //         return nullptr;
        //     }
        // } else {
        //     if (_CUDA_CHECK(cudaMemAdvise(vi->ptr, bytes, cudaMemAdviseSetPreferredLocation, loc)) != cudaSuccess) {
        //         return nullptr;
        //     }
        //     if (_CUDA_CHECK(cudaMemAdvise(vi->ptr, bytes, cudaMemAdviseSetAccessedBy, loc)) != cudaSuccess) {
        //         return nullptr;
        //     }
        // }
#else /* USE_CUDA_TESTING */
        if (_CUDA_CHECK(cudaMalloc(&vi->ptr, bytes)) != cudaSuccess) {
            return nullptr;
        }
#endif /* USE_CUDA_TESTING */
        if (_CUDA_CHECK(cudaMemset(vi->ptr, 0, bytes)) != cudaSuccess) {
            return nullptr;
        }

        VectorInfo* raw = vi.release();
        vectors_[raw->ptr] = raw;

        RegisterWriters(raw->node, {raw->ptr});
        return raw->ptr;
    }

    // -------------------------------
    // 2. Free
    // -------------------------------
    bool Free(const void* ptr) {
        if (vectors_.empty()) return true; // NOTE: allows deferred Free invocations after graph building
        auto it = vectors_.find(ptr);
        if (it == vectors_.end()) return false;

        VectorInfo* vi = it->second;

        to_free_.push_back(vi->ptr);

        vectors_.erase(ptr);
        delete vi;
        return true;
    }

    size_t Size(const void *ptr) const {
        auto it = vectors_.find(ptr);
        if (it == vectors_.end()) return 0;

        VectorInfo* vi = it->second;
        return vi->bytes;
    }

private:
    // -------------------------------
    // CopyHelper
    // -------------------------------
    bool CopyHelper(const void* ptr, void* dst, void* src, size_t bytes, size_t start = 0) {
        if (dst == src || (ptr != dst && ptr != src)) {
            return false;
        }

        auto it = vectors_.find(ptr);
        if (it == vectors_.end()) return false;

        VectorInfo* vi = it->second;
        if (start + bytes > vi->bytes) return false;

        if (immediateGraphOp && !_graphTLS.recursingCapture) {
            if (_CUDA_CHECK(cudaMemcpy(dst, src, bytes, ptr == dst ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost)) != cudaSuccess) {
                return false;
            }
        }

        if (!noCaptureGraphOp && !_graphTLS.recursingImmediate) {
            std::vector<cudaGraphNode_t> deps;
            if (ptr == dst) {
                if (!Deps(deps, {}, {vi->ptr})) return false;
            } else {
                if (!Deps(deps, {vi->ptr}, {})) return false;
            }

            const char* srcOffset = reinterpret_cast<const char*>(src) + (ptr == src ? start : 0);
            char* dstOffset = reinterpret_cast<char*>(dst) + (ptr == dst ? start : 0);
            cudaMemcpyKind kind = ptr == dst ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;

            cudaGraphNode_t node;
            if (_CUDA_CHECK(cudaGraphAddMemcpyNode1D(&node, graph_, deps.data(), deps.size(), dstOffset, srcOffset, bytes, kind)) != cudaSuccess) {
                return false;
            }

            if (ptr == dst) {
                RegisterWriters(node, {vi->ptr});
            } else {
                RegisterReaders(node, {vi->ptr});            
            }
        }
        return true;
    }

public:
    // -------------------------------
    // 3. CopyToDevice
    // -------------------------------
    bool CopyToDevice(void* ptr, const void* hostSrc,
                      size_t bytes, size_t start = 0) {
        return CopyHelper(ptr, ptr, const_cast<void*>(hostSrc), bytes, start);
    }

    // -------------------------------
    // 4. CopyToHost
    // -------------------------------
    bool CopyToHost(void* hostDst, const void* ptr,
                    size_t bytes, size_t start = 0) {
        return CopyHelper(ptr, hostDst, const_cast<void*>(ptr), bytes, start);
    }

    // -------------------------------
    // 5. Compute
    // -------------------------------
public:
    bool Compute(const std::vector<const void*>& inputs,
                 const std::vector<void*>& outputs,
                 const std::function<bool(cudaStream_t)>& op) {
        std::vector<cudaGraphNode_t> deps;
        if (!Deps(deps, inputs, outputs)) {
            return false;
        }

        if (immediateGraphOp && !_graphTLS.recursingCapture) {
            struct GraphRecursing {
                bool saveRecursing_;
                GraphRecursing() : saveRecursing_(_graphTLS.recursingImmediate) { _graphTLS.recursingImmediate = true; }
                ~GraphRecursing() { _graphTLS.recursingImmediate = saveRecursing_; }
            } gr;
            if (!op(0)) {
                return false;
            }
        }

        if (!noCaptureGraphOp && !_graphTLS.recursingImmediate) {
            struct GraphRecursing {
                bool saveRecursing_;
                GraphRecursing() : saveRecursing_(_graphTLS.recursingCapture) { _graphTLS.recursingCapture = true; }
                ~GraphRecursing() { _graphTLS.recursingCapture = saveRecursing_; }
            } gr;

            StreamGuard sg;
            if (!sg.valid()) {
                return false;
            }

            if (_CUDA_CHECK(cudaStreamBeginCapture(sg.get(), streamCaptureMode)) != cudaSuccess) {
                return false;
            }

            if (!op(sg.get())) {
                return false;
            }

            GraphGuard gg;
            if (_CUDA_CHECK(cudaStreamEndCapture(sg.get(), &gg.get())) != cudaSuccess) {
                return false;
            }

            cudaGraphNode_t node;
            if (_CUDA_CHECK(cudaGraphAddChildGraphNode(&node, graph_, deps.data(), deps.size(), gg.get())) != cudaSuccess) {
                return false;
            }
            gg.release();

            RegisterReaders(node, inputs);
            RegisterWriters(node, outputs);
        }
        return true;
    }

    // -------------------------------
    // Build
    // -------------------------------
    bool Build() {
        if (!ExecuteCleanup()) {
            return false;
        }

        struct BuildCleanupGuard {
            BuildCleanupGuard(CudaVectorGraphImpl& impl) : impl_(impl) {}
            ~BuildCleanupGuard() { impl_.BuildCleanup(); }
            CudaVectorGraphImpl& impl_;
        } build_cleanup(*this);

        // NOTE - for debug printing
        // if (!noCaptureGraphOp) {
        //     size_t numNodes = 0;
        //     cudaGraphGetNodes(graph_, nullptr, &numNodes);
        //     std::cerr << "Graph nodes: " << numNodes << std::endl;
        //     if (!printGraphNodes(graph_)) {
        //         return false;
        //     }
        //     std::cerr << "Done printing nodes" << std::endl;
        //     dotPrintGraph(graph_);
        // }

        if (_CUDA_CHECK(cudaGraphInstantiate(&graphExec_, graph_, nullptr, nullptr, 0)) != cudaSuccess) {
            return false;
        }

        return true;
    }

    // -------------------------------
    // Execute
    // -------------------------------
    bool Execute(cudaStream_t stream = 0) {
        // NOTE - for debug printing
        // if (!immediateGraphOp) {
        //     cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start, stream);
        //     if (_CUDA_CHECK(cudaGraphLaunch(graphExec_, stream)) != cudaSuccess) {
        //         return false;
        //     }
        //     cudaEventRecord(stop, stream); cudaEventSynchronize(stop); float ms; cudaEventElapsedTime(&ms, start, stop);
        //     if (!noCaptureGraphOp) {
        //         std::cerr << "=== GRAPH LAUNCH ms: " << ms << std::endl;
        //     }
        // }

        if (_CUDA_CHECK(cudaStreamSynchronize(stream)) != cudaSuccess) {
            return false;
        }

        return true;
    }

private:
    struct VectorInfo {
        void* ptr = nullptr;
        size_t bytes = 0;
        cudaGraphNode_t node;
        std::vector<cudaGraphNode_t> activeReaders;
        cudaGraphNode_t lastWriter;
    };

    // RAII wrapper for cudaEvent_t
    struct EventGuard {
        cudaEvent_t e;
        cudaError_t err;
        EventGuard() : e(nullptr), err(cudaSuccess) {
            if (cudaSuccess != (err = _CUDA_CHECK(cudaEventCreate(&e)))) e = nullptr;
        }
        EventGuard(EventGuard&& eg) { e = eg.e; eg.e = nullptr; }
        ~EventGuard() { if (e) _CUDA_CHECK(cudaEventDestroy(e)); e = nullptr; }
        bool valid() const { return e != nullptr; }
        void release() { e = nullptr; }
        cudaEvent_t& get() { return e; }
        const cudaEvent_t& get() const { return e; }
    };

    // RAII wrapper for cudaStream_t
    struct StreamGuard {
        cudaStream_t s;
        StreamGuard() : s(nullptr) {
            if (_CUDA_CHECK(cudaStreamCreate(&s)) != cudaSuccess) s = nullptr;
        }
        StreamGuard(StreamGuard&& sg) { s = sg.s; sg.s = nullptr; }
        ~StreamGuard() { if (s) _CUDA_CHECK(cudaStreamDestroy(s)); s = nullptr; }
        bool valid() const { return s != nullptr; }
        void release() { s = nullptr; }
        cudaStream_t& get() { return s; }
        const cudaStream_t& get() const { return s; }
    };

    // RAII wrapper for cudaGraph_t
    struct GraphGuard {
        cudaGraph_t g;
        GraphGuard() : g(nullptr) {}
        GraphGuard(GraphGuard&& gg) { g = gg.g; gg.g = nullptr; }
        ~GraphGuard() { if (g) _CUDA_CHECK(cudaGraphDestroy(g)); g = nullptr; }
        bool valid() const { return g != nullptr; }
        void release() { g = nullptr; }
        cudaGraph_t& get() { return g; }
        const cudaGraph_t& get() const { return g; }
    };

    bool Cleanup(bool destruct = false) {
        bool result = true;
        result = result && BuildCleanup(destruct);
        result = result && ExecuteCleanup();
        result = result && MemoryCleanup();
        return result;
    }

    bool BuildCleanup(bool destruct = false) {
        for (auto& e : vectors_) delete e.second;
        vectors_.clear();
        if (graph_) {
            cudaError_t err = _CUDA_CHECK(cudaGraphDestroy(graph_));
            graph_ = nullptr;
            if (err != cudaSuccess) {
                return false;
            }
        }
        if (!destruct) {
            if (_CUDA_CHECK(cudaGraphCreate(&graph_, 0)) != cudaSuccess) {
                return false;
            }
        }
        return true;
    }

    bool ExecuteCleanup() {
        if (graphExec_) {
            cudaError_t err = _CUDA_CHECK(cudaGraphExecDestroy(graphExec_));
            graphExec_ = nullptr;
            if (err != cudaSuccess) {
                return false;
            }
        }
        return true;
    }

    bool MemoryCleanup() {
        bool result = true;
        while (!to_free_.empty()) {
            void* ptr = to_free_[to_free_.size() - 1];
            to_free_.pop_back();
            if (cudaFree(ptr) != cudaSuccess) {
                result = false;
            }
        }
        return result;
    }

    CudaVectorGraph::Params params_;
    cudaGraph_t graph_;
    cudaGraphExec_t graphExec_;
    std::unordered_map<const void*, VectorInfo*> vectors_;
    std::vector<void*> to_free_;
};

CudaVectorGraph::CudaVectorGraph()
    : impl_(new CudaVectorGraphImpl()) {}

CudaVectorGraph::~CudaVectorGraph() {
    delete impl_;
    impl_ = nullptr;
}

//uncomment for debugging
// #define _CUDAGRAPH_DEBUG_CALLS

class DebugCall {
public:
    DebugCall() {
#ifdef _CUDAGRAPH_DEBUG_CALLS
        std::cerr << "Enter : ";
#endif
    }
    ~DebugCall() {
#ifdef _CUDAGRAPH_DEBUG_CALLS
        std::cerr << "(Exit)" << std::endl;
#endif
    }

    // Generic template operator<<
    template <typename T>
    DebugCall& operator<<(const T& value) {
#ifdef _CUDAGRAPH_DEBUG_CALLS
        std::cerr << value;
#endif
        return *this;
    }

    // Support manipulators like std::endl
    DebugCall& operator<<(std::ostream& (*manip)(std::ostream&)) {
#ifdef _CUDAGRAPH_DEBUG_CALLS
        std::cerr << manip;
#endif
        return *this;
    }
};

const CudaVectorGraph::Params& CudaVectorGraph::GetParams() const {
    DebugCall debug_call; debug_call << "impl_->GetParams()" << std::endl;
    return impl_->GetParams();
}

bool CudaVectorGraph::Reset() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    DebugCall debug_call; debug_call << "impl_->Reset()" << std::endl;
    return impl_->Reset();
}

void* CudaVectorGraph::Malloc(size_t bytes) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    DebugCall debug_call; debug_call << "impl_->Malloc(" << bytes << ")" << std::endl;
    return impl_->Malloc(bytes);
}

bool CudaVectorGraph::Free(const void* vectorId) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    DebugCall debug_call; debug_call << "impl_->Free(" << vectorId << ")" << std::endl;
    return impl_->Free(vectorId);
}

size_t CudaVectorGraph::Size(const void* vectorId) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    DebugCall debug_call; debug_call << "impl_->Size(" << vectorId << ")" << std::endl;
    return impl_->Size(vectorId);
}

bool CudaVectorGraph::CopyToDevice(void* ptr, const void* hostSrc,
                                   size_t bytes, size_t start) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    DebugCall debug_call; debug_call << "impl_->CopyToDevice(" << ptr << "," << hostSrc << "," << bytes << "," << start << ")" << std::endl;
    return impl_->CopyToDevice(ptr, hostSrc, bytes, start);
}

bool CudaVectorGraph::CopyToHost(void* hostDst, const void* ptr,
                                 size_t bytes, size_t start) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    DebugCall debug_call; debug_call << "impl_->CopyToHost(" << hostDst << "," << ptr << "," << bytes << "," << start << ")" << std::endl;
    return impl_->CopyToHost(hostDst, ptr, bytes, start);
}

bool CudaVectorGraph::Compute(const std::vector<const void*>& inputs,
                              const std::vector<void*>& outputs,
                              const std::function<bool(cudaStream_t)>& op) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    DebugCall debug_call; debug_call << "impl_->Compute(" << inputs.size() << "," << outputs.size() << "," << "?" << ")" << std::endl;
    return impl_->Compute(inputs, outputs, op);
}

bool CudaVectorGraph::Build() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    DebugCall debug_call; debug_call << "impl_->Build()" << std::endl;
    return impl_->Build();
}

bool CudaVectorGraph::Execute(cudaStream_t stream) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    DebugCall debug_call; debug_call << "impl_->Execute()" << std::endl;
    return impl_->Execute(stream);
}

CudaVectorGraph* CudaVectorGraph::Get(DoContext* ctx) {
    DoContextImpl* ctximpl = reinterpret_cast<DoContextImpl*>(ctx);
    if (!ctximpl) return nullptr;
    std::shared_ptr<CudaVectorGraph>* p_graph = ctximpl->get<std::shared_ptr<CudaVectorGraph>>();
    if (!p_graph) {
        p_graph = ctximpl->put(std::make_shared<CudaVectorGraph>());
    };
    return p_graph->get();
}

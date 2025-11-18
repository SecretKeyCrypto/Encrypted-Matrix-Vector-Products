#include <cuda_runtime.h>
#include <memory>
#include "hdcommon.h"

template <typename CubCallable, typename Allocator>
class CubCall {
    using T = typename Allocator::value_type;
public:
    CubCall(CubCallable&& cub_fn, Allocator& allocator)
      : cub_fn(cub_fn),
        allocator(allocator)
    {
        deleter_ = [this](T *ptr) mutable { if (ptr) CubCall::deallocate(*this, ptr); };
        d_temp_storage_ = std::shared_ptr<T>(nullptr, deleter_);
        temp_storage_bytes_ = 0;
    }

    CubCall(const CubCall& cc)
      : cub_fn(std::move(cc.cub_fn)),
        allocator(cc.allocator),
        d_temp_storage_(cc.d_temp_storage_),
        temp_storage_bytes_(cc.temp_storage_bytes_)
    {
    }

    cudaError_t Prepare() {
        cudaError_t status = lastStatus = cub_fn(nullptr, temp_storage_bytes_, 0);
        if (status != cudaSuccess) return status;
        d_temp_storage_.reset(allocator.allocate(temp_storage_bytes_), deleter_);
        lastStatus = CubCall::memoryStatus(d_temp_storage_ != nullptr);
        return lastStatus;
    }

    // Invoke with a CUB callable and stream
    cudaError_t Call(cudaStream_t stream) {
        cudaError_t status = lastStatus = _CUDA_CHECK(cub_fn(d_temp_storage_.get(), temp_storage_bytes_, stream));
        return status;
    }

    T* GetStorage() const { return d_temp_storage_.get(); }
    size_t GetStorageBytes() const { return temp_storage_bytes_; }
    cudaError_t GetLastStatus() const { return lastStatus; }

private:
    static cudaError memoryStatus(bool success) {
        return success ? cudaSuccess : cudaErrorMemoryAllocation;
    }
    static void deallocate(CubCall& cubcall, T* ptr) {
        cubcall.lastStatus = memoryStatus(cubcall.allocator.deallocate(ptr));
    }

private:
    Allocator& allocator;
    CubCallable cub_fn;
    std::function<void(T* ptr)> deleter_;
    cudaError_t lastStatus;

    std::shared_ptr<T> d_temp_storage_;
    size_t temp_storage_bytes_;
};

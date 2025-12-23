#include "cudatestutils.h"
#include "plaintestutils.h"
#include "../dataobjects/cudacommon.h"
#include <cuda_runtime.h>

#include <vector>

bool CudaFieldVectorsAreEqual(
    const uint32_t* a, uint32_t ao, uint32_t as,
    const uint32_t* b, uint32_t bo, uint32_t bs,
    uint32_t length, uint32_t steps
) {
    std::vector<uint32_t> host_a(steps * length);
    std::vector<uint32_t> host_b(steps * length);

    uint32_t* ha = host_a.data();
    uint32_t* hb = host_b.data();
    a += ao;
    b += bo;
    for (uint32_t s = 0; s < steps; s++, a += as, b += bs, ha += length, hb += length) {
        if (_CUDA_CHECK(cudaMemcpy(ha, a, sizeof(uint32_t) * length, cudaMemcpyDeviceToHost)) != cudaSuccess) {
            return false;
        }
        if (_CUDA_CHECK(cudaMemcpy(hb, b, sizeof(uint32_t) * length, cudaMemcpyDeviceToHost)) != cudaSuccess) {
            return false;
        }
    }
    return PlainFieldVectorsAreEqual(host_a.data(), 0, length, host_b.data(), 0, length, length, steps);
}

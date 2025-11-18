#include "plaintestutils.h"
#include <vector>
#include <iostream>

bool PlainFieldVectorsAreEqual(
    const uint32_t* a, uint32_t ao, uint32_t as,
    const uint32_t* b, uint32_t bo, uint32_t bs,
    uint32_t length, uint32_t steps
) {
    std::vector<uint32_t> misses;
    a += ao;
    b += bo;
    for (uint32_t s = 0; s < steps; s++, a += as, b += bs) {
        const uint32_t* ai = a;
        const uint32_t* bi = b;
        for (uint32_t i = 0; i < length; i++, ai++, bi++) {
            if (*ai != *bi) {
                misses.push_back(i);
            }
        }
    }
    if (!misses.empty()) {
        std::cerr << "MISSES:";
        for (uint32_t miss : misses) {
            std::cerr << ' ' << miss;
        }
        std::cerr << std::endl;
    }
    return misses.empty();
}

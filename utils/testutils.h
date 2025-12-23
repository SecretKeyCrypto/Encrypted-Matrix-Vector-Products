#ifndef _TESTUTILS_H_
#define _TESTUTILS_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

bool FieldVectorsAreEqual(
    const uint32_t* a, uint32_t ao, uint32_t as,
    const uint32_t* b, uint32_t bo, uint32_t bs,
    uint32_t length, uint32_t steps
);

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* _TESTUTILS_H_ */
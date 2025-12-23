#ifndef _PLAINTESTUTILS_H_
#define _PLAINTESTUTILS_H_

#include <stdint.h>

bool PlainFieldVectorsAreEqual(
    const uint32_t* a, uint32_t ao, uint32_t as,
    const uint32_t* b, uint32_t bo, uint32_t bs,
    uint32_t length, uint32_t steps
);

#endif /* _PLAINTESTUTILS_H_ */
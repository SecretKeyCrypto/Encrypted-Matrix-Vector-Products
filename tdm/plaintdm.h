#ifndef _PLAINTDM_H_
#define _PLAINTDM_H_

#include <stdint.h>
#include <stdbool.h>
#include "../dataobjects/docontext.h"

#ifdef __cplusplus
extern "C" {
#endif

bool PlainPermutedExtentsAssign(
    DoContext* ctx,
    uint32_t* r,
    uint32_t ro,
    uint32_t rfs,
    uint32_t rps,
    const uint32_t* s,
    uint32_t so,
    uint32_t ss,
    uint32_t sc,
    uint64_t extent,
    const uint32_t* perm,
    uint32_t po,
    uint64_t length);

bool PlainCircularCopy(DoContext* ctx, uint32_t* r, const uint32_t* v, uint64_t length);

#ifdef __cplusplus
} // exterm "C"
#endif

#endif /* _PLAINTDM_H_ */
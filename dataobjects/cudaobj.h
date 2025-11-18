#pragma once

#include <stdlib.h>
#include <stdbool.h>
#include "docontext.h"

#ifdef __cplusplus
extern "C" {
#endif

bool cuda_doreset(DoContext* doctx);

void* cuda_doalloc(DoContext* doctx, size_t size);

bool cuda_dofree(DoContext* doctx, void* ptr);

bool cuda_dosync(DoContext* doctx);

void* cuda_alloc(size_t size);

void cuda_free(void* ptr);

void cuda_sync();

#ifdef __cplusplus
} // extern "C"
#endif

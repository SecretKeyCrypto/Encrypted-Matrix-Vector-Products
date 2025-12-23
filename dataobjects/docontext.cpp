
#include "docontext.h"
#include "docontextimpl.h"
#include <iostream>

// #define USE_COMMON_DEBUG

#ifdef __cplusplus
extern "C" {
#endif

DoContext* NewDoContext() {
    DoContext* ctx = reinterpret_cast<DoContext*>(new DoContextImpl());
#ifdef USE_COMMON_DEBUG
    std::cerr << "global NewDoContext " << ctx << std::endl;
#endif /* USE_COMMON_DEBUG */
    return ctx;
}

void FreeDoContext(DoContext* ctx) {
#ifdef USE_COMMON_DEBUG
    std::cerr << "global FreeDoContext" << ctx << std::endl;
#endif
    delete reinterpret_cast<DoContextImpl*>(ctx);
}

#ifdef __cplusplus
} // extern "C"
#endif

#ifndef _DOCONTEXT_H_
#define _DOCONTEXT_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DoContext DoContext;

DoContext* NewDoContext();
void FreeDoContext(DoContext*);

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* _DOCONTEXT_H_ */

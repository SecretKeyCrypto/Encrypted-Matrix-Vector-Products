#ifndef _EMVP_DATAOBJ_H_
#define _EMVP_DATAOBJ_H_

// #include <inttypes.h>
// #include <memory>

const bool USE_FAST_CODE = true;

// #define EMVP_ALIGNMENT 16
// #define EMVP_ALIGNAS alignas(EMVP_ALIGNMENT)

// namespace aligned {
//     template <class T>
//     using type EMVP_ALIGNAS = T;// __attribute__((vector_size(16))) = T;

//     template <class T, int N>
//     using array EMVP_ALIGNAS = type<T>[N];

//     template <class T>
//     using ptr = type<T> *;
// }

// template <class T>
// inline T* emvp_align(T* a) {
//     void* a0 = a;
//     size_t s = sizeof(*a);
//     return (uint32_t *)std::align(EMVP_ALIGNMENT, s, a0, s);
// }

// template <class T>
// struct Align {
//     EMVP_ALIGNAS T v[1];
// };

// template <class T>
// inline Align<T>* emvp_aligned(T* a) {
//     return (Align<T> *)a;
// }

// template <class T>
// inline const Align<T>* emvp_aligned(const T* a) {
//     return (const Align<T> *)a;
// }

// #define EMVP_ALIGN(a) (emvp_aligned(a)->v)

#endif /* _EMVP_DATAOBJ_H_ */
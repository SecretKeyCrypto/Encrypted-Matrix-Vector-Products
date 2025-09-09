package tdm

/*
#cgo CXXFLAGS: -std=c++17 -Ofast -fomit-frame-pointer -march=native -mtune=native -I.
#cgo CFLAGS: -I../TDM
#cgo LDFLAGS: -L../TDM -L/opt/homebrew/lib -lcrypto -lNTT -lntl -lgmp -lstdc++
#include "NTT.h"
*/
import "C"
import (
	"runtime"
	"unsafe"
)

func NTT_Convolution(dataA, dataB, result []uint32, degree, root, q uint32) {
	C.ntt_convolution((*C.uint32_t)(unsafe.Pointer(&dataA[0])),
		(*C.uint32_t)(unsafe.Pointer(&dataB[0])),
		(*C.uint32_t)(unsafe.Pointer(&result[0])),
		C.size_t(degree), C.uint32_t(root), C.uint32_t(q))
	runtime.KeepAlive((dataA))
	runtime.KeepAlive((dataB))
	runtime.KeepAlive((result))
}

func NthRootOfUnity(q, n uint32) uint32 {
	return uint32(C.NthRootOfUnity(C.uint32_t(q), C.uint32_t(n)))
}

func NTT(coeff []uint32, n, root, q uint32) {
	C.ntt((*C.uint32_t)(unsafe.Pointer(&coeff[0])),
		C.size_t(n), C.uint32_t(root), C.uint32_t(q))
	runtime.KeepAlive(coeff)
}

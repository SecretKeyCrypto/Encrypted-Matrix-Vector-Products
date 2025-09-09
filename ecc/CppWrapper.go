package ecc

/*
#cgo CXXFLAGS: -std=c++17 -Ofast -fomit-frame-pointer -march=native -mtune=native -I.
#cgo CFLAGS: -I.
#cgo LDFLAGS: -L. -lReedSolomon -L/opt/homebrew/lib -lNTT -lntl -lgmp -lstdc++
#include "ReedSolomon.h"
*/
import "C"
import "unsafe"

func GenerateSystematicRSMatrix(ECCLength, M_1, p uint32, alphas, output []uint32) {
	C.GenerateSystematicRSMatrix_uint32(
		C.uint32_t(ECCLength), C.uint32_t(M_1), C.uint32_t(p),
		(*C.uint32_t)(unsafe.Pointer(&alphas[0])),
		(*C.uint32_t)(unsafe.Pointer(&output[0])))
}

func LagrangeInterpEval(x, y []uint32, k, index uint32, q uint32) uint32 {
	return uint32(C.LagrangeInterpEval(
		(*C.uint32_t)(unsafe.Pointer(&x[0])),
		(*C.uint32_t)(unsafe.Pointer(&y[0])),
		C.uint32_t(k), C.uint32_t(index), C.uint32_t(q)))
}

package mvp

/*
#cgo CXXFLAGS: -std=c++17 -Ofast -fomit-frame-pointer -march=native -mtune=native -I.
#cgo CFLAGS: -I.
#cgo LDFLAGS: -L. -L../tdm -lMVP -L/opt/homebrew/lib -lNTT -lntl -lgmp -lstdc++
#include "mvp.h"
*/
import "C"
import (
	"runtime"
	"unsafe"
)

func BlockMatVecProduct(mat, vec, out []uint32, row, col, numBlock, p uint32) {
	C.BlockMatVecProduct(
		(*C.uint32_t)(unsafe.Pointer(&mat[0])),
		(*C.uint32_t)(unsafe.Pointer(&vec[0])),
		(*C.uint32_t)(unsafe.Pointer(&out[0])),
		C.uint32_t(row), C.uint32_t(col), C.uint32_t(numBlock), C.uint32_t(p),
	)
	runtime.KeepAlive(mat)
	runtime.KeepAlive(vec)
	runtime.KeepAlive(out)
}

func MatVecProduct(mat, vec, out []uint32, row, col, p uint32) {
	C.MatVecProduct(
		(*C.uint32_t)(unsafe.Pointer(&mat[0])),
		(*C.uint32_t)(unsafe.Pointer(&vec[0])),
		(*C.uint32_t)(unsafe.Pointer(&out[0])),
		C.uint32_t(row), C.uint32_t(col), C.uint32_t(p),
	)
	runtime.KeepAlive(mat)
	runtime.KeepAlive(vec)
	runtime.KeepAlive(out)
}

func BlockVecMatProduct(mat, vec, out []uint32, row, col, numBlock, p uint32) {
	C.BlockVecMatProduct(
		(*C.uint32_t)(unsafe.Pointer(&mat[0])),
		(*C.uint32_t)(unsafe.Pointer(&vec[0])),
		(*C.uint32_t)(unsafe.Pointer(&out[0])),
		C.uint32_t(row), C.uint32_t(col), C.uint32_t(numBlock), C.uint32_t(p),
	)
	runtime.KeepAlive(mat)
	runtime.KeepAlive(vec)
	runtime.KeepAlive(out)
}

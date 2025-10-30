package dataobjects

/*
#cgo CXXFLAGS: -std=c++17 -Ofast -fomit-frame-pointer -march=native -mtune=native -I.
#cgo CFLAGS: -I.
#cgo LDFLAGS: -L. -L/opt/homebrew/lib -ldataobjects -lNTT -lntl -lgmp -lstdc++ -lcudart
#include "fields.h"
*/
import "C"
import (
	"runtime"
	"unsafe"
)

func FieldAddVectors(r []uint32, ro uint64, a []uint32, ao uint64, b []uint32, bo uint64, length uint64, p uint32) int {
	result := C.FieldAddVectors(
		(*C.uint32_t)(unsafe.Pointer(&r[0])), C.uint64_t(ro),
		(*C.uint32_t)(unsafe.Pointer(&a[0])), C.uint64_t(ao),
		(*C.uint32_t)(unsafe.Pointer(&b[0])), C.uint64_t(bo),
		C.uint64_t(length), C.uint32_t(p),
	)
	runtime.KeepAlive(r)
	runtime.KeepAlive(a)
	runtime.KeepAlive(b)
	return int(result)
}

func FieldMulVector(r []uint32, ro uint64, a []uint32, ao uint64, b uint32, length uint64, p uint32) int {
	result := C.FieldMulVector(
		(*C.uint32_t)(unsafe.Pointer(&r[0])), C.uint64_t(ro),
		(*C.uint32_t)(unsafe.Pointer(&a[0])), C.uint64_t(ao),
		C.uint32_t(b),
		C.uint64_t(length), C.uint32_t(p),
	)
	runtime.KeepAlive(r)
	runtime.KeepAlive(a)
	return int(result)
}

func FieldSubVectors(r []uint32, ro uint64, a []uint32, ao uint64, b []uint32, bo uint64, length uint64, p uint32) int {
	result := C.FieldSubVectors(
		(*C.uint32_t)(unsafe.Pointer(&r[0])), C.uint64_t(ro),
		(*C.uint32_t)(unsafe.Pointer(&a[0])), C.uint64_t(ao),
		(*C.uint32_t)(unsafe.Pointer(&b[0])), C.uint64_t(bo),
		C.uint64_t(length), C.uint32_t(p),
	)
	runtime.KeepAlive(r)
	runtime.KeepAlive(a)
	runtime.KeepAlive(b)
	return int(result)
}

func FieldNegVector(r []uint32, ro uint64, length uint64, p uint32) int {
	result := C.FieldNegVector(
		(*C.uint32_t)(unsafe.Pointer(&r[0])), C.uint64_t(ro),
		C.uint64_t(length), C.uint32_t(p),
	)
	runtime.KeepAlive(r)
	return int(result)
}

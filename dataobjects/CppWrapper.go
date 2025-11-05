package dataobjects

/*
#cgo CXXFLAGS: -std=c++17 -Ofast -fomit-frame-pointer -march=native -mtune=native -I.
#cgo CFLAGS: -I.
#cgo LDFLAGS: -L. -L/opt/homebrew/lib -ldataobjects -lNTT -lstdc++ -lcudart
#include "fields.h"
#include "matrices.h"
*/
import "C"
import (
	"runtime"
	"unsafe"
)

func FieldRangeVector(r []uint32, ro uint64, start uint32, length uint64) int {
	result := C.FieldRangeVector(
		(*C.uint32_t)(unsafe.Pointer(&r[0])), C.uint64_t(ro),
		C.uint32_t(start), C.uint64_t(length),
	)
	runtime.KeepAlive(r)
	return int(result)
}

func FieldCopyVector(r []uint32, ro uint64, a []uint32, ao uint64, length uint64) int {
	result := C.FieldCopyVector(
		(*C.uint32_t)(unsafe.Pointer(&r[0])), C.uint64_t(ro),
		(*C.uint32_t)(unsafe.Pointer(&a[0])), C.uint64_t(ao),
		C.uint64_t(length),
	)
	runtime.KeepAlive(r)
	runtime.KeepAlive(a)
	return int(result)
}

func FieldSetVector(r []uint32, ro uint64, length uint64, v uint32) int {
	result := C.FieldSetVector(
		(*C.uint32_t)(unsafe.Pointer(&r[0])), C.uint64_t(ro),
		C.uint64_t(length), C.uint32_t(v),
	)
	runtime.KeepAlive(r)
	return int(result)
}

func FieldAddToVector(r []uint32, ro uint64, v uint32, length uint64) int {
	result := C.FieldAddToVector(
		(*C.uint32_t)(unsafe.Pointer(&r[0])), C.uint64_t(ro),
		C.uint32_t(v),
		C.uint64_t(length),
	)
	runtime.KeepAlive(r)
	return int(result)
}

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

func FieldAddVectorIfNonZero(t []bool, t_index uint64, r []uint32, ro uint64, e []uint32, eo uint64, length uint64, p uint32) int {
	result := C.FieldAddVectorIfNonZero(
		(*C.bool)(unsafe.Pointer(&t[0])), C.uint64_t(t_index),
		(*C.uint32_t)(unsafe.Pointer(&r[0])), C.uint64_t(ro),
		(*C.uint32_t)(unsafe.Pointer(&e[0])), C.uint64_t(eo),
		C.uint64_t(length), C.uint32_t(p),
	)
	runtime.KeepAlive(t)
	runtime.KeepAlive(r)
	runtime.KeepAlive(e)
	return int(result)
}

func MatrixTraspose(r []uint32, a []uint32, M, N uint32) int {
	result := C.MatrixTranspose(
		(*C.uint32_t)(unsafe.Pointer(&r[0])),
		(*C.uint32_t)(unsafe.Pointer(&a[0])),
		C.uint32_t(M),
		C.uint32_t(N),
	)
	runtime.KeepAlive(r)
	runtime.KeepAlive(a)
	return int(result)
}

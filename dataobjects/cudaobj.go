//go:build cuda
// +build cuda

package dataobjects

/*
#include "cudaobj.h"
*/
import "C"
import (
	"reflect"
	"unsafe"
)

const USE_FAST_CODE_WITH_CUDA = true

var total_alloc = uint64(0)

func DoAlignedReset(doctx *DoContext) bool {
	c_result := C.cuda_doreset((*C.DoContext)(doctx.Doctx))
	return CheckResult(bool(c_result))
}

func DoAlignedMake[T any](doctx *DoContext, length uint64) []T {
	size := length * uint64(reflect.TypeOf((*T)(nil)).Elem().Size())
	ptr := C.cuda_doalloc((*C.DoContext)(doctx.Doctx), C.size_t(size))
	if ptr == nil {
		return nil
	}
	total_alloc += size
	return unsafe.Slice((*T)(ptr), length)
}

func DoAligned1DFree[T any](doctx *DoContext, array []T) bool {
	result := true
	if array != nil && len(array) > 0 {
		if !C.cuda_dofree((*C.DoContext)(doctx.Doctx), unsafe.Pointer(&array[0])) {
			result = false
		}
	}
	return CheckResult(result)
}

func DoAlignedSynchronize(doctx *DoContext) bool {
	c_result := C.cuda_dosync((*C.DoContext)(doctx.Doctx))
	return CheckResult(bool(c_result))
}

// Allocate returns a Go slice backed by CUDA-managed memory.
func AlignedMake[T any](length uint64) []T {
	size := length * uint64(reflect.TypeOf((*T)(nil)).Elem().Size())
	ptr := C.cuda_alloc(C.size_t(size))
	return unsafe.Slice((*T)(ptr), length)
}

// Free releases CUDA-managed memory from a slice.
func Aligned1DFree[T any](array []T) []T {
	if array != nil && len(array) > 0 {
		C.cuda_free(unsafe.Pointer(&array[0]))
	}
	return nil
}

func Aligned2DFree[T any](array [][]T) [][]T {
	if array != nil {
		for i := 0; i < len(array); i++ {
			array[i] = Aligned1DFree(array[i])
		}
	}
	return nil
}

func (v Array[T]) Free() Array[T] {
	return Aligned1DFree(v)
}

func (v Array2[T]) Free() Array2[T] {
	return Aligned2DFree(v)
}

func (v Vector) Free() Vector {
	Array[uint32](v).Free()
	return nil
}

func (v Vector2) Free() Vector2 {
	Array2[uint32](v).Free()
	return nil
}

func AlignedSynchronize() {
	C.cuda_sync()
}

//go:build !cuda
// +build !cuda

package dataobjects

import (
	"reflect"
	"unsafe"
)

const ALIGNMENT uint64 = 64 // aligns up to AVX512

// AlignedMake creates a slice of type T with a specified length and default alignment.
func AlignedMake[T Primitive](length uint64) []T {
	if length == 0 {
		return make([]T, length)
	}

	size := length * uint64(reflect.TypeOf((*T)(nil)).Elem().Size())
	bytearr := make([]byte, size+ALIGNMENT)
	addr := uint64(uintptr(unsafe.Pointer(&bytearr[0])))
	addrmod := addr % ALIGNMENT
	if addrmod > 0 {
		bytearr = bytearr[ALIGNMENT-addrmod:]
	}
	return unsafe.Slice((*T)(unsafe.Pointer(&bytearr[0])), length)
}

func Aligned1DFree[T Primitive](array []T) []T {
	// NO-OP - the caller is expected to drop the array
	return nil
}

func Aligned2DFree[T Primitive](array [][]T) [][]T {
	// NO-OP - the caller is expected to drop the array
	return nil
}

func AlignedSynchronize() {
	// NO_OP - the caller can immediately directly access data in aligned arrays
}

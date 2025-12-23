//go:build !cuda
// +build !cuda

package dataobjects

import (
	"reflect"
	"unsafe"
)

const USE_FAST_CODE_WITH_CUDA = false

const ALIGNMENT uint64 = 64 // aligns up to AVX512

func DoAlignedReset(doctx *DoContext) bool {
	return true
}

func DoAlignedMake[T Primitive](doctx *DoContext, length uint64) Array[T] {
	return AlignedMake[T](length)
}

func DoAligned1DFree[T Primitive](doctx *DoContext, array []T) bool {
	Aligned1DFree(array)
	return true
}

func DoAlignedSynchronize(doctx *DoContext) bool {
	AlignedSynchronize()
	return true
}

// AlignedMake creates a slice of type T with a specified length and default alignment.
func AlignedMake[T Primitive](length uint64) Array[T] {
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

func MakeVector(length uint64) Vector {
	return Vector(AlignedMake[uint32](length))
}

func Aligned1DFree[T Primitive](array []T) []T {
	// NO-OP - the caller is expected to drop the array
	return nil
}

func Aligned2DFree[T Primitive](array [][]T) [][]T {
	// NO-OP - the caller is expected to drop the array
	return nil
}

func (v Array[T]) Free() Array[T] {
	// NO-OP - the caller is expected to drop the array
	return nil
}

func (v Array2[T]) Free() Array2[T] {
	// NO-OP - the caller is expected to drop the array
	return nil
}

func (v Vector) Free() Vector {
	return nil
}

func (v Vector2) Free() Vector2 {
	// NO-OP - the caller is expected to drop the array
	return nil
}

func AlignedSynchronize() {
	// NO_OP - the caller can immediately directly access data in aligned arrays
}

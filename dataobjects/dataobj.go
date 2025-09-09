package dataobjects

import (
	"reflect"
	"unsafe"
)

const USE_FAST_CODE bool = true
const ALIGNMENT uint64 = 64 // aligns up to AVX512

// AlignedMake creates a slice of type T with a specified length and default alignment.
func AlignedMake[T any](length uint64) []T {
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

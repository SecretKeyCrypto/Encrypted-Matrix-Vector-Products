//go:build cuda
// +build cuda

package dataobjects

/*
#cgo LDFLAGS: -L/usr/lib/x86_64-linux-gnu -lcudart -lcudadevrt
#include <cuda_runtime.h>
#include <stdlib.h>

void* cuda_alloc(size_t size) {
    void* ptr;
    cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);
	cudaMemset(ptr, 0, size);
    return ptr;
}

void cuda_free(void* ptr) {
    cudaFree(ptr);
}

void cuda_sync() {
	cudaDeviceSynchronize();
}
*/
import "C"
import (
	"reflect"
	"unsafe"
)

const USE_FAST_CODE_WITH_CUDA = true

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

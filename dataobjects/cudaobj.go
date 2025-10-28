//go:build cuda
// +build cuda

package dataobjects

/*
#cgo LDFLAGS: -L/usr/lib/x86_64-linux-gnu -lcudart
#include <cuda_runtime.h>
#include <stdlib.h>

void* cuda_alloc(size_t size) {
    void* ptr;
    cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);
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

func AlignedSynchronize() {
	C.cuda_sync()
}

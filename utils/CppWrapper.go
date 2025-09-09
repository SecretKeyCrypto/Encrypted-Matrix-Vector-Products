package utils

/*
#cgo CXXFLAGS: -std=c++17 -Ofast -fomit-frame-pointer -march=native -mtune=native -I.
#cgo CFLAGS: -I.
#cgo LDFLAGS: -L. -L/opt/homebrew/lib -lNTT -lntl -lgmp -lstdc++
#include "rnd_api.h"
*/
import "C"
import (
	"runtime"
	"unsafe"
)

func _data_arg(data []uint32, length uint32) unsafe.Pointer {
	if data == nil || length == 0 {
		return unsafe.Pointer(nil)
	} else {
		return unsafe.Pointer(&data[0])
	}
}

func randomize_vector(data []uint32, length uint32) {
	C.randomize_vector(
		(*C.uint32_t)(_data_arg(data, length)),
		C.uint32_t(length),
	)
	runtime.KeepAlive(data)
}

func randomize_vector_with_seed(data []uint32, length uint32, seed int64) {
	C.randomize_vector_with_seed(
		(*C.uint32_t)(_data_arg(data, length)),
		C.uint32_t(length),
		C.int64_t(seed),
	)
	runtime.KeepAlive(data)
}

func randomize_vector_with_modulus(data []uint32, length uint32, modulus uint32) {
	C.randomize_vector_with_modulus(
		(*C.uint32_t)(_data_arg(data, length)),
		C.uint32_t(length),
		C.uint32_t(modulus),
	)
	runtime.KeepAlive(data)
}

func randomize_vector_with_modulus_and_seed(data []uint32, length uint32, modulus uint32, seed int64) {
	C.randomize_vector_with_modulus_and_seed(
		(*C.uint32_t)(_data_arg(data, length)),
		C.uint32_t(length),
		C.uint32_t(modulus),
		C.int64_t(seed),
	)
	runtime.KeepAlive(data)
}

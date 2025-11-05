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

func _data_arg(data []uint32, M, N uint32) unsafe.Pointer {
	if data == nil || (M == 0 && N == 0) {
		return unsafe.Pointer(nil)
	} else {
		return unsafe.Pointer(&data[0])
	}
}

func randomize_vector(data []uint32, M, N uint32, transpose bool) {
	C.randomize_vector(
		(*C.uint32_t)(_data_arg(data, M, N)),
		C.uint32_t(M), C.uint32_t(N), C.bool(transpose),
	)
	runtime.KeepAlive(data)
}

func randomize_vector_with_seed(data []uint32, M, N uint32, transpose bool, seed int64) {
	C.randomize_vector_with_seed(
		(*C.uint32_t)(_data_arg(data, M, N)),
		C.uint32_t(M), C.uint32_t(N), C.bool(transpose),
		C.int64_t(seed),
	)
	runtime.KeepAlive(data)
}

func randomize_vector_with_modulus(data []uint32, M, N uint32, transpose bool, modulus uint32) {
	C.randomize_vector_with_modulus(
		(*C.uint32_t)(_data_arg(data, M, N)),
		C.uint32_t(M), C.uint32_t(N), C.bool(transpose),
		C.uint32_t(modulus),
	)
	runtime.KeepAlive(data)
}

func randomize_vector_with_modulus_and_seed(data []uint32, M, N uint32, transpose bool, modulus uint32, seed int64) {
	C.randomize_vector_with_modulus_and_seed(
		(*C.uint32_t)(_data_arg(data, M, N)),
		C.uint32_t(M), C.uint32_t(N), C.bool(transpose),
		C.uint32_t(modulus),
		C.int64_t(seed),
	)
	runtime.KeepAlive(data)
}

func lpn_noise_vector(r []uint32, ro uint64, length uint64, epsi float64, p uint32) {
	C.lpn_noise_vector(
		(*C.uint32_t)(unsafe.Pointer(&r[ro])),
		C.uint64_t(length), C.double(epsi), C.uint32_t(p),
	)
	runtime.KeepAlive(r)
}

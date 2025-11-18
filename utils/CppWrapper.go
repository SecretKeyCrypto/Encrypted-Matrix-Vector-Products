package utils

/*
#cgo CXXFLAGS: -std=c++17 -Ofast -fomit-frame-pointer -march=native -mtune=native -I.
#cgo CFLAGS: -I.
#cgo LDFLAGS: -L. -L/opt/homebrew/lib -lutils -lNTT -lstdc++
#include "rnd_api.h"
#include "testutils.h"
*/
import "C"
import (
	"RandomLinearCodePIR/dataobjects"
	"unsafe"
)

func _data_arg(data []uint32, M, N uint32) unsafe.Pointer {
	if data == nil || (M == 0 && N == 0) {
		return unsafe.Pointer(nil)
	} else {
		return unsafe.Pointer(&data[0])
	}
}

func randomize_vector(doctx *dataobjects.DoContext, data []uint32, M, N uint32, transpose, circulant bool) bool {
	dataobjects.KeepAlive("randomize_vector", nil)
	g_result := dataobjects.GetDoWorker().Run(func() interface{} {
		c_result := C.randomize_vector(
			(*C.DoContext)(doctx.Doctx),
			(*C.uint32_t)(_data_arg(data, M, N)),
			C.uint32_t(M), C.uint32_t(N), C.bool(transpose), C.bool(circulant),
		)
		return bool(c_result)
	}).(bool)
	dataobjects.KeepAlive("data", data)
	return dataobjects.CheckResult(g_result)
}

func randomize_vector_with_seed(doctx *dataobjects.DoContext, data []uint32, M, N uint32, transpose, circulant bool, seed, offset int64) bool {
	dataobjects.KeepAlive("randomize_vector_with_seed", nil)
	g_result := dataobjects.GetDoWorker().Run(func() interface{} {
		c_result := C.randomize_vector_with_seed(
			(*C.DoContext)(doctx.Doctx),
			(*C.uint32_t)(_data_arg(data, M, N)),
			C.uint32_t(M), C.uint32_t(N), C.bool(transpose), C.bool(circulant),
			C.int64_t(seed), C.int64_t(offset),
		)
		return bool(c_result)
	}).(bool)
	dataobjects.KeepAlive("data", data)
	return dataobjects.CheckResult(g_result)
}

func randomize_vector_with_modulus(doctx *dataobjects.DoContext, data []uint32, M, N uint32, transpose, circulant bool, modulus uint32) bool {
	dataobjects.KeepAlive("randomize_vector_with_modulus", nil)
	g_result := dataobjects.GetDoWorker().Run(func() interface{} {
		c_result := C.randomize_vector_with_modulus(
			(*C.DoContext)(doctx.Doctx),
			(*C.uint32_t)(_data_arg(data, M, N)),
			C.uint32_t(M), C.uint32_t(N), C.bool(transpose), C.bool(circulant),
			C.uint32_t(modulus),
		)
		return bool(c_result)
	}).(bool)
	dataobjects.KeepAlive("data", data)
	return dataobjects.CheckResult(g_result)
}

func randomize_vector_with_modulus_and_seed(doctx *dataobjects.DoContext, data []uint32, M, N uint32, transpose, circulant bool, modulus uint32, seed, offset int64) bool {
	dataobjects.KeepAlive("randomize_vector_with_modulus_and_seed", nil)
	g_result := dataobjects.GetDoWorker().Run(func() interface{} {
		c_result := C.randomize_vector_with_modulus_and_seed(
			(*C.DoContext)(doctx.Doctx),
			(*C.uint32_t)(_data_arg(data, M, N)),
			C.uint32_t(M), C.uint32_t(N), C.bool(transpose), C.bool(circulant),
			C.uint32_t(modulus),
			C.int64_t(seed), C.int64_t(offset),
		)
		return bool(c_result)
	}).(bool)
	dataobjects.KeepAlive("data", data)
	return dataobjects.CheckResult(g_result)
}

func lpn_noise_vector(doctx *dataobjects.DoContext, r []uint32, ro uint64, length uint64, epsi float64, p uint32, seed, offset int64) bool {
	dataobjects.KeepAlive("lpn_noise_vector", nil)
	g_result := dataobjects.GetDoWorker().Run(func() interface{} {
		c_result := C.lpn_noise_vector(
			(*C.DoContext)(doctx.Doctx),
			(*C.uint32_t)(unsafe.Pointer(&r[0])), C.uint64_t(ro),
			C.uint64_t(length), C.double(epsi), C.uint32_t(p),
			C.int64_t(seed), C.int64_t(offset),
		)
		return bool(c_result)
	}).(bool)
	dataobjects.KeepAlive("r", r)
	return dataobjects.CheckResult(g_result)
}

func random_permutation(doctx *dataobjects.DoContext, perm []uint32, n uint32, seed, offset int64) bool {
	dataobjects.KeepAlive("random_permutation", nil)
	g_result := dataobjects.GetDoWorker().Run(func() interface{} {
		c_result := C.random_permutation(
			(*C.DoContext)(doctx.Doctx),
			(*C.uint32_t)(unsafe.Pointer(&perm[0])),
			C.uint32_t(n), C.int64_t(seed), C.int64_t(offset),
		)
		return bool(c_result)
	}).(bool)
	dataobjects.KeepAlive("perm", perm)
	return dataobjects.CheckResult(g_result)
}

func FieldVectorsAreEqual(a []uint32, ao, as uint32, b []uint32, bo, bs, length, steps uint32) bool {
	dataobjects.KeepAlive("FieldVectorsAreEqual", nil)
	g_result := dataobjects.GetDoWorker().Run(func() interface{} {
		c_result := C.FieldVectorsAreEqual(
			(*C.uint32_t)(unsafe.Pointer(&a[0])), C.uint32_t(ao), C.uint32_t(as),
			(*C.uint32_t)(unsafe.Pointer(&b[0])), C.uint32_t(bo), C.uint32_t(bs),
			C.uint32_t(length), C.uint32_t(steps),
		)
		return bool(c_result)
	}).(bool)
	dataobjects.KeepAlive("a", a)
	dataobjects.KeepAlive("b", b)
	return g_result
}

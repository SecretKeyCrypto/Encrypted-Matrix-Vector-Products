package tdm

/*
#cgo CXXFLAGS: -std=c++17 -Ofast -fomit-frame-pointer -march=native -mtune=native -I.
#cgo CFLAGS: -I../TDM
#cgo LDFLAGS: -L../TDM -L/opt/homebrew/lib -lcrypto -lNTT -lstdc++
#include "NTT.h"
#include "tdm.h"
*/
import "C"
import (
	"RandomLinearCodePIR/dataobjects"
	"unsafe"
)

func _data_arg(data []uint32) unsafe.Pointer {
	if data == nil {
		return unsafe.Pointer(nil)
	} else {
		return unsafe.Pointer(&data[0])
	}
}

func NTT_Convolution(doctx *dataobjects.DoContext, dataA []uint32, ao, as uint32, dataB []uint32, bo, bs uint32, result []uint32, ro, rs uint32, degree, steps, root, q uint32) bool {
	dataobjects.KeepAlive("NTT_Convolution", nil)
	g_result := dataobjects.GetDoWorker().Run(func() interface{} {
		c_result := C.ntt_convolution(
			(*C.DoContext)(doctx.Doctx),
			(*C.uint32_t)(unsafe.Pointer(&dataA[0])), C.uint32_t(ao), C.uint32_t(as),
			(*C.uint32_t)(unsafe.Pointer(&dataB[0])), C.uint32_t(bo), C.uint32_t(bs),
			(*C.uint32_t)(unsafe.Pointer(&result[0])), C.uint32_t(ro), C.uint32_t(rs),
			C.uint32_t(degree), C.uint32_t(steps),
			C.uint32_t(root), C.uint32_t(q))
		return bool(c_result)
	}).(bool)
	dataobjects.KeepAlive("dataA", dataA)
	dataobjects.KeepAlive("dataB", dataB)
	dataobjects.KeepAlive("result", result)
	return dataobjects.CheckResult(g_result)
}

func NthRootOfUnity(q, n uint32) uint32 {
	g_result := dataobjects.GetDoWorker().Run(func() interface{} {
		c_result := uint32(C.NthRootOfUnity(C.uint32_t(q), C.uint32_t(n)))
		return uint32(c_result)
	}).(uint32)
	return g_result
}

func NTT(doctx *dataobjects.DoContext, coeff []uint32, co uint32, n, stride, steps, root, q uint32) bool {
	dataobjects.KeepAlive("NTT", nil)
	g_result := dataobjects.GetDoWorker().Run(func() interface{} {
		c_result := C.ntt(
			(*C.DoContext)(doctx.Doctx),
			(*C.uint32_t)(unsafe.Pointer(&coeff[0])), C.uint32_t(co),
			C.uint32_t(n), C.uint32_t(stride), C.uint32_t(steps),
			C.uint32_t(root), C.uint32_t(q))
		return bool(c_result)
	}).(bool)
	dataobjects.KeepAlive("coeff", coeff)
	return dataobjects.CheckResult(g_result)
}

func CircularCopy(doctx *dataobjects.DoContext, r, v []uint32, length uint64) bool {
	dataobjects.KeepAlive("CircularCopy", nil)
	g_result := dataobjects.GetDoWorker().Run(func() interface{} {
		c_result := C.CircularCopy(
			(*C.DoContext)(doctx.Doctx),
			(*C.uint32_t)(unsafe.Pointer(&r[0])),
			(*C.uint32_t)(unsafe.Pointer(&v[0])),
			C.uint64_t(length))
		return bool(c_result)
	}).(bool)
	dataobjects.KeepAlive("r", r)
	dataobjects.KeepAlive("v", v)
	return dataobjects.CheckResult(g_result)
}

func PermutedExtentsAssign(doctx *dataobjects.DoContext, r []uint32, ro, rfs, rps uint32, s []uint32, so, ss, sc uint32, extent uint64, perm []uint32, po uint32, length uint64) bool {
	dataobjects.KeepAlive("PermutedExtentsAssign", nil)
	g_result := dataobjects.GetDoWorker().Run(func() interface{} {
		c_result := C.PermutedExtentsAssign(
			(*C.DoContext)(doctx.Doctx),
			(*C.uint32_t)(unsafe.Pointer(&r[0])),
			C.uint32_t(ro), C.uint32_t(rfs), C.uint32_t(rps),
			(*C.uint32_t)(_data_arg(s)),
			C.uint32_t(so), C.uint32_t(ss), C.uint32_t(sc),
			C.uint64_t(extent),
			(*C.uint32_t)(_data_arg(perm)),
			C.uint32_t(po),
			C.uint64_t(length))
		return bool(c_result)
	}).(bool)
	dataobjects.KeepAlive("r", r)
	dataobjects.KeepAlive("s", s)
	dataobjects.KeepAlive("perm", perm)
	return dataobjects.CheckResult(g_result)
}

package ecc

/*
#cgo CXXFLAGS: -std=c++17 -Ofast -fomit-frame-pointer -march=native -mtune=native -I.
#cgo CFLAGS: -I.
#cgo LDFLAGS: -L. -lReedSolomon -L/opt/homebrew/lib -lNTT -lstdc++
#include "ReedSolomon.h"
*/
import "C"
import (
	"RandomLinearCodePIR/dataobjects"
	"unsafe"
)

func GenerateSystematicRSMatrix(doctx *dataobjects.DoContext, ECCLength, M_1, p uint32, alphas, output []uint32) bool {
	dataobjects.KeepAlive("GenerateSystematicRSMatrix", nil)
	g_result := dataobjects.GetDoWorker().Run(func() interface{} {
		c_result := C.GenerateSystematicRSMatrix(
			(*C.DoContext)(doctx.Doctx),
			C.uint32_t(ECCLength), C.uint32_t(M_1), C.uint32_t(p),
			(*C.uint32_t)(unsafe.Pointer(&alphas[0])),
			(*C.uint32_t)(unsafe.Pointer(&output[0])))
		return bool(c_result)
	}).(bool)
	dataobjects.KeepAlive("output", output)
	return dataobjects.CheckResult(g_result)
}

func LagrangeInterpEval(doctx *dataobjects.DoContext, result, x, y []uint32, k, index uint32, q uint32) bool {
	dataobjects.KeepAlive("LagrangeInterpEval", nil)
	g_result := dataobjects.GetDoWorker().Run(func() interface{} {
		c_result := C.LagrangeInterpEval(
			(*C.DoContext)(doctx.Doctx),
			(*C.uint32_t)(unsafe.Pointer(&result[0])),
			(*C.uint32_t)(unsafe.Pointer(&x[0])),
			(*C.uint32_t)(unsafe.Pointer(&y[0])),
			C.uint32_t(k), C.uint32_t(index), C.uint32_t(q))
		return bool(c_result)
	}).(bool)
	dataobjects.KeepAlive("result", result)
	dataobjects.KeepAlive("x", x)
	dataobjects.KeepAlive("y", y)
	return dataobjects.CheckResult(g_result)
}

func ReedSolomonDecode(doctx *dataobjects.DoContext, code []uint32, co, cs uint64, noisyQuery []bool, ecc_len, ecc_k, q uint32, success []uint32, steps uint64) bool {
	dataobjects.KeepAlive("ReedSolomonDecode", nil)
	g_result := dataobjects.GetDoWorker().Run(func() interface{} {
		c_result := C.ReedSolomonDecode(
			(*C.DoContext)(doctx.Doctx),
			(*C.uint32_t)(unsafe.Pointer(&code[0])), C.uint64_t(co), C.uint64_t(cs),
			(*C.bool)(unsafe.Pointer(&noisyQuery[0])),
			C.uint32_t(ecc_len), C.uint32_t(ecc_k), C.uint32_t(q),
			(*C.uint32_t)(unsafe.Pointer(&success[0])),
			C.uint64_t(steps),
		)
		return bool(c_result)
	}).(bool)
	dataobjects.KeepAlive("code", code)
	dataobjects.KeepAlive("noisyQuery", noisyQuery)
	dataobjects.KeepAlive("success", success)
	return dataobjects.CheckResult(g_result)
}

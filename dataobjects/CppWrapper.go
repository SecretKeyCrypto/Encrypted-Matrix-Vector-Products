package dataobjects

/*
#cgo CXXFLAGS: -std=c++17 -Ofast -fomit-frame-pointer -march=native -mtune=native -I.
#cgo CFLAGS: -I.
#cgo LDFLAGS: -L. -L/opt/homebrew/lib -ldataobjects -lNTT -lstdc++ -L/usr/local/cuda/lib64 -lcudart -lcudadevrt
#include "common.h"
#include "fields.h"
#include "matrices.h"
*/
import "C"
import (
	"unsafe"
)

type DoContext struct {
	Doctx *C.DoContext
}

func NewDoContext() *DoContext {
	C.Setup()
	return &DoContext{
		Doctx: C.NewDoContext(),
	}
}

func FreeDoContext(ctx *DoContext) {
	C.FreeDoContext(ctx.Doctx)
}

func FieldRangeVector(doctx *DoContext, r []uint32, ro uint64, start uint32, length uint64) bool {
	KeepAlive("FieldRangeVector", nil)
	g_result := GetDoWorker().Run(func() interface{} {
		c_result := C.FieldRangeVector(
			doctx.Doctx,
			(*C.uint32_t)(unsafe.Pointer(&r[0])), C.uint64_t(ro),
			C.uint32_t(start), C.uint64_t(length),
		)
		return bool(c_result)
	}).(bool)
	KeepAlive("r", r)
	return CheckResult(g_result)
}

func FieldCopyVector(doctx *DoContext, r []uint32, ro uint64, a []uint32, ao uint64, length uint64) bool {
	KeepAlive("FieldCopyVector", nil)
	g_result := GetDoWorker().Run(func() interface{} {
		c_result := C.FieldCopyVector(
			doctx.Doctx,
			(*C.uint32_t)(unsafe.Pointer(&r[0])), C.uint64_t(ro),
			(*C.uint32_t)(unsafe.Pointer(&a[0])), C.uint64_t(ao),
			C.uint64_t(length),
		)
		return bool(c_result)
	}).(bool)
	KeepAlive("r", r)
	KeepAlive("a", a)
	return CheckResult(g_result)
}

func FieldSetVector(doctx *DoContext, r []uint32, ro uint64, length uint64, v uint32) bool {
	KeepAlive("FieldSetVector", nil)
	g_result := GetDoWorker().Run(func() interface{} {
		c_result := C.FieldSetVector(
			doctx.Doctx,
			(*C.uint32_t)(unsafe.Pointer(&r[0])), C.uint64_t(ro),
			C.uint64_t(length), C.uint32_t(v),
		)
		return bool(c_result)
	}).(bool)
	KeepAlive("r", r)
	return CheckResult(g_result)
}

func FieldAddToVector(doctx *DoContext, r []uint32, ro uint64, v uint32, length uint64) bool {
	KeepAlive("FieldAddToVector", nil)
	g_result := GetDoWorker().Run(func() interface{} {
		c_result := C.FieldAddToVector(
			doctx.Doctx,
			(*C.uint32_t)(unsafe.Pointer(&r[0])), C.uint64_t(ro),
			C.uint32_t(v),
			C.uint64_t(length),
		)
		return bool(c_result)
	}).(bool)
	KeepAlive("r", r)
	return CheckResult(g_result)
}

func FieldAddVectors(doctx *DoContext, r []uint32, ro uint64, a []uint32, ao uint64, b []uint32, bo uint64, length uint64, p uint32) bool {
	KeepAlive("FieldAddVectors", nil)
	g_result := GetDoWorker().Run(func() interface{} {
		c_result := C.FieldAddVectors(
			doctx.Doctx,
			(*C.uint32_t)(unsafe.Pointer(&r[0])), C.uint64_t(ro),
			(*C.uint32_t)(unsafe.Pointer(&a[0])), C.uint64_t(ao),
			(*C.uint32_t)(unsafe.Pointer(&b[0])), C.uint64_t(bo),
			C.uint64_t(length), C.uint32_t(p),
		)
		return bool(c_result)
	}).(bool)
	KeepAlive("r", r)
	KeepAlive("a", a)
	KeepAlive("b", b)
	return CheckResult(g_result)
}

func FieldAddVectorsExt(doctx *DoContext, r []uint32, ro, rs uint64, a []uint32, ao, as uint64, b []uint32, bo, bs uint64, length, steps uint64, p uint32) bool {
	KeepAlive("FieldAddVectorsExt", nil)
	g_result := GetDoWorker().Run(func() interface{} {
		c_result := C.FieldAddVectorsExt(
			doctx.Doctx,
			(*C.uint32_t)(unsafe.Pointer(&r[0])), C.uint64_t(ro), C.uint64_t(rs),
			(*C.uint32_t)(unsafe.Pointer(&a[0])), C.uint64_t(ao), C.uint64_t(as),
			(*C.uint32_t)(unsafe.Pointer(&b[0])), C.uint64_t(bo), C.uint64_t(bs),
			C.uint64_t(length), C.uint64_t(steps), C.uint32_t(p),
		)
		return bool(c_result)
	}).(bool)
	KeepAlive("r", r)
	KeepAlive("a", a)
	KeepAlive("b", b)
	return CheckResult(g_result)
}

func FieldMulVector(doctx *DoContext, r []uint32, ro uint64, a []uint32, ao uint64, b uint32, length uint64, p uint32) bool {
	KeepAlive("FieldMulVector", nil)
	g_result := GetDoWorker().Run(func() interface{} {
		c_result := C.FieldMulVector(
			doctx.Doctx,
			(*C.uint32_t)(unsafe.Pointer(&r[0])), C.uint64_t(ro),
			(*C.uint32_t)(unsafe.Pointer(&a[0])), C.uint64_t(ao),
			C.uint32_t(b),
			C.uint64_t(length), C.uint32_t(p),
		)
		return bool(c_result)
	}).(bool)
	KeepAlive("r", r)
	KeepAlive("a", a)
	return CheckResult(g_result)
}

func FieldMulVectorsExt(doctx *DoContext, r []uint32, ro, rs uint64, a []uint32, ao, as uint64, b []uint32, bo, bs uint64, length, steps uint64, p uint32) bool {
	KeepAlive("FieldMulVectorsExt", nil)
	g_result := GetDoWorker().Run(func() interface{} {
		c_result := C.FieldMulVectorsExt(
			doctx.Doctx,
			(*C.uint32_t)(unsafe.Pointer(&r[0])), C.uint64_t(ro), C.uint64_t(rs),
			(*C.uint32_t)(unsafe.Pointer(&a[0])), C.uint64_t(ao), C.uint64_t(as),
			(*C.uint32_t)(unsafe.Pointer(&b[0])), C.uint64_t(bo), C.uint64_t(bs),
			C.uint64_t(length), C.uint64_t(steps), C.uint32_t(p),
		)
		return bool(c_result)
	}).(bool)
	KeepAlive("r", r)
	KeepAlive("a", a)
	KeepAlive("b", b)
	return CheckResult(g_result)
}

func FieldSubVectors(doctx *DoContext, r []uint32, ro uint64, a []uint32, ao uint64, b []uint32, bo uint64, length uint64, p uint32) bool {
	KeepAlive("FieldSubVectors", nil)
	g_result := GetDoWorker().Run(func() interface{} {
		c_result := C.FieldSubVectors(
			doctx.Doctx,
			(*C.uint32_t)(unsafe.Pointer(&r[0])), C.uint64_t(ro),
			(*C.uint32_t)(unsafe.Pointer(&a[0])), C.uint64_t(ao),
			(*C.uint32_t)(unsafe.Pointer(&b[0])), C.uint64_t(bo),
			C.uint64_t(length), C.uint32_t(p),
		)
		return bool(c_result)
	}).(bool)
	KeepAlive("r", r)
	KeepAlive("a", a)
	KeepAlive("b", b)
	return CheckResult(g_result)
}

func FieldNegVector(doctx *DoContext, r []uint32, ro uint64, length uint64, p uint32) bool {
	KeepAlive("FieldNegVector", nil)
	g_result := GetDoWorker().Run(func() interface{} {
		c_result := C.FieldNegVector(
			doctx.Doctx,
			(*C.uint32_t)(unsafe.Pointer(&r[0])), C.uint64_t(ro),
			C.uint64_t(length), C.uint32_t(p),
		)
		return bool(c_result)
	}).(bool)
	KeepAlive("r", r)
	return CheckResult(g_result)
}

func FieldNegVectorExt(doctx *DoContext, r []uint32, ro uint64, length, stride, steps uint64, p uint32) bool {
	KeepAlive("FieldNegVectorExt", nil)
	g_result := GetDoWorker().Run(func() interface{} {
		c_result := C.FieldNegVectorExt(
			doctx.Doctx,
			(*C.uint32_t)(unsafe.Pointer(&r[0])), C.uint64_t(ro),
			C.uint64_t(length), C.uint64_t(stride), C.uint64_t(steps), C.uint32_t(p),
		)
		return bool(c_result)
	}).(bool)
	KeepAlive("r", r)
	return CheckResult(g_result)
}

func FieldAddVectorIfNonZero(doctx *DoContext, t []bool, t_index uint64, r []uint32, ro uint64, e []uint32, eo uint64, length uint64, p uint32) bool {
	KeepAlive("FieldAddVectorIfNonZero", nil)
	g_result := GetDoWorker().Run(func() interface{} {
		c_result := C.FieldAddVectorIfNonZero(
			doctx.Doctx,
			(*C.bool)(unsafe.Pointer(&t[0])), C.uint64_t(t_index),
			(*C.uint32_t)(unsafe.Pointer(&r[0])), C.uint64_t(ro),
			(*C.uint32_t)(unsafe.Pointer(&e[0])), C.uint64_t(eo),
			C.uint64_t(length), C.uint32_t(p),
		)
		return bool(c_result)
	}).(bool)
	KeepAlive("t", t)
	KeepAlive("r", r)
	KeepAlive("e", e)
	return CheckResult(g_result)
}

func FieldAddVectorIfNonZeroExt(doctx *DoContext, t []bool, to, ts uint64, r []uint32, ro, rs uint64, e []uint32, eo, es uint64, length, steps uint64, p uint32) bool {
	KeepAlive("FieldAddVectorIfNonZeroExt", nil)
	g_result := GetDoWorker().Run(func() interface{} {
		c_result := C.FieldAddVectorIfNonZeroExt(
			doctx.Doctx,
			(*C.bool)(unsafe.Pointer(&t[0])), C.uint64_t(to), C.uint64_t(ts),
			(*C.uint32_t)(unsafe.Pointer(&r[0])), C.uint64_t(ro), C.uint64_t(rs),
			(*C.uint32_t)(unsafe.Pointer(&e[0])), C.uint64_t(eo), C.uint64_t(es),
			C.uint64_t(length), C.uint64_t(steps), C.uint32_t(p),
		)
		return bool(c_result)
	}).(bool)
	KeepAlive("t", t)
	KeepAlive("r", r)
	KeepAlive("e", e)
	return CheckResult(g_result)
}

func FieldInvVector(doctx *DoContext, r []uint32, ro uint64, a []uint32, ao uint64, length uint64, p uint32) bool {
	KeepAlive("FieldInvVector", nil)
	g_result := GetDoWorker().Run(func() interface{} {
		c_result := C.FieldInvVector(
			doctx.Doctx,
			(*C.uint32_t)(unsafe.Pointer(&r[0])), C.uint64_t(ro),
			(*C.uint32_t)(unsafe.Pointer(&a[0])), C.uint64_t(ao),
			C.uint64_t(length), C.uint32_t(p),
		)
		return bool(c_result)
	}).(bool)
	KeepAlive("r", r)
	KeepAlive("a", a)
	return CheckResult(g_result)
}

func MatrixTranspose(doctx *DoContext, r []uint32, ro uint32, a []uint32, ao, M, N uint32) bool {
	KeepAlive("MatrixTranspose", nil)
	g_result := GetDoWorker().Run(func() interface{} {
		c_result := C.MatrixTranspose(
			doctx.Doctx,
			(*C.uint32_t)(unsafe.Pointer(&r[0])), C.uint32_t(ro),
			(*C.uint32_t)(unsafe.Pointer(&a[0])), C.uint32_t(ao),
			C.uint32_t(M), C.uint32_t(N),
		)
		return bool(c_result)
	}).(bool)
	KeepAlive("r", r)
	KeepAlive("a", a)
	return CheckResult(g_result)
}

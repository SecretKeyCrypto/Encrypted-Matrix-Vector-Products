package mvp

/*
#include "mvp.h"
#include "matrixShapeTransform.h"
#include "lpn.h"
*/
import "C"
import (
	"RandomLinearCodePIR/dataobjects"
	"unsafe"
)

func cBlockMatVecProduct(doctx *dataobjects.DoContext, mat, vec, out []uint32, row, col, numBlock, p uint32) bool {
	dataobjects.KeepAlive("cBlockMatVecProduct", nil)
	g_result := dataobjects.GetDoWorker().Run(func() interface{} {
		c_result := C.BlockMatVecProduct(
			(*C.DoContext)(doctx.Doctx),
			(*C.uint32_t)(unsafe.Pointer(&mat[0])),
			(*C.uint32_t)(unsafe.Pointer(&vec[0])),
			(*C.uint32_t)(unsafe.Pointer(&out[0])),
			C.uint32_t(row), C.uint32_t(col), C.uint32_t(numBlock), C.uint32_t(p),
		)
		return bool(c_result)
	}).(bool)
	dataobjects.KeepAlive("mat", mat)
	dataobjects.KeepAlive("vec", vec)
	dataobjects.KeepAlive("out", out)
	return dataobjects.CheckResult(g_result)
}

func cMatVecProduct(doctx *dataobjects.DoContext, mat, vec, out []uint32, row, col, p uint32) bool {
	dataobjects.KeepAlive("cMatVecProduct", nil)
	g_result := dataobjects.GetDoWorker().Run(func() interface{} {
		c_result := C.MatVecProduct(
			(*C.DoContext)(doctx.Doctx),
			(*C.uint32_t)(unsafe.Pointer(&mat[0])),
			(*C.uint32_t)(unsafe.Pointer(&vec[0])),
			(*C.uint32_t)(unsafe.Pointer(&out[0])),
			C.uint32_t(row), C.uint32_t(col), C.uint32_t(p),
		)
		return bool(c_result)
	}).(bool)
	dataobjects.KeepAlive("mat", mat)
	dataobjects.KeepAlive("vec", vec)
	dataobjects.KeepAlive("out", out)
	return dataobjects.CheckResult(g_result)
}

func cMatVecProductExt(doctx *dataobjects.DoContext, mat []uint32, mo, ms uint32, vec []uint32, vo, vs uint32, out []uint32, oo, os, row, col, steps, p uint32) bool {
	dataobjects.KeepAlive("cMatVecProductExt", nil)
	g_result := dataobjects.GetDoWorker().Run(func() interface{} {
		c_result := C.MatVecProductExt(
			(*C.DoContext)(doctx.Doctx),
			(*C.uint32_t)(unsafe.Pointer(&mat[0])), C.uint32_t(mo), C.uint32_t(ms),
			(*C.uint32_t)(unsafe.Pointer(&vec[0])), C.uint32_t(vo), C.uint32_t(vs),
			(*C.uint32_t)(unsafe.Pointer(&out[0])), C.uint32_t(oo), C.uint32_t(os),
			C.uint32_t(row), C.uint32_t(col), C.uint32_t(steps), C.uint32_t(p),
		)
		return bool(c_result)
	}).(bool)
	dataobjects.KeepAlive("mat", mat)
	dataobjects.KeepAlive("vec", vec)
	dataobjects.KeepAlive("out", out)
	return dataobjects.CheckResult(g_result)
}

func cBlockVecMatProduct(doctx *dataobjects.DoContext, mat, vec, out []uint32, row, col, numBlock, p uint32) bool {
	dataobjects.KeepAlive("cBlockVecMatProduct", nil)
	g_result := dataobjects.GetDoWorker().Run(func() interface{} {
		c_result := C.BlockVecMatProduct(
			(*C.DoContext)(doctx.Doctx),
			(*C.uint32_t)(unsafe.Pointer(&mat[0])),
			(*C.uint32_t)(unsafe.Pointer(&vec[0])),
			(*C.uint32_t)(unsafe.Pointer(&out[0])),
			C.uint32_t(row), C.uint32_t(col), C.uint32_t(numBlock), C.uint32_t(p),
		)
		return bool(c_result)
	}).(bool)
	dataobjects.KeepAlive("mat", mat)
	dataobjects.KeepAlive("vec", vec)
	dataobjects.KeepAlive("out", out)
	return dataobjects.CheckResult(g_result)
}

func cTransformToBlockwise(doctx *dataobjects.DoContext, mat []uint32, mo, ms uint32, matBlocked []uint32, bo, bs, n, m, s uint32) bool {
	dataobjects.KeepAlive("cTransformToBlockwise", nil)
	g_result := dataobjects.GetDoWorker().Run(func() interface{} {
		c_result := C.TransformRowMajorToBlockRowMajor(
			(*C.DoContext)(doctx.Doctx),
			(*C.uint32_t)(unsafe.Pointer(&mat[0])), C.uint32_t(mo), C.uint32_t(ms),
			(*C.uint32_t)(unsafe.Pointer(&matBlocked[0])), C.uint32_t(bo), C.uint32_t(bs),
			C.uint32_t(n), C.uint32_t(m), C.uint32_t(s),
		)
		return bool(c_result)
	}).(bool)
	dataobjects.KeepAlive("mat", mat)
	dataobjects.KeepAlive("matBlocked", matBlocked)
	return dataobjects.CheckResult(g_result)
}

func cLpnEncode(
	doctx *dataobjects.DoContext,
	input, rlcMatrix, generatorMatrix []uint32, // [M * L], [K * L], [ECCLength * M_1]
	encoded []uint32, // [ECCLength * M / M_1 * N]
	M, L, K, M_1, ECCLength, P uint32,
) bool {
	g_result := dataobjects.GetDoWorker().Run(func() interface{} {
		c_result := C.LpnEncode(
			(*C.DoContext)(doctx.Doctx),
			(*C.uint32_t)(unsafe.Pointer(&input[0])),
			(*C.uint32_t)(unsafe.Pointer(&rlcMatrix[0])),
			(*C.uint32_t)(unsafe.Pointer(&generatorMatrix[0])),
			(*C.uint32_t)(unsafe.Pointer(&encoded[0])),
			C.uint32_t(M), C.uint32_t(L), C.uint32_t(K), C.uint32_t(M_1), C.uint32_t(ECCLength), C.uint32_t(P),
		)
		return bool(c_result)
	}).(bool)
	dataobjects.KeepAlive("input", input)
	dataobjects.KeepAlive("rlcMatrix", rlcMatrix)
	dataobjects.KeepAlive("generatorMatrix", generatorMatrix)
	dataobjects.KeepAlive("encoded", encoded)
	return dataobjects.CheckResult(g_result)
}

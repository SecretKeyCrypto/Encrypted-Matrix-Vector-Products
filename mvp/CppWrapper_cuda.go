//go:build cuda
// +build cuda

package mvp

import "RandomLinearCodePIR/dataobjects"

/*
#cgo CXXFLAGS: -std=c++17 -Ofast -fomit-frame-pointer -march=native -mtune=native -I.
#cgo CFLAGS: -I.
#cgo LDFLAGS: -L.. cuda_devlink.o -L../dataobjects -ldataobjects -L../utils -lutils -L../tdm -lNTT -L../ecc -lReedSolomon -L../mvp -lMVP -L/usr/local/cuda/lib64 -lcudart -lcudadevrt
#include "mvp.h"
#include "matrixShapeTransform.h"
*/
import "C"

func BlockMatVecProduct(doctx *dataobjects.DoContext, mat, vec, out []uint32, row, col, numBlock, p uint32) bool {
	return cBlockMatVecProduct(doctx, mat, vec, out, row, col, numBlock, p)
}

func MatVecProduct(doctx *dataobjects.DoContext, mat, vec, out []uint32, row, col, p uint32) bool {
	return cMatVecProduct(doctx, mat, vec, out, row, col, p)
}

func MatVecProductExt(doctx *dataobjects.DoContext, mat []uint32, mo, ms uint32, vec []uint32, vo, vs uint32, out []uint32, oo, os, row, col, steps, p uint32) bool {
	return cMatVecProductExt(doctx, mat, mo, ms, vec, vo, vs, out, oo, os, row, col, steps, p)
}

func BlockVecMatProduct(doctx *dataobjects.DoContext, mat, vec, out []uint32, row, col, numBlock, p uint32) bool {
	return cBlockVecMatProduct(doctx, mat, vec, out, row, col, numBlock, p)
}

func TransformToBlockwise(doctx *dataobjects.DoContext, mat []uint32, mo, ms uint32, matBlocked []uint32, bo, bs, n, m, s uint32) bool {
	return cTransformToBlockwise(doctx, mat, mo, ms, matBlocked, bo, bs, n, m, s)
}

func LpnEncode(
	doctx *dataobjects.DoContext,
	input, rlcMatrix, generatorMatrix []uint32, // [M * L], [K * L], [ECCLength * M_1]
	encoded []uint32, // [ECCLength * M / M_1 * N]
	M, L, K, M_1, ECCLength, P uint32,
) bool {
	return cLpnEncode(doctx, input, rlcMatrix, generatorMatrix, encoded, M, L, K, M_1, ECCLength, P)
}

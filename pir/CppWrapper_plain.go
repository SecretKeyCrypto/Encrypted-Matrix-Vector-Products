//go:build !cuda
// +build !cuda

package pir

/*
#cgo LDFLAGS: -L../dataobjects -ldataobjects -L../utils -lutils -L../tdm -lNTT -L../ecc -lReedSolomon -L../mvp -lMVP
#include "BitMVP.h"
*/
import "C"

func VecMatMulF4(bit1Result, bitPResult, bit1Matrix, bitPMatrix, bit1Vec, bitPVec []uint32, rows, cols uint32) {
	cVecMatMulF4(bit1Result, bitPResult, bit1Matrix, bitPMatrix, bit1Vec, bitPVec, rows, cols)
}

func VecMatrixMulF2(result, matrix, vector []uint32, rows, cols uint32) {
	cVecMatrixMulF2(result, matrix, vector, rows, cols)
}

func MatrixColXORByBlock2D(vector_1, vector_2, matrixData, result_1, result_2 []uint32, rows, cols, block_size uint32) {
	cMatrixColXORByBlock2D(vector_1, vector_2, matrixData, result_1, result_2, rows, cols, block_size)
}

func MatrixColXORByBlock(vector, matrixData, result []uint32, rows, cols, block_size uint32) {
	cMatrixColXORByBlock(vector, matrixData, result, rows, cols, block_size)
}

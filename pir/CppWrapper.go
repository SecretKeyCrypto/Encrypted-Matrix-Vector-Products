package pir

/*
#cgo LDFLAGS: -L../tdm -L../ecc -lNTT -lReedSolomon -lntl -lgmp
#include "BitMVP.h"
*/
import "C"
import (
	"unsafe"
)

func VecMatMulF4(bit1Result, bitPResult, bit1Matrix, bitPMatrix, bit1Vec, bitPVec []uint32, rows, cols uint32) {
	// (a + bp) * (x + yp)
	ax := make([]uint32, cols)
	by := make([]uint32, cols)
	ay := make([]uint32, cols)
	bx := make([]uint32, cols)

	VecMatrixMulF2(ax, bit1Matrix, bit1Vec, rows, cols)
	VecMatrixMulF2(by, bitPMatrix, bitPVec, rows, cols)
	VecMatrixMulF2(ay, bitPMatrix, bit1Vec, rows, cols)
	VecMatrixMulF2(bx, bit1Matrix, bitPVec, rows, cols)

	for i := range ax {
		bit1Result[i] = ax[i] ^ by[i]
		bitPResult[i] = ay[i] ^ bx[i] ^ by[i]
	}
}

func VecMatrixMulF2(result, matrix, vector []uint32, rows, cols uint32) {
	cVector := (*C.uint32_t)(unsafe.Pointer(&vector[0]))
	cMatrix := (*C.uint32_t)(unsafe.Pointer(&matrix[0]))
	cResult := (*C.uint32_t)(unsafe.Pointer(&result[0]))

	C.VecMatrixMulF2(cResult, cMatrix, cVector, (C.uint32_t)(rows), (C.uint32_t)(cols))
}

// For 2D Split LSN
func MatrixColXORByBlock2D(vector_1, vector_2, matrixData, result_1, result_2 []uint32, rows, cols, block_size uint32) {

	cVector_1 := (*C.uint32_t)(unsafe.Pointer(&vector_1[0]))
	cVector_2 := (*C.uint32_t)(unsafe.Pointer(&vector_2[0]))

	cMatrix := (*C.uint32_t)(unsafe.Pointer(&matrixData[0]))
	cResult1 := (*C.uint32_t)(unsafe.Pointer(&result_1[0]))
	cResult2 := (*C.uint32_t)(unsafe.Pointer(&result_2[0]))

	C.MatrixColXORByBlock2D(cResult1, cResult2, cMatrix, cVector_1, cVector_2, (C.uint32_t)(rows), (C.uint32_t)(cols), (C.uint32_t)(block_size))
}

// For 1D Split LSN
func MatrixColXORByBlock(vector, matrixData, result []uint32, rows, cols, block_size uint32) {
	cVector := (*C.uint32_t)(unsafe.Pointer(&vector[0]))
	cMatrix := (*C.uint32_t)(unsafe.Pointer(&matrixData[0]))
	cResult := (*C.uint32_t)(unsafe.Pointer(&result[0]))

	C.MatrixColXORByBlock(cResult, cMatrix, cVector, (C.uint32_t)(rows), (C.uint32_t)(cols), (C.uint32_t)(block_size))
}

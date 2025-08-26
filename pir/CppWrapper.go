package pir

/*
#include "BitMVP.h"
*/
import "C"
import (
	"unsafe"
)

func MatrixColXORByBlock(vector_1, vector_2, matrixData, result_1, result_2 []uint32, rows, cols, block_size uint32) {

	cVector_1 := (*C.uint32_t)(unsafe.Pointer(&vector_1[0]))
	cVector_2 := (*C.uint32_t)(unsafe.Pointer(&vector_2[0]))

	cMatrix := (*C.uint32_t)(unsafe.Pointer(&matrixData[0]))
	cResult1 := (*C.uint32_t)(unsafe.Pointer(&result_1[0]))
	cResult2 := (*C.uint32_t)(unsafe.Pointer(&result_2[0]))

	C.MatrixColXORByBlock(cResult1, cResult2, cMatrix, cVector_1, cVector_2, (C.uint32_t)(rows), (C.uint32_t)(cols), (C.uint32_t)(block_size))
}

package pir

/*
#include "BitMVP.h"
*/
import "C"
import (
	"RandomLinearCodePIR/dataobjects"
	"RandomLinearCodePIR/utils"
	"math/rand"
	"unsafe"
)

type BasePIR struct {
	Params BaseParams
}

type BaseParams struct {
	Rows           uint32
	Cols           uint32
	NumberOfBlocks uint32
	CodewordLength uint32
	PackedSize     uint32
}

type BasePIRQuery struct {
	Vector_1 []uint32
	Vector_2 []uint32
}

type BasePIRAux struct {
	FlipVector []uint32
	MaskValue  uint32
}
type BasePIRAnswer struct {
	Result_1 []uint32
	Result_2 []uint32
}

func (p *BasePIR) KeyGen(N, ell, lambda int, seed int64) SecretKey {
	return SecretKey{
		LinearCodeKey: seed,
		// Hardcode for now
		MaskKey: seed + 1,
		Lambda:  lambda,
		Ell:     ell,
	}
}

func (p *BasePIR) Encode(sk SecretKey, matrix Matrix) *Matrix {
	M := p.Params.CodewordLength

	encodedMatrix := SystematicEncoding(M, sk.LinearCodeKey, matrix)

	// Transpose the encoded matrix for more efficient access pattern in C for XOR of rows instead of columns.
	packedData := PackAndTransposeMatrix(encodedMatrix, matrix.Rows, M)

	// Update the packed Data Length
	p.Params.PackedSize = (matrix.Rows + 31) / 32

	// Mask the packed matrix column wisely
	for i := uint32(0); i < p.Params.PackedSize; i++ {
		rng := rand.New(rand.NewSource(sk.MaskKey + int64(i)))
		index := i
		for j := uint32(0); j < M; j++ {
			packedData[index] ^= rng.Uint32()
			index += p.Params.PackedSize

		}
	}

	return &Matrix{
		Rows:      M,
		Cols:      p.Params.PackedSize,
		EntryBits: 32,
		Data:      packedData,
	}
}

func (p *BasePIR) Query(sk SecretKey, queryIndex uint64) (*BasePIRQuery, *BasePIRAux) {
	vector_1 := SampleVectorFromNullSpace(p.Params.Cols, p.Params.CodewordLength, sk.LinearCodeKey)

	// Add Unit Vector to retrieve the ith column
	vector_1[queryIndex%uint64(p.Params.Cols)] ^= 1

	// Calculate the mask for the final result
	rng := rand.New(rand.NewSource(int64(sk.MaskKey) + int64((queryIndex/uint64(p.Params.Cols))/32)))

	mask := uint32(0)

	for i := range vector_1 {
		mask ^= vector_1[i] * rng.Uint32()
	}

	vector_2 := dataobjects.AlignedMake[uint32](uint64(p.Params.CodewordLength))

	flipVector := utils.RandomizeFlipVector(p.Params.NumberOfBlocks)

	// TODO : Make sure it devides
	blockSize := p.Params.CodewordLength / p.Params.NumberOfBlocks

	index := uint32(0)
	for ind, val := range flipVector {
		index = uint32(blockSize) * uint32(ind)

		if val == 0 {
			for j := uint32(0); j < uint32(blockSize); j++ {
				vector_2[index+j] = uint32(rand.Intn(2))
			}
		} else if val == 1 {
			for j := uint32(0); j < uint32(blockSize); j++ {
				vector_2[index+j] ^= vector_1[index+j]
				vector_1[index+j] = uint32(rand.Intn(2))
			}
		} else {
			for j := uint32(0); j < uint32(blockSize); j++ {
				vector_2[index+j] = uint32(rand.Intn(2))
				vector_1[index+j] ^= vector_2[index+j]
			}
		}
	}

	return &BasePIRQuery{
			Vector_1: vector_1,
			Vector_2: vector_2,
		}, &BasePIRAux{
			FlipVector: flipVector,
			MaskValue:  mask,
		}
}

func (p *BasePIR) Answer(matrix *Matrix, clientQuery *BasePIRQuery) *BasePIRAnswer {
	rows := matrix.Rows
	cols := matrix.Cols

	// TODO : Make sure it devides
	block_size := p.Params.CodewordLength / p.Params.NumberOfBlocks

	result1 := dataobjects.AlignedMake[uint32](uint64(cols * rows / block_size))
	result2 := dataobjects.AlignedMake[uint32](uint64(cols * rows / block_size))

	cVector_1 := (*C.uint32_t)(unsafe.Pointer(&clientQuery.Vector_1[0]))
	cVector_2 := (*C.uint32_t)(unsafe.Pointer(&clientQuery.Vector_2[0]))

	cMatrix := (*C.uint32_t)(unsafe.Pointer(&matrix.Data[0]))
	cResult1 := (*C.uint32_t)(unsafe.Pointer(&result1[0]))
	cResult2 := (*C.uint32_t)(unsafe.Pointer(&result2[0]))

	C.MatrixColXORByBlock(cResult1, cResult2, cMatrix, cVector_1, cVector_2, (C.uint32_t)(rows), (C.uint32_t)(cols), (C.uint32_t)(block_size))
	return &BasePIRAnswer{
		Result_1: result1,
		Result_2: result2,
	}
}

func (p *BasePIR) Decode(sk SecretKey, index uint64, response *BasePIRAnswer, aux *BasePIRAux) uint32 {
	res := dataobjects.AlignedMake[uint32](uint64(p.Params.PackedSize))
	row := index / uint64(p.Params.Cols)
	wordIndex := row / 32
	bitOffset := row % 32

	start := 0
	for _, val := range aux.FlipVector {
		if val != 0 {
			for i := range res {
				res[i] ^= response.Result_2[start+i]
			}

		}

		if val != 1 {
			for i := range res {
				res[i] ^= response.Result_1[start+i]
			}
		}

		start += int(p.Params.PackedSize)
	}

	val := res[wordIndex] ^ aux.MaskValue

	return (val >> bitOffset) & 1
}

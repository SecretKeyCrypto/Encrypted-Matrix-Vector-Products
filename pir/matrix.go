package pir

import (
	"RandomLinearCodePIR/dataobjects"
	"math"
	"math/rand"
)

type Matrix struct {
	Rows      uint32
	Cols      uint32
	EntryBits uint32
	Data      []uint32
}

func GenerateMatrix(rows, cols, entryBits uint32, seed int64) Matrix {
	rng := rand.New(rand.NewSource(seed))

	dataSize := uint64(rows) * uint64(cols)

	data := dataobjects.AlignedMake[uint32](uint64(dataSize))

	bitmask := uint32((1 << entryBits) - 1)

	for i := range data {
		data[i] = rng.Uint32() & bitmask
	}

	return Matrix{
		Rows:      rows,
		Cols:      cols,
		EntryBits: entryBits,
		Data:      data,
	}
}

func GenerateRandomMatrix(rows, cols uint32, seed int64) [][]uint32 {
	rng := rand.New(rand.NewSource(seed)) // PRNG with fixed seed
	matrix := make([][]uint32, rows)

	for i := uint32(0); i < rows; i++ {
		matrix[i] = dataobjects.AlignedMake[uint32](uint64(cols))
		for j := uint32(0); j < cols; j++ {
			matrix[i][j] = rng.Uint32() % 2 // Random F₂ value (0 or 1)
		}
	}

	return matrix
}

func GeneratePrimeFieldMatrix(rows, cols, p uint32, seed int64) Matrix {
	rng := rand.New(rand.NewSource(seed))

	dataSize := uint64(rows) * uint64(cols)

	data := dataobjects.AlignedMake[uint32](uint64(dataSize))

	for i := range data {
		data[i] = uint32(rng.Intn(int(p)))
	}

	return Matrix{
		Rows:      rows,
		Cols:      cols,
		EntryBits: uint32(math.Ceil(math.Log2(float64(p)))),
		Data:      data,
	}
}

func PackAndTransposeMatrix(matrix [][]uint32, rows, cols uint32) []uint32 {
	packedSize := (rows + 31) / 32 * cols // Each column stores 32 rows
	packedData := dataobjects.AlignedMake[uint32](uint64(packedSize))

	// Column-major packing (transpose)
	for col := uint32(0); col < cols; col++ {
		for row := uint32(0); row < rows; row++ {
			index := col*((rows+31)/32) + row/32 // Compute index in packed data, which is stored column by column
			offset := row % 32                   // Bit position in uint32
			if matrix[row][col] == 1 {
				packedData[index] |= (1 << offset) // Set bit
			}
		}
	}

	return packedData
}

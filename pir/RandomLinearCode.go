package pir

import (
	"RandomLinearCodePIR/dataobjects"
	"RandomLinearCodePIR/utils"
	"math/rand"
)

// The functions in this file are hardcoded for F_2

// Generate Sysmtematic Random Linear Code which has form G=(I_N | P) where P has dimension N * (M-N)
// Random each column of P using seed = seed + col where col := range 1 to M-N
func GenerateRandomLinearCode(N, M uint32, seed int64) {

}

func GenerateRandomColsOfRLC(N, M uint32, seed int64) [][]uint32 {
	if M < N {
		panic("Codeword length should be longer than Message length.")
	}

	matrix := make([][]uint32, N)

	for i := uint32(0); i < N; i++ {
		matrix[i] = dataobjects.AlignedMake[uint32](uint64(M - N))
	}

	for j := uint32(0); j < M-N; j++ {
		rng := rand.New(rand.NewSource(seed + int64(j)))
		for i := uint32(0); i < N; i++ {
			matrix[i][j] = rng.Uint32() % 2
		}
	}

	return matrix
}

func SystematicEncoding(M uint32, seed int64, matrix Matrix) [][]uint32 {
	N := matrix.Cols
	RandomColsOfRLC := GenerateRandomColsOfRLC(N, M, seed)

	encodedMatrix := make([][]uint32, matrix.Rows)
	for i := range encodedMatrix {
		encodedMatrix[i] = dataobjects.AlignedMake[uint32](uint64(M))
	}

	for row := uint32(0); row < matrix.Rows; row++ {
		start := uint64(row) * uint64(matrix.Cols)
		end := start + uint64(matrix.Cols)
		originalRow := matrix.Data[start:end]

		// Copy systematic part (first N bits)
		copy(encodedMatrix[row][:N], originalRow[:N])

		// Compute last λ bits using precomputed random matrix
		for j := uint32(0); j < M-N; j++ {
			encodedMatrix[row][N+j] = 0
			for i := uint32(0); i < N; i++ {
				if RandomColsOfRLC[i][j] == 1 {
					encodedMatrix[row][N+j] ^= encodedMatrix[row][i]
				}
			}
		}
	}

	return encodedMatrix
}

// The Parity check matrix has the form H = vcat(P, I_(M-N))
// We sample a vector of length M-N in F2 to be the coefficients of the linear combination of the columns
// We can do XOR of the columns while we know the column i is composed by the ith column of P and the ith unit vector
func SampleVectorFromNullSpace(N, M uint32, seed int64) []uint32 {
	coeff := utils.RandomizeBinaryVector(M - N)
	res := dataobjects.AlignedMake[uint32](uint64(M))

	for i := uint32(0); i < M-N; i++ {
		if coeff[i] == 1 {
			rng := rand.New(rand.NewSource(seed + int64(i)))
			for j := uint32(0); j < N; j++ {
				res[j] ^= rng.Uint32() % 2
			}
			res[N+i] ^= 1
		}
	}

	return res
}

package utils

import (
	"RandomLinearCodePIR/dataobjects"
	"math/rand"
)

func RandomizeVector(data []uint32, length uint32) {
	randomize_vector(data, length)
}

func RandomizeVectorWithSeed(data []uint32, length uint32, seed int64) {
	randomize_vector_with_seed(data, length, seed)
}

func RandomizeVectorWithModulus(data []uint32, length uint32, modulus uint32) {
	randomize_vector_with_modulus(data, length, modulus)
}

func RandomizeVectorWithModulusAndSeed(data []uint32, length uint32, modulus uint32, seed int64) {
	randomize_vector_with_modulus_and_seed(data, length, modulus, seed)
}

// =========== Random Vectors ===========

func RandomizeQueryVector(N uint32, i uint64) []uint32 {
	return RandomizeBinaryVector(N)
}

func RandomizeBinaryVector(N uint32) []uint32 {
	vector := dataobjects.AlignedMake[uint32](uint64(N))
	for i := range vector {
		vector[i] = uint32(rand.Intn(2)) // Generates either 0 or 1
	}
	return vector
}

// Generator vector that contains values in {0,1,2}
func RandomizeFlipVector(N uint32) []uint32 {
	vector := dataobjects.AlignedMake[uint32](uint64(N))
	for i := range vector {
		vector[i] = uint32(rand.Intn(3)) // Generates either 0 or 1
	}
	return vector
}

func RandomSplitLSNNoiseCoeff(s uint32, p uint32) []uint32 {
	vector := dataobjects.AlignedMake[uint32](uint64(s))
	for i := range vector {
		vector[i] = uint32(rand.Intn(int(p)-1) + 1) // Generates non-zero values in F_p
	}
	return vector
}

func RandomPrimeFieldVector(n uint32, p uint32) []uint32 {
	vector := dataobjects.AlignedMake[uint32](uint64(n))
	if dataobjects.USE_FAST_CODE {
		randomize_vector_with_modulus(vector, n, p)
	} else {
		for i := range vector {
			vector[i] = uint32(rand.Intn(int(p))) // Generates non-zero values in F_p
		}
	}
	return vector
}

func RandomNoiseVector(n uint32, epsi float32, p uint32) []uint32 {
	vector := dataobjects.AlignedMake[uint32](uint64(n))
	for i := range vector {
		if rand.Float32() <= epsi {
			vector[i] = uint32(rand.Intn(int(p-1))) + 1 // Generates non-zero values in F_p
		}
	}
	return vector
}

func RandomLPNNoiseVector(n uint32, epsi float64, field dataobjects.Field) []uint32 {
	vector := dataobjects.AlignedMake[uint32](uint64(n))
	for i := range vector {
		if rand.Float64() <= epsi {
			var val uint32
			for {
				val = field.SampleElement()
				if val != 0 {
					break
				}
			}
			vector[i] = val
		}
	}
	return vector
}

func IsZeroVector(v []uint32) bool {
	for _, val := range v {
		if val != 0 {
			return false
		}
	}
	return true
}

// =========== Random Matrix ===========
func GeneratePrimeFieldMatrix(rows, cols, p uint32, seed int64) dataobjects.Matrix {
	dataSize := uint64(rows) * uint64(cols)

	data := dataobjects.AlignedMake[uint32](uint64(dataSize))

	if dataobjects.USE_FAST_CODE {
		RandomizeVectorWithModulusAndSeed(data, uint32(dataSize), p, seed)
	} else {
		rng := rand.New(rand.NewSource(seed))
		for i := range data {
			data[i] = uint32(rng.Intn(int(p)))
		}
	}

	return dataobjects.Matrix{
		Rows: rows,
		Cols: cols,
		Data: data,
	}
}

// =========== Other ===========

func ModInverse(a, p uint32) uint32 {
	var t, newT int64 = 0, 1
	var r, newR int64 = int64(p), int64(a)

	for newR != 0 {
		quotient := r / newR
		t, newT = newT, t-quotient*newT
		r, newR = newR, r-quotient*newR
	}

	if r > 1 {
		panic("a is not invertible")
	}

	if t < 0 {
		t += int64(p)
	}

	return uint32(t)
}

func RoundUp(x, b uint32) uint32 {
	if x%b == 0 {
		return x
	}
	return ((x / b) + 1) * b
}

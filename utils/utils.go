package utils

import (
	"RandomLinearCodePIR/dataobjects"
	"math/rand"
)

func RandomizeVector(data []uint32, M, N uint32, transpose bool) {
	randomize_vector(data, M, N, transpose)
}

func RandomizeVectorWithSeed(data []uint32, M, N uint32, transpose bool, seed int64) {
	randomize_vector_with_seed(data, M, N, transpose, seed)
}

func RandomizeVectorWithModulus(data []uint32, M, N uint32, transpose bool, modulus uint32) {
	randomize_vector_with_modulus(data, M, N, transpose, modulus)
}

func RandomizeVectorWithModulusAndSeed(data []uint32, M, N uint32, transpose bool, modulus uint32, seed int64) {
	randomize_vector_with_modulus_and_seed(data, M, N, transpose, modulus, seed)
}

// =========== Random Vectors ===========

func RandomizeQueryVector(N uint32, i uint64) []uint32 {
	return RandomizeBinaryVector(N)
}

func RandomizeBinaryVectorWithSeed(N uint32, seed int64) []uint32 {
	rng := rand.New(rand.NewSource(seed))
	vector := make([]uint32, N)
	for i := range vector {
		vector[i] = uint32(rng.Intn(2)) // Generates either 0 or 1
	}
	return vector
}

func RandomizeUInt32Vector(N uint32) []uint32 {
	vector := make([]uint32, N)
	for i := range vector {
		vector[i] = rand.Uint32() // Generates either 0 or 1
	}
	return vector
}

func RandomizeBinaryVector(N uint32) []uint32 {
	vector := dataobjects.AlignedMake[uint32](uint64(N))
	for i := range vector {
		vector[i] = uint32(rand.Intn(2)) // Generates either 0 or 1
	}
	return vector
}

func PackBinaryVectorByBlock(src []uint32, nBlock uint32) []uint32 {
	n := uint32(len(src))
	blockSize := n / nBlock
	dst := make([]uint32, ((blockSize+31)/32)*nBlock)
	bitIdx := uint32(0)
	for b := uint32(0); b < nBlock; b++ {
		blockOffset := b * ((blockSize + 31) / 32)
		for i := uint32(0); i < blockSize && bitIdx < n; i++ {
			if src[bitIdx]&1 != 0 {
				w := i >> 5
				bit := uint(i & 31)
				dst[blockOffset+w] |= 1 << bit
			}
			bitIdx++
		}
	}
	return dst
}

func PackBinaryVector(src []uint32) []uint32 {
	n := len(src)
	m := (n + 31) >> 5 // ceil(n/32)
	dst := make([]uint32, m)
	for i, v := range src {
		if v&1 != 0 {
			dst[i>>5] |= 1 << uint(i&31)
		}
	}
	return dst
}

// mask of valid bits [lo, hi) inside a word (0 <= lo < hi <= 32).
func WordMask(lo, hi int) uint32 {
	left := ^uint32(0) << lo
	right := ^uint32(0) >> (32 - hi)
	return left & right
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

func RandomSplitLSNNoiseCoeffF4(s uint32) ([]uint32, []uint32, []uint32, []uint32) {
	bit1Vec := make([]uint32, s)
	bitPVec := make([]uint32, s)
	bit1Inv := make([]uint32, s)
	bitPInv := make([]uint32, s)

	for i := uint32(0); i < s; i++ {
		r := rand.Intn(3) + 1 // pick 1,2,3 (nonzero)
		switch r {
		case 1: // element = 1
			bit1Vec[i] = 1
			bitPVec[i] = 0
			bit1Inv[i] = 1
			bitPInv[i] = 0
		case 2: // element = p
			bit1Vec[i] = 0
			bitPVec[i] = 1
			bit1Inv[i] = 1
			bitPInv[i] = 1
		case 3: // element = 1+p
			bit1Vec[i] = 1
			bitPVec[i] = 1
			bit1Inv[i] = 0
			bitPInv[i] = 1
		}
	}
	return bit1Vec, bitPVec, bit1Inv, bitPInv
}

func RandomPrimeFieldVector(vector []uint32, p uint32) {
	if dataobjects.USE_FAST_CODE {
		RandomizeVectorWithModulus(vector, uint32(len(vector)), 1, false, p)
	} else {
		for i := range vector {
			vector[i] = uint32(rand.Intn(int(p))) // Generates non-zero values in F_p
		}
	}
}

// TODO accept array instead of returning it - deferred since this is not used in MVP
func RandomNoiseVector(n uint32, epsi float32, p uint32) []uint32 {
	vector := dataobjects.AlignedMake[uint32](uint64(n))
	for i := range vector {
		if rand.Float32() <= epsi {
			vector[i] = uint32(rand.Intn(int(p-1))) + 1 // Generates non-zero values in F_p
		}
	}
	return vector
}

func RandomLPNNoiseVector(result []uint32, epsi float64, field dataobjects.Field) {
	if dataobjects.USE_FAST_CODE {
		lpn_noise_vector(result, 0, uint64(len(result)), epsi, field.Mod())
	} else {
		for i := range result {
			if rand.Float64() <= epsi {
				var val uint32
				for {
					val = SampleElement(field)
					if val != 0 {
						break
					}
				}
				result[i] = val
			}
		}
	}
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
		RandomizeVectorWithModulusAndSeed(data, uint32(dataSize), 1, false, p, seed)
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

package linearcode

import (
	"RandomLinearCodePIR/dataobjects"
	"RandomLinearCodePIR/utils"
	"math/rand"
)

// Return -P of C = (-P // I) flattened
func Generate1DDualMatrix(L, K uint32, field dataobjects.Field, seed int64) []uint32 {
	P := GenerateP(L, K, field, seed)

	vmatrix := dataobjects.AlignedMake[uint32](uint64(K * L))

	idx := 0

	for i := uint32(0); i < L; i++ {
		for j := uint32(0); j < K; j++ {
			vmatrix[idx] = field.Neg(P[i][j])
			idx += 1
		}
	}

	return vmatrix
}

func GenerateP(L, K uint32, field dataobjects.Field, seed int64) [][]uint32 {
	P := make([][]uint32, L)

	var rng *rand.Rand
	if dataobjects.USE_FAST_CODE {
		utils.RandomizeVectorWithSeed(nil, 0, seed)
	} else {
		rng = rand.New(rand.NewSource(seed))
	}

	for i := uint32(0); i < L; i++ {
		P[i] = dataobjects.AlignedMake[uint32](uint64(K))
		if dataobjects.USE_FAST_CODE {
			utils.RandomizeVectorWithModulus(P[i], K, field.Mod())
		} else {
			for j := uint32(0); j < K; j++ {
				P[i][j] = field.SampleElementWithSeed(rng)
			}
		}
	}

	return P
}

// Generate D = (I | P), transpose to D' = (I // P^T) and flatten
func Generate1DRLCMatrix(L, K uint32, p dataobjects.Field, seed int64) []uint32 {
	P := GenerateP(L, K, p, seed)

	vmatrix := dataobjects.AlignedMake[uint32](uint64(K * L))

	idx := 0

	for j := uint32(0); j < K; j++ {
		for i := uint32(0); i < L; i++ {
			vmatrix[idx] = P[i][j]
			idx += 1
		}
	}

	return vmatrix
}

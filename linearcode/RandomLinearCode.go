package linearcode

import (
	"RandomLinearCodePIR/dataobjects"
	"RandomLinearCodePIR/utils"
	"math/rand"
)

// Return -P of C = (-P // I) flattened
func Generate1DDualMatrix(L, K uint32, field dataobjects.Field, seed int64) []uint32 {
	P := GenerateP(L, K, field, seed)

	if dataobjects.USE_FAST_CODE {
		dataobjects.FieldNegVector(P, 0, uint64(K*L), field.Mod())
		return P
	} else {
		vmatrix := dataobjects.AlignedMake[uint32](uint64(K * L))

		idx := 0

		for i := uint32(0); i < L; i++ {
			for j := uint32(0); j < K; j++ {
				vmatrix[idx] = field.Neg(P[i*K+j])
				idx += 1
			}
		}

		return vmatrix
	}
}

func GenerateP(L, K uint32, field dataobjects.Field, seed int64) []uint32 {
	P := make([]uint32, L*K)

	var rng *rand.Rand
	if dataobjects.USE_FAST_CODE {
		utils.RandomizeVectorWithSeed(nil, 0, seed)
	} else {
		rng = rand.New(rand.NewSource(seed))
	}

	if dataobjects.USE_FAST_CODE {
		utils.RandomizeVectorWithModulus(P, L*K, field.Mod())
	} else {
		for i := uint32(0); i < L; i++ {
			for j := uint32(0); j < K; j++ {
				P[i*K+j] = field.SampleElementWithSeed(rng)
			}
		}
	}

	return P
}

// Generate D = (I | P), transpose to D' = (I // P^T) and flatten
func Generate1DRLCMatrix(L, K uint32, p dataobjects.Field, seed int64) []uint32 {
	P := GenerateP(L, K, p, seed)
	defer dataobjects.Aligned1DFree(P)

	vmatrix := dataobjects.AlignedMake[uint32](uint64(K) * uint64(L))

	idx := 0

	for j := uint32(0); j < K; j++ {
		for i := uint32(0); i < L; i++ {
			vmatrix[idx] = P[i*K+j]
			idx += 1
		}
	}

	return vmatrix
}

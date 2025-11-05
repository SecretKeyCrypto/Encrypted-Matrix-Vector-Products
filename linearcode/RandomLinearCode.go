package linearcode

import (
	"RandomLinearCodePIR/dataobjects"
	"RandomLinearCodePIR/utils"
	"context"
	"math/rand"
)

// Return -P of C = (-P // I) flattened
func Generate1DDualMatrix(ctx context.Context, L, K uint32, field dataobjects.Field, seed int64) []uint32 {
	P := GenerateP(L, K, field, seed)

	if dataobjects.USE_FAST_CODE {
		dataobjects.FieldNegVector(P, 0, uint64(K*L), field.Mod())
	} else {
		for i := range P {
			P[i] = field.Neg(P[i])
		}
	}

	return P
}

func generateP(L, K uint32, field dataobjects.Field, seed int64, transpose bool) []uint32 {
	P := dataobjects.AlignedMake[uint32](uint64(L) * uint64(K))

	var rng *rand.Rand
	if dataobjects.USE_FAST_CODE {
		utils.RandomizeVectorWithSeed(nil, 0, 1, transpose, seed)
	} else {
		rng = rand.New(rand.NewSource(seed))
	}

	if dataobjects.USE_FAST_CODE {
		utils.RandomizeVectorWithModulus(P, L, K, transpose, field.Mod())
	} else {
		if transpose {
			for j := uint32(0); j < L; j++ {
				for i := uint32(0); i < K; i++ {
					P[i*L+j] = utils.SampleElementWithSeed(field, rng)
				}
			}
		} else {
			for i := uint32(0); i < L; i++ {
				for j := uint32(0); j < K; j++ {
					P[i*K+j] = utils.SampleElementWithSeed(field, rng)
				}
			}
		}
	}

	return P
}

func GenerateP(L, K uint32, field dataobjects.Field, seed int64) []uint32 {
	return generateP(L, K, field, seed, false)
}

// Generate D = (I | P), transpose to D' = (I // P^T) and flatten
func Generate1DRLCMatrix(L, K uint32, field dataobjects.Field, seed int64) []uint32 {
	if dataobjects.USE_FAST_CODE {
		return generateP(L, K, field, seed, true)
	} else {
		P := GenerateP(L, K, field, seed)
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
}

package linearcode

import (
	"RandomLinearCodePIR/dataobjects"
	"RandomLinearCodePIR/utils"
	"context"
	"math/rand"
)

// Return -P of C = (-P // I) flattened
func Generate1DDualMatrix(ctx context.Context, L, K uint32, field dataobjects.Field, seed int64) []uint32 {
	doctx := dataobjects.GetDeferralDoContext(ctx)
	P := GenerateP(ctx, L, K, field, seed)

	if dataobjects.USE_FAST_CODE {
		dataobjects.FieldNegVector(doctx, P, 0, uint64(K*L), field.Mod())
	} else {
		for i := range P {
			P[i] = field.Neg(P[i])
		}
	}

	return P
}

func generateP(ctx context.Context, L, K uint32, field dataobjects.Field, seed int64, transpose bool) []uint32 {
	doctx := dataobjects.GetDeferralDoContext(ctx)
	P := dataobjects.DoAlignedMake[uint32](doctx, uint64(L)*uint64(K))

	if dataobjects.USE_FAST_CODE {
		utils.RandomizeVectorWithModulusAndSeed(doctx, P, L, K, transpose, false, field.Mod(), seed, 0)
	} else {
		rng := rand.New(rand.NewSource(seed))
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

func GenerateP(ctx context.Context, L, K uint32, field dataobjects.Field, seed int64) []uint32 {
	return generateP(ctx, L, K, field, seed, false)
}

// Generate D = (I | P), transpose to D' = (I // P^T) and flatten
func Generate1DRLCMatrix(ctx context.Context, L, K uint32, field dataobjects.Field, seed int64) []uint32 {
	if dataobjects.USE_FAST_CODE {
		return generateP(ctx, L, K, field, seed, true)
	} else {
		doctx := dataobjects.GetDeferralDoContext(ctx)
		frame := dataobjects.MakeDeferralFrame(ctx)
		defer frame.Close()

		P := GenerateP(ctx, L, K, field, seed)
		frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, P) })

		vmatrix := dataobjects.DoAlignedMake[uint32](doctx, uint64(K)*uint64(L))

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

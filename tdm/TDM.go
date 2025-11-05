package tdm

/*
#cgo CFLAGS: -I../TDM
#cgo LDFLAGS: -L../TDM -L/opt/homebrew/lib -lNTT -lntl -lgmp -lstdc++
#include "NTT.h"
*/
import "C"
import (
	"RandomLinearCodePIR/dataobjects"
	"RandomLinearCodePIR/utils"
	"context"
	"math"
	"math/rand"
)

const (
	USE_FAST_CODE_FOR_CIRCULANT = true
	ExpansionFactor             = 2
	SliceSeedShift              = 13758
)

type TDM struct {
	Ctx    context.Context
	M      uint32
	N      uint32
	Q      uint32
	SeedL  int64
	SeedC  int64
	SeedR  int64
	SeedPL int64
	SeedPR int64
	// Internal Use
	m      uint32
	n      uint32
	rootK  uint32
	root2K uint32
	block  uint32
}

func (td *TDM) GenerateTrapDooredMatrix(seedL, seedPL, seedC, seedPR, seedR int64) []uint32 {
	frame := dataobjects.MakeDeferralFrame(td.Ctx)
	defer frame.Close()

	td.updateInternalUseParams()
	fullTDM := dataobjects.AlignedMake[uint32](uint64(td.m) * uint64(td.n))
	blockTDM := dataobjects.AlignedMake[uint32](uint64(td.block) * uint64(td.block))
	frame.Defer(func() { dataobjects.Aligned1DFree(blockTDM) })

	for i := uint32(0); i < td.m/td.block; i++ {
		for j := uint32(0); j < td.n/td.block; j++ {
			seed := int64(i*td.m/td.block + j)
			td.GenerateBasicTrapDooredMatrix(blockTDM, seedL+seed, seedPL+seed, seedC+seed, seedPR+seed, seedR+seed)

			for k := uint32(0); k < td.block; k++ {
				i0 := i*td.block + k
				j0 := j * td.block
				if dataobjects.USE_FAST_CODE {
					dataobjects.FieldCopyVector(fullTDM, uint64(i0*td.n+j0), blockTDM, uint64(k*td.block), uint64(min(td.n, td.block)))
				} else {
					copy(fullTDM[i0*td.n+j0:(i0+1)*td.n], blockTDM[k*td.block:(k+1)*td.block])
				}
			}
		}
	}

	return fullTDM
}

// The basic Trapdoor matrix has the form R = S_L * Pi_L * S * Pi_R * S_R where it expands k x k matrix by factor of the ExpansionFactor (2)
func (td *TDM) GenerateBasicTrapDooredMatrix(result []uint32, seedL, seedPL, seedC, seedPR, seedR int64) {
	frame := dataobjects.MakeDeferralFrame(td.Ctx)
	defer frame.Close()

	permR := dataobjects.AlignedMake[uint32](uint64(ExpansionFactor * td.block))
	utils.GetPermutation(permR, ExpansionFactor*td.block, seedPR)
	frame.Defer(func() { dataobjects.Aligned1DFree(permR) })
	S_R := GetQuasiCyclicMatrix(td.Ctx, td.block, td.Q, seedR, permR)
	frame.Defer(func() { dataobjects.Aligned1DFree(S_R) })

	// R = S x perm(S_R)
	permL := dataobjects.AlignedMake[uint32](uint64(ExpansionFactor * td.block))
	utils.GetPermutation(permL, ExpansionFactor*td.block, seedPL)
	frame.Defer(func() { dataobjects.Aligned1DFree(permL) })
	R := dataobjects.AlignedMake[uint32](uint64(ExpansionFactor*td.block) * uint64(td.block))
	CirculantMatrixMul(td.Ctx, R, ExpansionFactor*td.block, td.Q, td.root2K, seedC, S_R, 2*td.block, td.block, permL)
	frame.Defer(func() { dataobjects.Aligned1DFree(R) })

	// S_L has the form [I | C]
	L := result
	CirculantMatrixMul(td.Ctx, L, td.block, td.Q, td.rootK, seedL, R[td.block*td.block:], (ExpansionFactor-1)*td.block, td.block, nil)
	if dataobjects.USE_FAST_CODE {
		dataobjects.FieldAddVectors(L, 0, R, 0, L, 0, uint64(td.block*td.block), td.Q)
	} else {
		for i := uint32(0); i < td.block; i++ {
			for j := uint32(0); j < td.block; j++ {
				L[i*td.block+j] = uint32((uint64(R[i*td.block+j]) + uint64(L[i*td.block+j])) % uint64(td.Q))
			}
		}
	}
}

func (td *TDM) GenerateFlattenedTrapDooredMatrix() []uint32 {
	frame := dataobjects.MakeDeferralFrame(td.Ctx)
	defer frame.Close()

	td.updateInternalUseParams()
	result := dataobjects.AlignedMake[uint32](uint64(td.M * td.N))
	R := td.GenerateTrapDooredMatrix(td.SeedL, td.SeedPL, td.SeedC, td.SeedPR, td.SeedR)
	frame.Defer(func() { dataobjects.Aligned1DFree(R) })

	// Only return the upper-left cornor of the TDM
	for i := uint32(0); i < td.M; i++ {
		if dataobjects.USE_FAST_CODE {
			dataobjects.FieldCopyVector(result, uint64(i*td.N), R, uint64(i*td.n), uint64(min(td.N, td.n)))
		} else {
			copy(result[i*td.N:(i+1)*td.N], R[i*td.n:(i+1)*td.n])
		}
	}
	return result
}

func (td *TDM) GenerateFlattenedTrapDooredMatrixPerSlice(sliceNum int64) []uint32 {
	frame := dataobjects.MakeDeferralFrame(td.Ctx)
	defer frame.Close()

	td.updateInternalUseParams()
	result := dataobjects.AlignedMake[uint32](uint64(td.M * td.N))
	R := td.GenerateTrapDooredMatrix(td.SeedL+sliceNum*SliceSeedShift,
		td.SeedPL+sliceNum*SliceSeedShift,
		td.SeedC+sliceNum*SliceSeedShift,
		td.SeedPR+sliceNum*SliceSeedShift,
		td.SeedR+sliceNum*SliceSeedShift)
	frame.Defer(func() { dataobjects.Aligned1DFree(R) })

	// Only return the upper-left cornor of the TDM
	for i := uint32(0); i < td.M; i++ {
		if dataobjects.USE_FAST_CODE {
			dataobjects.FieldCopyVector(result, uint64(i*td.N), R, uint64(i*td.n), uint64(min(td.N, td.n)))
		} else {
			copy(result[i*td.N:(i+1)*td.N], R[i*td.n:(i+1)*td.n])
		}
	}
	return result
}

func (td *TDM) EvaluationCircuit(v []uint32) []uint32 {
	return td.EvaluationCircuitPerSlice(v, 0)
}

func (td *TDM) EvaluationCircuitPerSliceInPlace(masks, bv, v []uint32, sliceNum int64) []uint32 {
	td.updateInternalUseParams()

	if !dataobjects.USE_FAST_CODE {
		if int(td.n) > len(v) {
			padded := dataobjects.AlignedMake[uint32](uint64(td.n))
			copy(padded, v)
			v = padded
		}
	}

	for j := uint32(0); j < td.n/td.block; j++ {
		if dataobjects.USE_FAST_CODE {
			vstart, vend, vlength, bvlength := uint64(j)*uint64(td.block), uint64(j+1)*uint64(td.block), uint64(len(v)), uint64(len(bv))
			if vstart < vlength {
				copylen := min(bvlength, vlength-vstart)
				dataobjects.FieldCopyVector(bv, 0, v, vstart, copylen)
				if vend > copylen && bvlength > copylen {
					dataobjects.FieldSetVector(bv, copylen, min(bvlength-copylen, vend-copylen), 0)
				}
			}
		} else {
			copy(bv, v[j*td.block:(j+1)*td.block])
		}
		for i := uint32(0); i < td.m/td.block; i++ {
			// Calculate the seed for each block, and use ECBasic to evaluate
			temp := td.EvaluationCircuitBasic(bv, int64(i*td.m/td.block+j)+sliceNum*SliceSeedShift)
			if dataobjects.USE_FAST_CODE {
				dataobjects.FieldAddVectors(masks, uint64(i*td.block), masks, uint64(i*td.block), temp, 0, uint64(td.block), td.Q)
			} else {
				for k := uint32(0); k < td.block; k++ {
					masks[i*td.block+k] = uint32((uint64(masks[i*td.block+k]) + uint64(temp[k])) % uint64(td.Q))
				}
			}
		}
	}

	// The slice starting from 0 is critical for it to be correctly freed
	return masks[0:td.M]
}

func (td *TDM) AllocateMaskForEvaluationCircuitPerSlice() []uint32 {
	td.updateInternalUseParams()
	return dataobjects.AlignedMake[uint32](uint64(td.m))
}

func (td *TDM) AllocateBlockVectorForEvaluationCircuitPerSlice() []uint32 {
	td.updateInternalUseParams()
	return dataobjects.AlignedMake[uint32](uint64(td.block))
}

func (td *TDM) EvaluationCircuitPerSlice(v []uint32, sliceNum int64) []uint32 {
	frame := dataobjects.MakeDeferralFrame(td.Ctx)
	defer frame.Close()

	masks := td.AllocateMaskForEvaluationCircuitPerSlice()
	bv := td.AllocateBlockVectorForEvaluationCircuitPerSlice()
	frame.Defer(func() { dataobjects.Aligned1DFree(bv) })

	return td.EvaluationCircuitPerSliceInPlace(masks, bv, v, sliceNum)
}

func (td *TDM) EvaluationCircuitBasic(v []uint32, addOnSeed int64) []uint32 {
	frame := dataobjects.MakeDeferralFrame(td.Ctx)
	defer frame.Close()

	// S_R = [I | C] x v
	resR := dataobjects.AlignedMake[uint32](uint64(ExpansionFactor * td.block))
	frame.Defer(func() { dataobjects.Aligned1DFree(resR) })
	vecR := CirculantVectorMul(td.Ctx, td.block, td.Q, td.rootK, td.SeedR+addOnSeed, v, nil)
	frame.Defer(func() { dataobjects.Aligned1DFree(vecR) })

	// Apply PermR
	permR := dataobjects.AlignedMake[uint32](uint64(ExpansionFactor * td.block))
	utils.GetPermutation(permR, ExpansionFactor*td.block, td.SeedPR+addOnSeed)
	frame.Defer(func() { dataobjects.Aligned1DFree(permR) })
	i := uint32(0)
	for ; i < td.block; i++ {
		ii := permR[i]
		resR[ii] = v[i]
	}
	for ; i < ExpansionFactor*td.block; i++ {
		ii := permR[i]
		ix := i - td.block
		resR[ii] = vecR[ix]
	}

	// Multiply by S
	permC := dataobjects.AlignedMake[uint32](uint64(ExpansionFactor * td.block))
	utils.GetPermutation(permC, ExpansionFactor*td.block, td.SeedPL+addOnSeed)
	frame.Defer(func() { dataobjects.Aligned1DFree(permC) })
	resC := CirculantVectorMul(td.Ctx, ExpansionFactor*td.block, td.Q, td.root2K, td.SeedC+addOnSeed, resR, permC)

	// S_L = [I // C] x resC
	vecC := CirculantVectorMul(td.Ctx, td.block, td.Q, td.rootK, td.SeedL+addOnSeed, resC[td.block:], nil)
	frame.Defer(func() { dataobjects.Aligned1DFree(vecC) })
	if dataobjects.USE_FAST_CODE {
		dataobjects.FieldAddVectors(resC, 0, resC, 0, vecC, 0, uint64(td.block), td.Q)
	} else {
		for i := uint32(0); i < td.block; i++ {
			resC[i] = uint32((uint64(resC[i]) + uint64(vecC[i])) % uint64(td.Q))
		}
	}

	return resC[:td.block]
}

func CirculantMatrixMul(ctx context.Context, result []uint32, blockSize, q, root uint32, seed int64, mat []uint32, matRows, matCols uint32, perm []uint32) {
	frame := dataobjects.MakeDeferralFrame(ctx)
	defer frame.Close()

	polyQC := dataobjects.AlignedMake[uint32](uint64(blockSize))
	frame.Defer(func() { dataobjects.Aligned1DFree(polyQC) })
	res := dataobjects.AlignedMake[uint32](uint64(blockSize))
	frame.Defer(func() { dataobjects.Aligned1DFree(res) })

	if dataobjects.USE_FAST_CODE && USE_FAST_CODE_FOR_CIRCULANT {
		utils.RandomizeVectorWithModulusAndSeed(polyQC, blockSize, 1, false, q, seed)
		for t := uint32(1); t < blockSize/2; t++ {
			polyQC[t], polyQC[blockSize-t] = polyQC[blockSize-t], polyQC[t]
		}
	} else {
		rng := rand.New(rand.NewSource(seed))
		polyQC[0] = uint32(rng.Intn(int(q)))
		for t := uint32(1); t < blockSize; t++ {
			polyQC[blockSize-t] = uint32(rng.Intn(int(q)))
		}
	}

	v := dataobjects.AlignedMake[uint32](uint64(blockSize))
	frame.Defer(func() { dataobjects.Aligned1DFree(v) })
	tmpA := dataobjects.AlignedMake[uint32](uint64(blockSize))
	frame.Defer(func() { dataobjects.Aligned1DFree(tmpA) })
	tmpB := dataobjects.AlignedMake[uint32](uint64(blockSize))
	frame.Defer(func() { dataobjects.Aligned1DFree(tmpB) })

	for j := uint32(0); j < matCols; j++ {
		for i := uint32(0); i < matRows; i++ {
			v[i] = mat[i*matCols+j]
		}
		NTT_Convolution(polyQC, v, tmpA, tmpB, res, blockSize, root, q)
		if perm == nil {
			for i := uint32(0); i < uint32(len(res)); i++ {
				result[i*matCols+j] = res[i]
			}
		} else {
			for i := uint32(0); i < uint32(len(res)); i++ {
				ii := perm[i]
				result[ii*matCols+j] = res[i]
			}
		}
	}
}

func CirculantVectorMul(ctx context.Context, blockSize, q, root uint32, seed int64, v, perm []uint32) []uint32 {
	frame := dataobjects.MakeDeferralFrame(ctx)
	defer frame.Close()

	polyQC := dataobjects.AlignedMake[uint32](uint64(blockSize))
	frame.Defer(func() { dataobjects.Aligned1DFree(polyQC) })

	if dataobjects.USE_FAST_CODE && USE_FAST_CODE_FOR_CIRCULANT {
		utils.RandomizeVectorWithModulusAndSeed(polyQC, blockSize, 1, false, q, seed)
		for t := uint32(1); t < blockSize/2; t++ {
			polyQC[t], polyQC[blockSize-t] = polyQC[blockSize-t], polyQC[t]
		}
	} else {
		rng := rand.New(rand.NewSource(seed))
		polyQC[0] = uint32(rng.Intn(int(q)))
		for t := uint32(1); t < blockSize; t++ {
			polyQC[blockSize-t] = uint32(rng.Intn(int(q)))
		}
	}

	tmpA := dataobjects.AlignedMake[uint32](uint64(blockSize))
	frame.Defer(func() { dataobjects.Aligned1DFree(tmpA) })
	tmpB := dataobjects.AlignedMake[uint32](uint64(blockSize))
	frame.Defer(func() { dataobjects.Aligned1DFree(tmpB) })
	conv := dataobjects.AlignedMake[uint32](uint64(blockSize))
	NTT_Convolution(polyQC, v, tmpA, tmpB, conv, blockSize, root, q)

	if perm == nil {
		return conv
	} else {
		frame.Defer(func() { dataobjects.Aligned1DFree(conv) })
		result := dataobjects.AlignedMake[uint32](uint64(blockSize))
		for i := uint32(0); i < blockSize; i++ {
			ii := perm[i]
			result[ii] = conv[i]
		}
		return result
	}
}

// Q has the form [I // C] where C is a circulant matrix
func GetQuasiCyclicMatrix(ctx context.Context, blockSize, q uint32, seed int64, perm []uint32) []uint32 {
	frame := dataobjects.MakeDeferralFrame(ctx)
	defer frame.Close()

	row := 2 * blockSize
	Q := dataobjects.AlignedMake[uint32](uint64(row) * uint64(blockSize))

	S := GetCirculantMatrix(ctx, blockSize, q, seed)
	frame.Defer(func() { dataobjects.Aligned1DFree(S) })

	for i := uint32(0); i < blockSize; i++ {
		ii := perm[i]
		Q[ii*blockSize+i] = 1
	}
	for i := uint32(0); i < blockSize; i++ {
		ii := perm[i+blockSize]
		if dataobjects.USE_FAST_CODE {
			dataobjects.FieldCopyVector(Q, uint64(ii*blockSize), S, uint64(i*blockSize), uint64(blockSize))
		} else {
			copy(Q[ii*blockSize:(ii+1)*blockSize], S[i*blockSize:(i+1)*blockSize])
		}
	}

	return Q
}

func GetCirculantMatrix(ctx context.Context, k, q uint32, seed int64) []uint32 {
	frame := dataobjects.MakeDeferralFrame(ctx)
	defer frame.Close()

	S := dataobjects.AlignedMake[uint32](uint64(k) * uint64(k))

	poly := dataobjects.AlignedMake[uint32](uint64(k))
	frame.Defer(func() { dataobjects.Aligned1DFree(poly) })
	if dataobjects.USE_FAST_CODE && USE_FAST_CODE_FOR_CIRCULANT {
		utils.RandomizeVectorWithModulusAndSeed(poly, k, 1, false, q, seed)
	} else {
		rng := rand.New(rand.NewSource(seed))
		for t := uint32(0); t < k; t++ {
			poly[t] = uint32(rng.Intn(int(q)))
		}
	}

	if dataobjects.USE_FAST_CODE {
		for t := uint32(0); t < k; t++ {
			dataobjects.FieldCopyVector(S, uint64(t*k+t), poly, 0, uint64(k-t))
			dataobjects.FieldCopyVector(S, uint64(t*k), poly, uint64(k-t), uint64(t))
		}
	} else {
		for i := uint32(0); i < k; i++ {
			for t := uint32(0); t < k; t++ {
				copy(S[t*k+t:t*k+k], poly[0:k-t])
				copy(S[t*k+0:t*k+t], poly[k-t:k])
			}
		}
	}

	return S
}

func (td *TDM) updateInternalUseParams() {
	if td.m != 0 {
		return
	}

	td.block = td.determineBlockSize(td.M, td.N)
	td.m = utils.RoundUp(td.M, td.block)
	td.n = utils.RoundUp(td.N, td.block)

	td.rootK = NthRootOfUnity(td.Q, td.block)
	td.root2K = NthRootOfUnity(td.Q, td.block*2)
}

func roundUpToPowerOf2(m uint32) uint32 {
	return uint32(1) << uint32(math.Ceil(math.Log2(float64(m))))
}

// Currently hardcode it to be max(2^(ceil(log2(min(m,n)))), (q-1)/2) for m x n matrix
// TODO: update for m,n,q in general
func (td *TDM) determineBlockSize(m, n uint32) uint32 {
	minOfMN := min(m, n)
	if minOfMN >= (td.Q-1)/2 {
		return (td.Q - 1) / 2
	}

	return uint32(1) << uint32(math.Ceil(math.Log2(float64(minOfMN))))
}

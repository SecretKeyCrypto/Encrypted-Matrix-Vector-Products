package tdm

/*
#cgo CFLAGS: -I../TDM
#cgo LDFLAGS: -L../TDM -L/opt/homebrew/lib -lNTT -lstdc++
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
	ExpansionFactor = 2
	SliceSeedShift  = 13758
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
	doctx := dataobjects.GetDeferralDoContext(td.Ctx)
	frame := dataobjects.MakeDeferralFrame(td.Ctx)
	defer frame.Close()

	td.updateInternalUseParams()
	fullTDM := dataobjects.DoAlignedMake[uint32](doctx, uint64(td.m)*uint64(td.n))
	blockTDM := dataobjects.DoAlignedMake[uint32](doctx, uint64(td.block)*uint64(td.block))
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, blockTDM) })

	// NOTE - rely on a small number of iterations here
	for i := uint32(0); i < td.m/td.block; i++ {
		for j := uint32(0); j < td.n/td.block; j++ {
			seed := int64(i*td.m/td.block + j)
			td.GenerateBasicTrapDooredMatrix(blockTDM, seedL+seed, seedPL+seed, seedC+seed, seedPR+seed, seedR+seed)

			if dataobjects.USE_FAST_CODE {
				PermutedExtentsAssign(doctx, fullTDM, (i+j)*td.block, td.n, 0, blockTDM, 0, td.block, 0, uint64(min(td.n, td.block)), nil, 0, uint64(td.block))
			} else {
				for k := uint32(0); k < td.block; k++ {
					i0 := i*td.block + k
					j0 := j * td.block
					if dataobjects.USE_FAST_CODE {
						dataobjects.FieldCopyVector(doctx, fullTDM, uint64(i0*td.n+j0), blockTDM, uint64(k*td.block), uint64(min(td.n, td.block)))
					} else {
						copy(fullTDM[i0*td.n+j0:(i0+1)*td.n], blockTDM[k*td.block:(k+1)*td.block])
					}
				}
			}
		}
	}

	return fullTDM
}

// The basic Trapdoor matrix has the form R = S_L * Pi_L * S * Pi_R * S_R where it expands k x k matrix by factor of the ExpansionFactor (2)
func (td *TDM) GenerateBasicTrapDooredMatrix(result []uint32, seedL, seedPL, seedC, seedPR, seedR int64) {
	doctx := dataobjects.GetDeferralDoContext(td.Ctx)
	frame := dataobjects.MakeDeferralFrame(td.Ctx)
	defer frame.Close()

	permR := dataobjects.DoAlignedMake[uint32](doctx, uint64(ExpansionFactor*td.block))
	utils.GetPermutation(doctx, permR, ExpansionFactor*td.block, seedPR, 0)
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, permR) })
	S_R := GetQuasiCyclicMatrix(td.Ctx, td.block, td.Q, seedR, permR)
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, S_R) })

	// R = S x perm(S_R)
	permL := dataobjects.DoAlignedMake[uint32](doctx, uint64(ExpansionFactor*td.block))
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, permL) })
	utils.GetPermutation(doctx, permL, ExpansionFactor*td.block, seedPL, 0)
	R := dataobjects.DoAlignedMake[uint32](doctx, uint64(ExpansionFactor*td.block)*uint64(td.block))
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, R) })
	CirculantMatrixMul(td.Ctx, R, ExpansionFactor*td.block, td.Q, td.root2K, seedC, S_R, 0, 2*td.block, td.block, permL)

	// S_L has the form [I | C]
	L := result
	CirculantMatrixMul(td.Ctx, L, td.block, td.Q, td.rootK, seedL, R, td.block*td.block, (ExpansionFactor-1)*td.block, td.block, nil)
	if dataobjects.USE_FAST_CODE {
		dataobjects.FieldAddVectors(doctx, L, 0, R, 0, L, 0, uint64(td.block*td.block), td.Q)
	} else {
		for i := uint32(0); i < td.block; i++ {
			for j := uint32(0); j < td.block; j++ {
				L[i*td.block+j] = uint32((uint64(R[i*td.block+j]) + uint64(L[i*td.block+j])) % uint64(td.Q))
			}
		}
	}
}

func (td *TDM) GenerateFlattenedTrapDooredMatrix() []uint32 {
	doctx := dataobjects.GetDeferralDoContext(td.Ctx)
	frame := dataobjects.MakeDeferralFrame(td.Ctx)
	defer frame.Close()

	td.updateInternalUseParams()
	result := dataobjects.DoAlignedMake[uint32](doctx, uint64(td.M*td.N))
	R := td.GenerateTrapDooredMatrix(td.SeedL, td.SeedPL, td.SeedC, td.SeedPR, td.SeedR)
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, R) })

	// Only return the upper-left cornor of the TDM
	if dataobjects.USE_FAST_CODE {
		PermutedExtentsAssign(doctx, result, 0, td.N, 0, R, 0, td.n, 0, uint64(min(td.N, td.n)), nil, 0, uint64(td.M))
	} else {
		for i := uint32(0); i < td.M; i++ {
			if dataobjects.USE_FAST_CODE {
				dataobjects.FieldCopyVector(doctx, result, uint64(i*td.N), R, uint64(i*td.n), uint64(min(td.N, td.n)))
			} else {
				copy(result[i*td.N:(i+1)*td.N], R[i*td.n:(i+1)*td.n])
			}
		}
	}
	return result
}

func (td *TDM) GenerateFlattenedTrapDooredMatrixPerSlice(sliceNum int64) []uint32 {
	doctx := dataobjects.GetDeferralDoContext(td.Ctx)
	frame := dataobjects.MakeDeferralFrame(td.Ctx)
	defer frame.Close()

	td.updateInternalUseParams()
	result := dataobjects.DoAlignedMake[uint32](doctx, uint64(td.M*td.N))
	R := td.GenerateTrapDooredMatrix(td.SeedL+sliceNum*SliceSeedShift,
		td.SeedPL+sliceNum*SliceSeedShift,
		td.SeedC+sliceNum*SliceSeedShift,
		td.SeedPR+sliceNum*SliceSeedShift,
		td.SeedR+sliceNum*SliceSeedShift)
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, R) })

	// Only return the upper-left cornor of the TDM
	if dataobjects.USE_FAST_CODE {
		PermutedExtentsAssign(doctx, result, 0, td.N, 0, R, 0, td.n, 0, uint64(min(td.N, td.n)), nil, 0, uint64(td.M))
	} else {
		for i := uint32(0); i < td.M; i++ {
			if dataobjects.USE_FAST_CODE {
				dataobjects.FieldCopyVector(doctx, result, uint64(i*td.N), R, uint64(i*td.n), uint64(min(td.N, td.n)))
			} else {
				copy(result[i*td.N:(i+1)*td.N], R[i*td.n:(i+1)*td.n])
			}
		}
	}
	return result
}

func (td *TDM) EvaluationCircuit(v []uint32) []uint32 {
	return td.EvaluationCircuitPerSlice(v, 0)
}

func (td *TDM) EvaluationCircuitPerSliceInPlace(masks []uint32, mo uint32, bv []uint32, bo, bvlen uint32, v []uint32, vo, vlen uint32, sliceNum int64) []uint32 {
	doctx := dataobjects.GetDeferralDoContext(td.Ctx)
	td.updateInternalUseParams()

	if !dataobjects.USE_FAST_CODE {
		if td.n > vlen {
			padded := dataobjects.DoAlignedMake[uint32](doctx, uint64(td.n))
			copy(padded, v)
			v = padded
		}
	}

	// TODO - rely on a small number of iterations here - EvaluationCircuitBasic is not easy to lift
	for j := uint32(0); j < td.n/td.block; j++ {
		if dataobjects.USE_FAST_CODE {
			vstart, vend, vlength, bvlength := uint64(j)*uint64(td.block), uint64(j+1)*uint64(td.block), uint64(vlen), uint64(bvlen)
			if vstart < vlength {
				copylen := min(bvlength, vlength-vstart)
				dataobjects.FieldCopyVector(doctx, bv, uint64(bo), v, uint64(vo)+vstart, copylen)
				if vend > copylen && bvlength > copylen {
					dataobjects.FieldSetVector(doctx, bv, uint64(bo)+copylen, min(bvlength-copylen, vend-copylen), 0)
				}
			}
		} else {
			copy(bv, v[vo+j*td.block:(j+1)*td.block])
		}
		for i := uint32(0); i < td.m/td.block; i++ {
			// Calculate the seed for each block, and use ECBasic to evaluate
			temp := td.EvaluationCircuitBasic(bv, bo, int64(i*td.m/td.block+j)+sliceNum*SliceSeedShift)
			if dataobjects.USE_FAST_CODE {
				dataobjects.FieldAddVectors(doctx, masks, uint64(mo+i*td.block), masks, uint64(mo+i*td.block), temp, 0, uint64(td.block), td.Q)
			} else {
				for k := uint32(0); k < td.block; k++ {
					masks[mo+i*td.block+k] = uint32((uint64(masks[mo+i*td.block+k]) + uint64(temp[k])) % uint64(td.Q))
				}
			}
		}
	}

	return masks[mo : mo+td.M]
}

func (td *TDM) AllocateMaskForEvaluationCircuitPerSlice(count uint32) []uint32 {
	doctx := dataobjects.GetDeferralDoContext(td.Ctx)
	td.updateInternalUseParams()
	return dataobjects.DoAlignedMake[uint32](doctx, uint64(td.m*count))
}

func (td *TDM) AllocateBlockVectorForEvaluationCircuitPerSlice(count uint32) []uint32 {
	doctx := dataobjects.GetDeferralDoContext(td.Ctx)
	td.updateInternalUseParams()
	return dataobjects.DoAlignedMake[uint32](doctx, uint64(td.block*count))
}

func (td *TDM) EvaluationCircuitPerSlice(v []uint32, sliceNum int64) []uint32 {
	doctx := dataobjects.GetDeferralDoContext(td.Ctx)
	frame := dataobjects.MakeDeferralFrame(td.Ctx)
	defer frame.Close()

	masks := td.AllocateMaskForEvaluationCircuitPerSlice(1)
	bv := td.AllocateBlockVectorForEvaluationCircuitPerSlice(1)
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, bv) })

	return td.EvaluationCircuitPerSliceInPlace(masks, 0, bv, 0, uint32(len(bv)), v, 0, uint32(len(v)), sliceNum)
}

func (td *TDM) EvaluationCircuitBasic(v []uint32, vo uint32, addOnSeed int64) []uint32 {
	doctx := dataobjects.GetDeferralDoContext(td.Ctx)
	frame := dataobjects.MakeDeferralFrame(td.Ctx)
	defer frame.Close()

	// S_R = [I | C] x v
	resR := dataobjects.DoAlignedMake[uint32](doctx, uint64(ExpansionFactor*td.block))
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, resR) })
	vecR := CirculantVectorMul(td.Ctx, td.block, td.Q, td.rootK, td.SeedR+addOnSeed, v, vo, nil)
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, vecR) })

	// Apply PermR
	permR := dataobjects.DoAlignedMake[uint32](doctx, uint64(ExpansionFactor*td.block))
	utils.GetPermutation(doctx, permR, ExpansionFactor*td.block, td.SeedPR+addOnSeed, 0)
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, permR) })
	if dataobjects.USE_FAST_CODE {
		PermutedExtentsAssign(doctx, resR, 0, 0, 1, v, vo, 1, 0, 1, permR, 0, uint64(td.block))
		PermutedExtentsAssign(doctx, resR, 0, 0, 1, vecR, 0, 1, 0, 1, permR, td.block, uint64((ExpansionFactor-1)*td.block))
	} else {
		i := uint32(0)
		for ; i < td.block; i++ {
			ii := permR[i]
			resR[ii] = v[vo+i]
		}
		for ; i < ExpansionFactor*td.block; i++ {
			ii := permR[i]
			ix := i - td.block
			resR[ii] = vecR[ix]
		}
	}

	// Multiply by S
	permC := dataobjects.DoAlignedMake[uint32](doctx, uint64(ExpansionFactor*td.block))
	utils.GetPermutation(doctx, permC, ExpansionFactor*td.block, td.SeedPL+addOnSeed, 0)
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, permC) })
	resC := CirculantVectorMul(td.Ctx, ExpansionFactor*td.block, td.Q, td.root2K, td.SeedC+addOnSeed, resR, 0, permC)

	// S_L = [I // C] x resC
	vecC := CirculantVectorMul(td.Ctx, td.block, td.Q, td.rootK, td.SeedL+addOnSeed, resC, td.block, nil)
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, vecC) })
	if dataobjects.USE_FAST_CODE {
		dataobjects.FieldAddVectors(doctx, resC, 0, resC, 0, vecC, 0, uint64(td.block), td.Q)
	} else {
		for i := uint32(0); i < td.block; i++ {
			resC[i] = uint32((uint64(resC[i]) + uint64(vecC[i])) % uint64(td.Q))
		}
	}

	return resC[:td.block]
}

func CirculantMatrixMul(ctx context.Context, result []uint32, blockSize, q, root uint32, seed int64, mat []uint32, matOffset, matRows, matCols uint32, perm []uint32) {
	doctx := dataobjects.GetDeferralDoContext(ctx)
	frame := dataobjects.MakeDeferralFrame(ctx)
	defer frame.Close()

	polyQC := dataobjects.DoAlignedMake[uint32](doctx, uint64(blockSize))
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, polyQC) })

	if dataobjects.USE_FAST_CODE {
		utils.RandomizeVectorWithModulusAndSeed(doctx, polyQC, 1, blockSize, false, true, q, seed, 0)
	} else {
		rng := rand.New(rand.NewSource(seed))
		polyQC[0] = uint32(rng.Intn(int(q)))
		for t := uint32(1); t < blockSize; t++ {
			polyQC[blockSize-t] = uint32(rng.Intn(int(q)))
		}
	}

	if dataobjects.USE_FAST_CODE {
		res := dataobjects.DoAlignedMake[uint32](doctx, uint64(blockSize*matCols))
		frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, res) })
		resT := dataobjects.DoAlignedMake[uint32](doctx, uint64(blockSize*matCols))
		frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, resT) })
		v := dataobjects.DoAlignedMake[uint32](doctx, uint64(blockSize*matCols))
		frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, v) })

		dataobjects.MatrixTranspose(doctx, v, 0, mat, matOffset, matRows, matCols)
		NTT_Convolution(doctx, polyQC, 0, 0, v, 0, blockSize, resT, 0, blockSize, blockSize, matCols, root, q)
		dataobjects.MatrixTranspose(doctx, res, 0, resT, 0, matCols, blockSize)
		PermutedExtentsAssign(doctx, result, 0, 0, matCols, res, 0, matCols, 0, uint64(matCols), perm, 0, uint64(blockSize))
	} else {
		res := dataobjects.DoAlignedMake[uint32](doctx, uint64(blockSize))
		frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, res) })
		v := dataobjects.DoAlignedMake[uint32](doctx, uint64(blockSize))
		frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, v) })

		for j := uint32(0); j < matCols; j++ {
			if dataobjects.USE_FAST_CODE {
				PermutedExtentsAssign(doctx, v, 0, 1, 0, mat, matOffset+j, matCols, 0, 1, nil, 0, uint64(matRows))
			} else {
				for i := uint32(0); i < matRows; i++ {
					v[i] = mat[i*matCols+j+matOffset]
				}
			}
			NTT_Convolution(doctx, polyQC, 0, 0, v, 0, 0, res, 0, 0, blockSize, 1, root, q)
			if dataobjects.USE_FAST_CODE {
				PermutedExtentsAssign(doctx, result, j, 0, matCols, res, 0, 1, 0, 1, perm, 0, uint64(len(res)))
			} else {
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
	}
}

func CirculantVectorMul(ctx context.Context, blockSize, q, root uint32, seed int64, v []uint32, vo uint32, perm []uint32) []uint32 {
	doctx := dataobjects.GetDeferralDoContext(ctx)
	frame := dataobjects.MakeDeferralFrame(ctx)
	defer frame.Close()

	polyQC := dataobjects.DoAlignedMake[uint32](doctx, uint64(blockSize))
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, polyQC) })

	if dataobjects.USE_FAST_CODE {
		utils.RandomizeVectorWithModulusAndSeed(doctx, polyQC, 1, blockSize, false, true, q, seed, 0)
	} else {
		rng := rand.New(rand.NewSource(seed))
		polyQC[0] = uint32(rng.Intn(int(q)))
		for t := uint32(1); t < blockSize; t++ {
			polyQC[blockSize-t] = uint32(rng.Intn(int(q)))
		}
	}

	conv := dataobjects.DoAlignedMake[uint32](doctx, uint64(blockSize))
	NTT_Convolution(doctx, polyQC, 0, 0, v, vo, 0, conv, 0, 0, blockSize, 1, root, q)

	if perm == nil {
		return conv
	} else {
		frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, conv) })
		result := dataobjects.DoAlignedMake[uint32](doctx, uint64(blockSize))
		if dataobjects.USE_FAST_CODE {
			PermutedExtentsAssign(doctx, result, 0, 0, 1, conv, 0, 1, 0, 1, perm, 0, uint64(blockSize))
		} else {
			for i := uint32(0); i < blockSize; i++ {
				ii := perm[i]
				result[ii] = conv[i]
			}
		}
		return result
	}
}

// Q has the form [I // C] where C is a circulant matrix
func GetQuasiCyclicMatrix(ctx context.Context, blockSize, q uint32, seed int64, perm []uint32) []uint32 {
	doctx := dataobjects.GetDeferralDoContext(ctx)
	frame := dataobjects.MakeDeferralFrame(ctx)
	defer frame.Close()

	row := 2 * blockSize
	Q := dataobjects.DoAlignedMake[uint32](doctx, uint64(row)*uint64(blockSize))

	S := GetCirculantMatrix(ctx, blockSize, q, seed)
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, S) })

	if dataobjects.USE_FAST_CODE {
		PermutedExtentsAssign(doctx, Q, 0, 1, blockSize, nil, 0, 0, 1, 1, perm, 0, uint64(blockSize))
	} else {
		for i := uint32(0); i < blockSize; i++ {
			ii := perm[i]
			Q[ii*blockSize+i] = 1
		}
	}
	if dataobjects.USE_FAST_CODE {
		PermutedExtentsAssign(doctx, Q, 0, 0, blockSize, S, 0, blockSize, 0, uint64(blockSize), perm, blockSize, uint64(blockSize))
	} else {
		for i := uint32(0); i < blockSize; i++ {
			ii := perm[i+blockSize]
			if dataobjects.USE_FAST_CODE {
				dataobjects.FieldCopyVector(doctx, Q, uint64(ii*blockSize), S, uint64(i*blockSize), uint64(blockSize))
			} else {
				copy(Q[ii*blockSize:(ii+1)*blockSize], S[i*blockSize:(i+1)*blockSize])
			}
		}
	}

	return Q
}

func GetCirculantMatrix(ctx context.Context, k, q uint32, seed int64) []uint32 {
	doctx := dataobjects.GetDeferralDoContext(ctx)
	frame := dataobjects.MakeDeferralFrame(ctx)
	defer frame.Close()

	S := dataobjects.DoAlignedMake[uint32](doctx, uint64(k)*uint64(k))

	poly := dataobjects.DoAlignedMake[uint32](doctx, uint64(k))
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, poly) })
	if dataobjects.USE_FAST_CODE {
		utils.RandomizeVectorWithModulusAndSeed(doctx, poly, k, 1, false, false, q, seed, 0)
	} else {
		rng := rand.New(rand.NewSource(seed))
		for t := uint32(0); t < k; t++ {
			poly[t] = uint32(rng.Intn(int(q)))
		}
	}

	if dataobjects.USE_FAST_CODE {
		CircularCopy(doctx, S, poly, uint64(k))
	} else {
		if dataobjects.USE_FAST_CODE {
			for t := uint32(0); t < k; t++ {
				dataobjects.FieldCopyVector(doctx, S, uint64(t*k+t), poly, 0, uint64(k-t))
				dataobjects.FieldCopyVector(doctx, S, uint64(t*k), poly, uint64(k-t), uint64(t))
			}
		} else {
			for i := uint32(0); i < k; i++ {
				for t := uint32(0); t < k; t++ {
					copy(S[t*k+t:t*k+k], poly[0:k-t])
					copy(S[t*k+0:t*k+t], poly[k-t:k])
				}
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

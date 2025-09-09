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
	"math"
	"math/rand"
)

const (
	USE_FAST_CODE_FOR_CIRCULANT = true
	ExpansionFactor             = 2
	SliceSeedShift              = 13758
)

type TDM struct {
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

func (td *TDM) GenerateTrapDooredMatrix(seedL, seedPL, seedC, seedPR, seedR int64) [][]uint32 {
	td.updateInternalUseParams()
	fullTDM := make([][]uint32, td.m)
	for i := range fullTDM {
		fullTDM[i] = dataobjects.AlignedMake[uint32](uint64(td.n))
	}

	for i := uint32(0); i < td.m/td.block; i++ {
		for j := uint32(0); j < td.n/td.block; j++ {
			seed := int64(i*td.m/td.block + j)
			blockTDM := td.GenerateBasicTrapDooredMatrix(seedL+seed, seedPL+seed, seedC+seed, seedPR+seed, seedR+seed)

			for k := uint32(0); k < td.block; k++ {
				copy(fullTDM[i*td.block+k][j*td.block:], blockTDM[k])
			}
		}
	}

	return fullTDM
}

// The basic Trapdoor matrix has the form R = S_L * Pi_L * S * Pi_R * S_R where it expands k x k matrix by factor of the ExpansionFactor (2)
func (td *TDM) GenerateBasicTrapDooredMatrix(seedL, seedPL, seedC, seedPR, seedR int64) [][]uint32 {
	S_R := GetQuasiCyclicMatrix(td.block, td.Q, seedR)

	permR := GetPermutation(ExpansionFactor*td.block, seedPR)
	PermuteRowsInPlace(S_R, permR)

	// R = S x perm(S_R)
	R := CirculantMatrixMul(ExpansionFactor*td.block, td.Q, td.root2K, seedC, S_R)

	permL := GetPermutation(ExpansionFactor*td.block, seedPL)
	PermuteRowsInPlace(R, permL)

	// S_L has the form [I | C]
	L := CirculantMatrixMul(td.block, td.Q, td.rootK, seedL, R[td.block:])
	if dataobjects.USE_FAST_CODE {
		for i := uint32(0); i < td.block; i++ {
			dataobjects.FieldAddVectors(L[i], 0, R[i], 0, L[i], 0, uint64(td.block), td.Q)
		}
	} else {
		for i := uint32(0); i < td.block; i++ {
			for j := uint32(0); j < td.block; j++ {
				L[i][j] = uint32((uint64(R[i][j]) + uint64(L[i][j])) % uint64(td.Q))
			}
		}
	}

	return L
}

func (td *TDM) GenerateFlattenedTrapDooredMatrix() []uint32 {
	result := dataobjects.AlignedMake[uint32](uint64(td.M * td.N))
	R := td.GenerateTrapDooredMatrix(td.SeedL, td.SeedPL, td.SeedC, td.SeedPR, td.SeedR)

	// Only return the upper-left cornor of the TDM
	for i := uint32(0); i < td.M; i++ {
		copy(result[i*td.N:(i+1)*td.N], R[i])
	}
	return result
}

func (td *TDM) GenerateFlattenedTrapDooredMatrixPerSlice(sliceNum int64) []uint32 {
	result := dataobjects.AlignedMake[uint32](uint64(td.M * td.N))
	R := td.GenerateTrapDooredMatrix(td.SeedL+sliceNum*SliceSeedShift,
		td.SeedPL+sliceNum*SliceSeedShift,
		td.SeedC+sliceNum*SliceSeedShift,
		td.SeedPR+sliceNum*SliceSeedShift,
		td.SeedR+sliceNum*SliceSeedShift)

	// Only return the upper-left cornor of the TDM
	for i := uint32(0); i < td.M; i++ {
		copy(result[i*td.N:(i+1)*td.N], R[i])
	}
	return result
}

func (td *TDM) EvaluationCircuit(v []uint32) []uint32 {
	return td.EvaluationCircuitPerSlice(v, 0)
}

func (td *TDM) EvaluationCircuitPerSlice(v []uint32, sliceNum int64) []uint32 {
	if td.m == 0 {
		td.updateInternalUseParams()
	}

	if int(td.n) > len(v) {
		padded := dataobjects.AlignedMake[uint32](uint64(td.n))
		copy(padded, v)
		v = padded
	}

	masks := dataobjects.AlignedMake[uint32](uint64(td.m))
	bv := dataobjects.AlignedMake[uint32](uint64(td.block))
	for j := uint32(0); j < td.n/td.block; j++ {
		copy(bv, v[j*td.block:(j+1)*td.block])
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

	return masks[0:td.M]
}

func (td *TDM) EvaluationCircuitBasic(v []uint32, addOnSeed int64) []uint32 {
	// S_R = [I | C] x v
	resR := dataobjects.AlignedMake[uint32](uint64(ExpansionFactor * td.block))
	copy(resR, v)
	vec := CirculantVectorMul(td.block, td.Q, td.rootK, td.SeedR+addOnSeed, v)
	copy(resR[td.block:], vec)

	// Apply PermR
	perm := GetPermutation(ExpansionFactor*td.block, td.SeedPR+addOnSeed)
	PermuteVectorInPlace(resR, perm)

	// Multiply by S
	resC := CirculantVectorMul(ExpansionFactor*td.block, td.Q, td.root2K, td.SeedC+addOnSeed, resR)

	// Apply PermL
	perm = GetPermutation(ExpansionFactor*td.block, td.SeedPL+addOnSeed)
	PermuteVectorInPlace(resC, perm)

	// S_L = [I // C] x resC
	vec = CirculantVectorMul(td.block, td.Q, td.rootK, td.SeedL+addOnSeed, resC[td.block:])
	if dataobjects.USE_FAST_CODE {
		dataobjects.FieldAddVectors(resC, 0, resC, 0, vec, 0, uint64(td.block), td.Q)
	} else {
		for i := uint32(0); i < td.block; i++ {
			resC[i] = uint32((uint64(resC[i]) + uint64(vec[i])) % uint64(td.Q))
		}
	}

	return resC[:td.block]
}

func CirculantMatrixMul(blockSize, q, root uint32, seed int64, mat [][]uint32) [][]uint32 {
	result := make([][]uint32, blockSize)
	for i := range result {
		result[i] = dataobjects.AlignedMake[uint32](uint64(len(mat[0])))
	}
	polyQC := dataobjects.AlignedMake[uint32](uint64(blockSize))
	res := dataobjects.AlignedMake[uint32](uint64(blockSize))

	if dataobjects.USE_FAST_CODE && USE_FAST_CODE_FOR_CIRCULANT {
		utils.RandomizeVectorWithModulusAndSeed(polyQC, blockSize, q, seed)
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

	for j := 0; j < len(mat[0]); j++ {
		for i := range mat {
			v[i] = mat[i][j]
		}
		NTT_Convolution(polyQC, v, res, blockSize, root, q)
		for i := range res {
			result[i][j] = res[i]
		}
	}

	return result
}

func CirculantVectorMul(blockSize, q, root uint32, seed int64, v []uint32) []uint32 {
	result := dataobjects.AlignedMake[uint32](uint64(blockSize))
	polyQC := dataobjects.AlignedMake[uint32](uint64(blockSize))

	if dataobjects.USE_FAST_CODE && USE_FAST_CODE_FOR_CIRCULANT {
		utils.RandomizeVectorWithModulusAndSeed(polyQC, blockSize, q, seed)
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

	NTT_Convolution(polyQC, v, result, blockSize, root, q)
	return result
}

func PermuteRowsInPlace(matrix [][]uint32, perm []uint32) {
	n := uint32(len(matrix))

	for i := uint32(0); i < n; i++ {
		if perm[i] == i {
			continue
		}

		j := i
		temp := matrix[i]
		for {
			next := perm[j]
			perm[j] = j

			if next == i {
				matrix[j] = temp
				break
			}

			matrix[j] = matrix[next]
			j = next
		}
	}
}

func PermuteVectorInPlace(vec []uint32, perm []uint32) {
	n := uint32(len(vec))

	for i := uint32(0); i < n; i++ {
		if perm[i] == i {
			continue
		}

		j := i
		temp := vec[i]
		for {
			next := perm[j]
			perm[j] = j

			if next == i {
				vec[j] = temp
				break
			}

			vec[j] = vec[next]
			j = next
		}
	}
}

func GetPermutation(n uint32, seed int64) []uint32 {
	rng := rand.New(rand.NewSource(seed))
	perm := dataobjects.AlignedMake[uint32](uint64(n))
	for i := uint32(0); i < n; i++ {
		perm[i] = i
	}
	rng.Shuffle(int(n), func(i, j int) {
		perm[i], perm[j] = perm[j], perm[i]
	})

	return perm
}

// Q has the form [I // C] where C is a circulant matrix
func GetQuasiCyclicMatrix(blockSize, q uint32, seed int64) [][]uint32 {
	row := 2 * blockSize
	Q := make([][]uint32, row)
	for i := uint32(0); i < row; i++ {
		Q[i] = dataobjects.AlignedMake[uint32](uint64(blockSize))
	}

	for i := uint32(0); i < blockSize; i++ {
		Q[i][i] = 1
	}

	S := GetCirculantMatrix(blockSize, q, seed)

	for i := uint32(0); i < blockSize; i++ {
		copy(Q[blockSize+i], S[i])
	}

	return Q
}

func GetCirculantMatrix(k, q uint32, seed int64) [][]uint32 {
	S := make([][]uint32, k)
	for i := uint32(0); i < k; i++ {
		S[i] = dataobjects.AlignedMake[uint32](uint64(k))
	}

	poly := dataobjects.AlignedMake[uint32](uint64(k))
	if dataobjects.USE_FAST_CODE && USE_FAST_CODE_FOR_CIRCULANT {
		utils.RandomizeVectorWithModulusAndSeed(poly, k, q, seed)
	} else {
		rng := rand.New(rand.NewSource(seed))
		for t := uint32(0); t < k; t++ {
			poly[t] = uint32(rng.Intn(int(q)))
		}
	}

	if dataobjects.USE_FAST_CODE {
		for t := uint32(0); t < k; t++ {
			copy(S[t][t:k], poly[0:k-t])
			copy(S[t][0:t], poly[k-t:k])
		}
	} else {
		for i := uint32(0); i < k; i++ {
			for t := uint32(0); t < k; t++ {
				copy(S[t][t:k], poly[0:k-t])
				copy(S[t][0:t], poly[k-t:k])
			}
		}
	}

	return S
}

func (td *TDM) updateInternalUseParams() {
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

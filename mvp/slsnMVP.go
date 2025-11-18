package mvp

import (
	"RandomLinearCodePIR/dataobjects"
	"RandomLinearCodePIR/linearcode"
	"RandomLinearCodePIR/tdm"
	"RandomLinearCodePIR/utils"
	"context"
	"time"
)

type SlsnMVP struct {
	Ctx    context.Context
	Params SlsnParams
}

type SecretKey struct {
	LinearCodeKey   int64
	TDMKey          int64
	PreLoadedMatrix []uint32
	TDM             *tdm.TDM
}

func (sk *SecretKey) Free() {
	sk.PreLoadedMatrix = dataobjects.Aligned1DFree(sk.PreLoadedMatrix)
}

func (sk *SecretKey) DoFree(doctx *dataobjects.DoContext) bool {
	if sk.PreLoadedMatrix != nil {
		if !dataobjects.DoAligned1DFree(doctx, sk.PreLoadedMatrix) {
			return false
		}
		sk.PreLoadedMatrix = nil
	}
	return true
}

// N = K + L denotes the length of the codeword
// Encoding Matrix D with dimension N x L
// Original Data Matrix has dimension M x N
// S denotes the number of blocks
// B denotes the block size
// We assume N = S x B
type SlsnParams struct {
	Field dataobjects.Field
	// Temporarily add P here
	P uint32
	S uint32
	B uint32
	K uint32
	L uint32
	N uint32
	M uint32
}

type SlsnQuery struct {
	Vec []uint32
}

func (slsnQuery *SlsnQuery) Free() {
	slsnQuery.Vec = dataobjects.Aligned1DFree(slsnQuery.Vec)
}

func (slsnQuery *SlsnQuery) DoFree(doctx *dataobjects.DoContext) bool {
	if slsnQuery.Vec != nil {
		if !dataobjects.DoAligned1DFree(doctx, slsnQuery.Vec) {
			return false
		}
		slsnQuery.Vec = nil
	}
	return true
}

type SlsnAux struct {
	Coeff []uint32
	Masks []uint32
	Dur   time.Duration
}

func (slsnAux *SlsnAux) Free() {
	slsnAux.Coeff = dataobjects.Aligned1DFree(slsnAux.Coeff)
	slsnAux.Masks = dataobjects.Aligned1DFree(slsnAux.Masks)
}

func (slsnAux *SlsnAux) DoFree(doctx *dataobjects.DoContext) bool {
	if slsnAux.Coeff != nil {
		if !dataobjects.DoAligned1DFree(doctx, slsnAux.Coeff) {
			return false
		}
		slsnAux.Coeff = nil
	}
	if slsnAux.Masks != nil {
		if !dataobjects.DoAligned1DFree(doctx, slsnAux.Masks) {
			return false
		}
		slsnAux.Masks = nil
	}
	return true
}

func (slsn *SlsnMVP) KeyGen(seed int64) SecretKey {
	params := slsn.Params
	return SecretKey{
		LinearCodeKey:   seed,
		PreLoadedMatrix: linearcode.Generate1DDualMatrix(slsn.Ctx, params.L, params.K, params.Field, seed),
		TDM: &tdm.TDM{
			Ctx: slsn.Ctx,
			M:   params.M,
			N:   params.N,
			// NOTE: Now TDM only support Q = 2^x + 1, Change this to Field later
			Q:      params.P,
			SeedL:  seed + 1,
			SeedPL: seed + 1<<10,
			SeedC:  seed + 1<<11,
			SeedPR: seed + 1<<12,
			SeedR:  seed + 1<<13,
		},
	}
}

func (slsn *SlsnMVP) GenerateTDM(sk SecretKey) []uint32 {
	return sk.TDM.GenerateFlattenedTrapDooredMatrix()
}

func (slsn *SlsnMVP) Encode(sk SecretKey, input dataobjects.Matrix, mask []uint32) *dataobjects.Matrix {
	doctx := dataobjects.GetDeferralDoContext(slsn.Ctx)
	frame := dataobjects.MakeDeferralFrame(slsn.Ctx)
	defer frame.Close()

	params := slsn.Params
	rlcMatrix := linearcode.Generate1DRLCMatrix(slsn.Ctx, params.L, params.K, params.Field, sk.LinearCodeKey)
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, rlcMatrix) })
	encoded := dataobjects.DoAlignedMake[uint32](doctx, uint64(input.Rows*params.N))
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, encoded) })

	if dataobjects.USE_FAST_CODE {
		tdm.PermutedExtentsAssign(doctx, encoded, 0, params.N, 0, input.Data, 0, params.L, 0, uint64(params.L), nil, 0, uint64(input.Rows))
		MatVecProductExt(doctx, rlcMatrix, 0, 0, input.Data, 0, params.L, encoded, params.L, params.N, params.K, params.L, input.Rows, params.P)
	} else {
		for i := uint32(0); i < input.Rows; i++ {
			if dataobjects.USE_FAST_CODE {
				dataobjects.FieldCopyVector(doctx, encoded, uint64(i*params.N), input.Data, uint64(i*params.L), uint64(params.L))
			} else {
				copy(encoded[i*params.N:i*params.N+params.L], input.Data[i*params.L:(i+1)*params.L])
			}

			MatVecProduct(doctx, rlcMatrix, input.Data[i*input.Cols:(i+1)*input.Cols], encoded[i*params.N+params.L:(i+1)*params.N],
				params.K, params.L, params.P)
		}
	}

	// Add Masks
	params.Field.AddVectors(encoded, 0, encoded, 0, mask, 0, uint64(len(encoded)))

	blockwizeEncodedMatrix := dataobjects.DoAlignedMake[uint32](doctx, uint64(len(encoded)))
	TransformToBlockwise(doctx, encoded, 0, params.N, blockwizeEncodedMatrix, 0, params.M, params.M, params.N, params.S)

	return &dataobjects.Matrix{
		Rows: params.M,
		Cols: params.N,
		Data: blockwizeEncodedMatrix,
	}
}

func (slsn *SlsnMVP) Query(sk SecretKey, vec []uint32) (*SlsnQuery, *SlsnAux) {
	doctx := dataobjects.GetDeferralDoContext(slsn.Ctx)
	frame := dataobjects.MakeDeferralFrame(slsn.Ctx)
	defer frame.Close()

	// FIXME - use appropriate seed
	seed := int64(257) //utils.MakeSeed(vec)

	params := slsn.Params

	PofDual := sk.PreLoadedMatrix

	// Sample codeword c From NullSpace
	nullspaceCoeff := dataobjects.DoAlignedMake[uint32](doctx, uint64(params.K))
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, nullspaceCoeff) })
	utils.SampleVector(doctx, params.Field, nullspaceCoeff, params.K, seed+1)

	queryVector := dataobjects.DoAlignedMake[uint32](doctx, uint64(params.N))

	MatVecProduct(doctx, PofDual, nullspaceCoeff, queryVector, params.L, params.K, params.P)

	if dataobjects.USE_FAST_CODE {
		dataobjects.FieldCopyVector(doctx, queryVector, uint64(params.L), nullspaceCoeff, 0, uint64(params.K)) // relies on K + L = N
	} else {
		copy(queryVector[params.L:params.N], nullspaceCoeff[:params.K])
	}

	// Add Vector v to c
	params.Field.AddVectors(queryVector, 0, queryVector, 0, vec, 0, uint64(params.L))

	// The time is just for benchmark
	start := time.Now()
	// Calculate The Mask
	masks := sk.TDM.EvaluationCircuit(queryVector)
	dur := time.Since(start)

	// Generate Non-zero coefficient
	coeff := dataobjects.DoAlignedMake[uint32](doctx, uint64(params.S))
	utils.SampleInvertibleVec(doctx, params.Field, coeff, params.S, seed+2)

	for i := uint32(0); i < params.S; i++ {
		dataobjects.FieldMulVectorsExt(doctx, queryVector, uint64(i*params.B), 1, queryVector, uint64(i*params.B), 1, coeff, uint64(i), 0, 1, uint64(params.B), params.Field.Mod())
	}

	return &SlsnQuery{
			Vec: queryVector,
		}, &SlsnAux{
			Coeff: coeff,
			Masks: masks,
			Dur:   dur,
		}
}

func (slsn *SlsnMVP) Answer(encodedMatrix dataobjects.Matrix, clientQuery SlsnQuery) []uint32 {
	doctx := dataobjects.GetDeferralDoContext(slsn.Ctx)
	params := slsn.Params
	result := dataobjects.DoAlignedMake[uint32](doctx, uint64(params.S*params.M))

	BlockMatVecProduct(doctx, encodedMatrix.Data, clientQuery.Vec, result, params.M, params.N, params.S, params.P)
	return result
}

func (slsn *SlsnMVP) Decode(sk SecretKey, response []uint32, aux SlsnAux) []uint32 {
	doctx := dataobjects.GetDeferralDoContext(slsn.Ctx)
	frame := dataobjects.MakeDeferralFrame(slsn.Ctx)
	defer frame.Close()

	params := slsn.Params

	vec := dataobjects.DoAlignedMake[uint32](doctx, uint64(len(aux.Coeff)))
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, vec) })
	params.Field.InvertVector(aux.Coeff, vec)

	result := dataobjects.DoAlignedMake[uint32](doctx, uint64(params.M))

	BlockVecMatProduct(doctx, response, vec, result, params.S, params.M, 1, params.P)
	// Unmask
	if dataobjects.USE_FAST_CODE {
		dataobjects.FieldSubVectors(doctx, result, 0, result, 0, aux.Masks, 0, uint64(params.M), params.Field.Mod())
	} else {
		for i := uint32(0); i < params.M; i++ {
			result[i] = params.Field.Sub(result[i], aux.Masks[i])
		}
	}

	return result
}

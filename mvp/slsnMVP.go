package mvp

import (
	"RandomLinearCodePIR/dataobjects"
	"RandomLinearCodePIR/linearcode"
	"RandomLinearCodePIR/tdm"
	"time"
)

type SlsnMVP struct {
	Params SlsnParams
}

type SecretKey struct {
	LinearCodeKey   int64
	TDMKey          int64
	PreLoadedMatrix []uint32
	TDM             *tdm.TDM
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

type SlsnAux struct {
	Coeff []uint32
	Masks []uint32
	Dur   time.Duration
}

func (slsn *SlsnMVP) KeyGen(seed int64) SecretKey {
	params := slsn.Params
	return SecretKey{
		LinearCodeKey:   seed,
		PreLoadedMatrix: linearcode.Generate1DDualMatrix(params.L, params.K, params.Field, seed),
		TDM: &tdm.TDM{
			M: params.M,
			N: params.N,
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
	params := slsn.Params
	rlcMatrix := linearcode.Generate1DRLCMatrix(params.L, params.K, params.Field, sk.LinearCodeKey)
	encoded := dataobjects.AlignedMake[uint32](uint64(input.Rows * params.N))

	for i := uint32(0); i < input.Rows; i++ {
		copy(encoded[i*params.N:i*params.N+params.L], input.Data[i*params.L:(i+1)*params.L])

		MatVecProduct(rlcMatrix, input.Data[i*input.Cols:(i+1)*input.Cols], encoded[i*params.N+params.L:(i+1)*params.N],
			params.K, params.L, params.P)
	}

	// Add Masks
	params.Field.AddVectors(encoded, 0, encoded, 0, mask, 0, uint64(len(encoded)))

	return &dataobjects.Matrix{
		Rows: params.M,
		Cols: params.N,
		Data: encoded,
	}
}

func (slsn *SlsnMVP) Query(sk SecretKey, vec []uint32) (*SlsnQuery, *SlsnAux) {
	params := slsn.Params

	PofDual := sk.PreLoadedMatrix
	if len(PofDual) == 0 {
		PofDual = linearcode.Generate1DDualMatrix(params.L, params.K, params.Field, sk.LinearCodeKey)
	}

	// Sample codeword c From NullSpace
	nullspaceCoeff := params.Field.SampleVector(params.K)

	queryVector := dataobjects.AlignedMake[uint32](uint64(params.N))

	MatVecProduct(PofDual, nullspaceCoeff, queryVector, params.L, params.K, params.P)

	copy(queryVector[params.L:params.N], nullspaceCoeff[:params.K])

	// Add Vector v to c
	params.Field.AddVectors(queryVector, 0, queryVector, 0, vec, 0, uint64(params.L))

	// The time is just for benchmark
	start := time.Now()
	// Calculate The Mask
	masks := sk.TDM.EvaluationCircuit(queryVector)
	dur := time.Since(start)

	// Generate Non-zero coefficient
	coeff := params.Field.SampleInvertibleVec(params.S)

	for i := uint32(0); i < params.S; i++ {
		params.Field.MulVector(queryVector, uint64(i*params.B), queryVector, uint64(i*params.B), coeff[i], uint64(params.B))
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
	params := slsn.Params
	result := dataobjects.AlignedMake[uint32](uint64(params.S * params.M))

	BlockMatVecProduct(encodedMatrix.Data, clientQuery.Vec, result, params.M, params.N, params.S, params.P)
	return result
}

func (slsn *SlsnMVP) Decode(sk SecretKey, response []uint32, aux SlsnAux) []uint32 {
	params := slsn.Params

	vec := params.Field.InvertVector(aux.Coeff)

	result := dataobjects.AlignedMake[uint32](uint64(params.M))

	BlockVecMatProduct(response, vec, result, params.S, params.M, 1, params.P)
	// Unmask
	for i := uint32(0); i < params.M; i++ {
		result[i] = params.Field.Sub(result[i], aux.Masks[i])
	}

	return result
}

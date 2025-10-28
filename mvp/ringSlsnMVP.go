package mvp

import (
	"RandomLinearCodePIR/dataobjects"
	"RandomLinearCodePIR/linearcode"
	"time"
)

type RingSlsnMVP struct {
	SlsnMVP           SlsnMVP
	LinearCodeEncoder linearcode.LinearCode
}

func (rmvp *RingSlsnMVP) KeyGen(seed int64) SecretKey {
	rmvp.LinearCodeEncoder = linearcode.GetLinearCode(linearcode.LinearCodeConfig{
		Name:  linearcode.Vandermonde,
		K:     rmvp.SlsnMVP.Params.K,
		L:     rmvp.SlsnMVP.Params.L,
		Field: rmvp.SlsnMVP.Params.Field,
	})
	return rmvp.SlsnMVP.KeyGen(seed)
}

func (rmvp *RingSlsnMVP) GenerateTDM(sk SecretKey) []uint32 {
	return rmvp.SlsnMVP.GenerateTDM(sk)
}

func (rmvp *RingSlsnMVP) Encode(sk SecretKey, input dataobjects.Matrix, mask []uint32) *dataobjects.Matrix {
	params := rmvp.SlsnMVP.Params
	encoded := dataobjects.AlignedMake[uint32](uint64(input.Rows * params.N))
	defer dataobjects.Aligned1DFree(encoded)

	for i := uint32(0); i < input.Rows; i++ {
		copy(encoded[i*params.N:i*params.N+params.L], input.Data[i*params.L:(i+1)*params.L])
		encodedDual, encodedDualIsNew := rmvp.LinearCodeEncoder.EncodeDual(input.Data[i*params.L : (i+1)*params.L])
		if encodedDualIsNew {
			defer dataobjects.Aligned1DFree(encodedDual)
		}
		copy(encoded[i*params.N+params.L:(i+1)*params.N], encodedDual)
	}

	params.Field.AddVectors(encoded, 0, encoded, 0, mask, 0, uint64(len(encoded)))

	blockwizeEncodedMatrix := dataobjects.AlignedMake[uint32](uint64(len(encoded)))

	TransformToBlockwise(encoded, blockwizeEncodedMatrix, params.M, params.N, params.S)

	return &dataobjects.Matrix{
		Rows: params.M,
		Cols: params.N,
		Data: blockwizeEncodedMatrix,
	}
}

func (rmvp *RingSlsnMVP) Query(sk SecretKey, vec []uint32) (*SlsnQuery, *SlsnAux) {
	params := rmvp.SlsnMVP.Params

	nullspaceCoeff := params.Field.SampleVector(params.K)
	defer dataobjects.Aligned1DFree(nullspaceCoeff)

	queryVector := dataobjects.AlignedMake[uint32](uint64(params.N))

	copy(queryVector[params.L:params.N], nullspaceCoeff[:params.K])

	encodedNullspaceCoeff, encodedNullspaceCoeffIsNew := rmvp.LinearCodeEncoder.EncodeLSN(nullspaceCoeff)
	if encodedNullspaceCoeffIsNew {
		defer dataobjects.Aligned1DFree(encodedNullspaceCoeff)
	}
	copy(queryVector[:params.L], encodedNullspaceCoeff)

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

func (rmvp *RingSlsnMVP) Answer(encodedMatrix dataobjects.Matrix, clientQuery SlsnQuery) []uint32 {
	return rmvp.SlsnMVP.Answer(encodedMatrix, clientQuery)
}

func (rmvp *RingSlsnMVP) Decode(sk SecretKey, response []uint32, aux SlsnAux) []uint32 {
	return rmvp.SlsnMVP.Decode(sk, response, aux)
}

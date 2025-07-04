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
	encoded := make([]uint32, input.Rows*params.N)

	for i := uint32(0); i < input.Rows; i++ {
		copy(encoded[i*params.N:i*params.N+params.L], input.Data[i*params.L:(i+1)*params.L])
		copy(encoded[i*params.N+params.L:(i+1)*params.N], rmvp.LinearCodeEncoder.EncodeDual(input.Data[i*params.L:(i+1)*params.L]))
	}

	for j := uint64(0); j < uint64(len(encoded)); j++ {
		encoded[j] = params.Field.Add(encoded[j], mask[j])
	}

	return &dataobjects.Matrix{
		Rows: params.M,
		Cols: params.N,
		Data: encoded,
	}
}

func (rmvp *RingSlsnMVP) Query(sk SecretKey, vec []uint32) (*SlsnQuery, *SlsnAux) {
	params := rmvp.SlsnMVP.Params

	nullspaceCoeff := params.Field.SampleVector(params.K)

	queryVector := make([]uint32, params.N)

	copy(queryVector[params.L:params.N], nullspaceCoeff[:params.K])

	copy(queryVector[:params.L], rmvp.LinearCodeEncoder.EncodeLSN(nullspaceCoeff))

	// Add Vector v to c
	for i := uint32(0); i < params.L; i++ {
		queryVector[i] = params.Field.Add(queryVector[i], vec[i])
	}

	// The time is just for benchmark
	start := time.Now()
	// Calculate The Mask
	masks := sk.TDM.EvaluationCircuit(queryVector)
	dur := time.Since(start)

	// Generate Non-zero coefficient
	coeff := params.Field.SampleInvertibleVec(params.S)

	for i := uint32(0); i < params.S; i++ {
		for j := uint32(0); j < params.B; j++ {
			queryVector[i*params.B+j] = params.Field.Mul(queryVector[i*params.B+j], coeff[i])
		}
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

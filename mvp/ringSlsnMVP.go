package mvp

import (
	"RandomLinearCodePIR/dataobjects"
	"RandomLinearCodePIR/linearcode"
	"RandomLinearCodePIR/utils"
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
	frame := dataobjects.MakeDeferralFrame(rmvp.SlsnMVP.Ctx)
	defer frame.Close()

	params := rmvp.SlsnMVP.Params
	encoded := dataobjects.AlignedMake[uint32](uint64(input.Rows * params.N))
	frame.Defer(func() { dataobjects.Aligned1DFree(encoded) })
	encodedDualBuffer := dataobjects.AlignedMake[uint32](uint64(rmvp.LinearCodeEncoder.EncodeLength()))
	frame.Defer(func() { dataobjects.Aligned1DFree(encodedDualBuffer) })

	for i := uint32(0); i < input.Rows; i++ {
		if dataobjects.USE_FAST_CODE {
			dataobjects.FieldCopyVector(encoded, uint64(i*params.N), input.Data, uint64(i*params.L), uint64(params.L))
		} else {
			copy(encoded[i*params.N:i*params.N+params.L], input.Data[i*params.L:(i+1)*params.L])
		}
		encodedDual := rmvp.LinearCodeEncoder.EncodeDual(input.Data[i*params.L : (i+1)*params.L]) // relies on `encodedDual` sharing memory with `input`
		if dataobjects.USE_FAST_CODE {
			dataobjects.FieldCopyVector(encoded, uint64(i*params.N+params.L), encodedDual, 0, min(uint64(len(encodedDual)), uint64(params.K))) // relies on K + L = N
		} else {
			copy(encoded[i*params.N+params.L:(i+1)*params.N], encodedDual)
		}
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
	frame := dataobjects.MakeDeferralFrame(rmvp.SlsnMVP.Ctx)
	defer frame.Close()

	params := rmvp.SlsnMVP.Params

	nullspaceCoeff := dataobjects.AlignedMake[uint32](uint64(rmvp.LinearCodeEncoder.EncodeLength()))
	utils.SampleVector(params.Field, nullspaceCoeff, params.K)
	frame.Defer(func() { dataobjects.Aligned1DFree(nullspaceCoeff) })

	queryVector := dataobjects.AlignedMake[uint32](uint64(params.N))

	if dataobjects.USE_FAST_CODE {
		dataobjects.FieldCopyVector(queryVector, uint64(params.L), nullspaceCoeff, 0, uint64(params.K)) // relies on K + L = N
	} else {
		copy(queryVector[params.L:params.N], nullspaceCoeff[:params.K])
	}

	encodedNullspaceCoeff := rmvp.LinearCodeEncoder.EncodeLSN(nullspaceCoeff)
	if dataobjects.USE_FAST_CODE {
		dataobjects.FieldCopyVector(queryVector, 0, encodedNullspaceCoeff, 0, min(uint64(params.L), uint64(len(encodedNullspaceCoeff))))
	} else {
		copy(queryVector[:params.L], encodedNullspaceCoeff)
	}

	// Add Vector v to c
	params.Field.AddVectors(queryVector, 0, queryVector, 0, vec, 0, uint64(params.L))

	// The time is just for benchmark
	start := time.Now()
	// Calculate The Mask
	masks := sk.TDM.EvaluationCircuit(queryVector)
	dur := time.Since(start)

	// Generate Non-zero coefficient
	coeff := dataobjects.AlignedMake[uint32](uint64(params.S))
	utils.SampleInvertibleVec(params.Field, coeff, params.S)

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

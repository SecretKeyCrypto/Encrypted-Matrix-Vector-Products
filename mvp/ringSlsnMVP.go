package mvp

import (
	"RandomLinearCodePIR/dataobjects"
	"RandomLinearCodePIR/linearcode"
	"RandomLinearCodePIR/tdm"
	"RandomLinearCodePIR/utils"
	"time"
)

type RingSlsnMVP struct {
	SlsnMVP           SlsnMVP
	LinearCodeEncoder linearcode.LinearCode
}

func (rmvp *RingSlsnMVP) KeyGen(seed int64) SecretKey {
	rmvp.LinearCodeEncoder = linearcode.GetLinearCode(rmvp.SlsnMVP.Ctx, linearcode.LinearCodeConfig{
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
	doctx := dataobjects.GetDeferralDoContext(rmvp.SlsnMVP.Ctx)
	frame := dataobjects.MakeDeferralFrame(rmvp.SlsnMVP.Ctx)
	defer frame.Close()

	params := rmvp.SlsnMVP.Params
	enclen := rmvp.LinearCodeEncoder.EncodeLength() + params.L // sufficient for encoding at offset params.L
	encoded := dataobjects.DoAlignedMake[uint32](doctx, uint64(input.Rows*enclen))
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, encoded) })
	encodedDualBuffer := dataobjects.DoAlignedMake[uint32](doctx, uint64(rmvp.LinearCodeEncoder.EncodeLength()))
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, encodedDualBuffer) })

	if dataobjects.USE_FAST_CODE {
		tdm.PermutedExtentsAssign(doctx, encoded, 0, enclen, 0, input.Data, 0, params.L, 0, uint64(params.L), nil, 0, uint64(input.Rows))
		tdm.PermutedExtentsAssign(doctx, encoded, params.L, enclen, 0, input.Data, 0, params.L, 0, uint64(params.L), nil, 0, uint64(input.Rows))
		rmvp.LinearCodeEncoder.EncodeDual(encoded, params.L, enclen, input.Rows)
	} else {
		for i := uint32(0); i < input.Rows; i++ {
			if dataobjects.USE_FAST_CODE {
				dataobjects.FieldCopyVector(doctx, encoded, uint64(i*enclen), input.Data, uint64(i*params.L), uint64(params.L))
			} else {
				copy(encoded[i*enclen:i*enclen+params.L], input.Data[i*params.L:(i+1)*params.L])
			}
			encodedDual := input.Data[i*params.L : (i+1)*params.L]
			rmvp.LinearCodeEncoder.EncodeDual(encodedDual, 0, 0, 1) // relies on `encodedDual` sharing memory with `input`
			if dataobjects.USE_FAST_CODE {
				dataobjects.FieldCopyVector(doctx, encoded, uint64(i*enclen+params.L), encodedDual, 0, min(uint64(len(encodedDual)), uint64(params.K))) // relies on K + L = N
			} else {
				copy(encoded[i*enclen+params.L:(i+1)*enclen], encodedDual)
			}
		}
	}

	params.Field.AddVectorsExt(encoded, 0, uint64(enclen), encoded, 0, uint64(enclen), mask, 0, uint64(params.N), uint64(params.N), uint64(params.M))

	blockwizeEncodedMatrix := dataobjects.DoAlignedMake[uint32](doctx, uint64(input.Rows)*uint64(params.N))

	TransformToBlockwise(doctx, encoded, 0, enclen, blockwizeEncodedMatrix, 0, params.M, params.M, params.N, params.S)

	return &dataobjects.Matrix{
		Rows: params.M,
		Cols: params.N,
		Data: blockwizeEncodedMatrix,
	}
}

func (rmvp *RingSlsnMVP) Query(sk SecretKey, vec []uint32) (*SlsnQuery, *SlsnAux) {
	doctx := dataobjects.GetDeferralDoContext(rmvp.SlsnMVP.Ctx)
	frame := dataobjects.MakeDeferralFrame(rmvp.SlsnMVP.Ctx)
	defer frame.Close()

	seed := utils.MakeSeed(vec)

	params := rmvp.SlsnMVP.Params

	nullspaceCoeff := dataobjects.DoAlignedMake[uint32](doctx, uint64(rmvp.LinearCodeEncoder.EncodeLength()))
	utils.SampleVector(doctx, params.Field, nullspaceCoeff, params.K, seed+1)
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, nullspaceCoeff) })

	queryVector := dataobjects.DoAlignedMake[uint32](doctx, uint64(params.N))

	if dataobjects.USE_FAST_CODE {
		dataobjects.FieldCopyVector(doctx, queryVector, uint64(params.L), nullspaceCoeff, 0, uint64(params.K)) // relies on K + L = N
	} else {
		copy(queryVector[params.L:params.N], nullspaceCoeff[:params.K])
	}

	rmvp.LinearCodeEncoder.EncodeLSN(nullspaceCoeff, 0, 0, 1)
	encodedNullspaceCoeff := nullspaceCoeff[:params.L]
	if dataobjects.USE_FAST_CODE {
		dataobjects.FieldCopyVector(doctx, queryVector, 0, encodedNullspaceCoeff, 0, min(uint64(params.L), uint64(len(encodedNullspaceCoeff))))
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

func (rmvp *RingSlsnMVP) Answer(encodedMatrix dataobjects.Matrix, clientQuery SlsnQuery) []uint32 {
	return rmvp.SlsnMVP.Answer(encodedMatrix, clientQuery)
}

func (rmvp *RingSlsnMVP) Decode(sk SecretKey, response []uint32, aux SlsnAux) []uint32 {
	return rmvp.SlsnMVP.Decode(sk, response, aux)
}

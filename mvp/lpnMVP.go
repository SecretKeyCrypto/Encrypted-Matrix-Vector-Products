package mvp

import (
	"RandomLinearCodePIR/dataobjects"
	"RandomLinearCodePIR/ecc"
	"RandomLinearCodePIR/linearcode"
	"RandomLinearCodePIR/tdm"
	"RandomLinearCodePIR/utils"
	"context"
)

type LpnMVP struct {
	Ctx    context.Context
	Params LpnParams
}

type LpnParams struct {
	Field     dataobjects.Field
	Epsi      float64
	N         uint32
	M         uint32
	L         uint32
	K         uint32
	M_1       uint32
	P         uint32
	ECCLength uint32
	ECCName   string
}

type LpnQuery struct {
	Vec          []uint32
	QueryLen     uint32
	NumOfQueries uint32
}

func (lpnQuery *LpnQuery) Free() {
	lpnQuery.Vec = dataobjects.Aligned1DFree(lpnQuery.Vec)
}

func (lpnQuery *LpnQuery) DoFree(doctx *dataobjects.DoContext) bool {
	if lpnQuery.Vec != nil {
		if !dataobjects.DoAligned1DFree(doctx, lpnQuery.Vec) {
			return false
		}
		lpnQuery.Vec = nil
	}
	return true
}

type LpnAux struct {
	NoisyQueryIndicator []bool
	Masks               []uint32
}

func (lpnAux *LpnAux) Free() {
	lpnAux.NoisyQueryIndicator = dataobjects.Aligned1DFree(lpnAux.NoisyQueryIndicator)
	lpnAux.Masks = dataobjects.Aligned1DFree(lpnAux.Masks)
}

func (lpnAux *LpnAux) DoFree(doctx *dataobjects.DoContext) bool {
	if lpnAux.NoisyQueryIndicator != nil {
		if !dataobjects.DoAligned1DFree(doctx, lpnAux.NoisyQueryIndicator) {
			return false
		}
		lpnAux.NoisyQueryIndicator = nil
	}
	if lpnAux.Masks != nil {
		if !dataobjects.DoAligned1DFree(doctx, lpnAux.Masks) {
			return false
		}
		lpnAux.Masks = nil
	}
	return true
}

type LpnResponse struct {
	Answers []uint32
	AnsLen  uint32
}

func (lpnResponse *LpnResponse) Free() {
	lpnResponse.Answers = dataobjects.Aligned1DFree(lpnResponse.Answers)
}

func (lpnResponse *LpnResponse) DoFree(doctx *dataobjects.DoContext) bool {
	if lpnResponse.Answers != nil {
		if !dataobjects.DoAligned1DFree(doctx, lpnResponse.Answers) {
			return false
		}
		lpnResponse.Answers = nil
	}
	return true
}

func (lpn *LpnMVP) KeyGen(seed int64) SecretKey {
	params := lpn.Params
	return SecretKey{
		LinearCodeKey:   seed,
		PreLoadedMatrix: linearcode.Generate1DDualMatrix(lpn.Ctx, params.L, params.K, params.Field, seed),
		TDM: &tdm.TDM{
			Ctx: lpn.Ctx,
			// Trapdoored matrix would be applied Each Slice with params.M / params.M_1 rows
			M: params.M / params.M_1,
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

func (lpn *LpnMVP) GenerateTDM(sk SecretKey) [][]uint32 {
	masks := make([][]uint32, lpn.Params.ECCLength)
	for i := range masks {
		masks[i] = sk.TDM.GenerateFlattenedTrapDooredMatrixPerSlice(int64(i))
	}
	return masks
}

func (lpn *LpnMVP) Encode(sk SecretKey, input dataobjects.Matrix, masks [][]uint32) *dataobjects.Matrix {
	doctx := dataobjects.GetDeferralDoContext(lpn.Ctx)
	frame := dataobjects.MakeDeferralFrame(lpn.Ctx)
	defer frame.Close()

	params := lpn.Params
	rlcMatrix := linearcode.Generate1DRLCMatrix(lpn.Ctx, params.L, params.K, params.Field, sk.LinearCodeKey)
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, rlcMatrix) })

	// Assume M_1 | M for now
	rowPerSlice := params.M / params.M_1
	entryPerSlice := rowPerSlice * params.N

	encoded := dataobjects.DoAlignedMake[uint32](doctx, uint64(entryPerSlice*params.ECCLength))

	// Re-use slot for ECC encoding
	message := dataobjects.DoAlignedMake[uint32](doctx, uint64(params.ECCLength))
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, message) })

	generatorMatrix := ecc.GetECCCode(lpn.Ctx, ecc.ECCConfig{
		Name: params.ECCName,
		Q:    params.P,
		N:    params.ECCLength,
		K:    params.M_1}).GetGeneratorMatrix(params.M_1, params.ECCLength, params.P)
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, generatorMatrix) })

	if dataobjects.USE_FAST_CODE {
		LpnEncode(doctx, input.Data, rlcMatrix, generatorMatrix, encoded, params.M, params.L, params.K, params.M_1, params.ECCLength, params.P)
	} else {
		for i := range rowPerSlice {
			for j := uint32(0); j < params.M_1; j++ {
				// Input matrix with each row length L, block size M_1
				inputStart := (i*params.M_1 + j) * params.L

				// Put into the jth slice, ith row, each row with length N
				outputStart := j*entryPerSlice + i*params.N

				// Copy the input row into the first L element of the output row
				if dataobjects.USE_FAST_CODE {
					dataobjects.FieldCopyVector(doctx, encoded, uint64(outputStart), input.Data, uint64(inputStart), uint64(params.L))
				} else {
					copy(encoded[outputStart:outputStart+params.L], input.Data[inputStart:inputStart+params.L])
				}

				MatVecProductExt(doctx, rlcMatrix, 0, 0, input.Data, inputStart, params.L, encoded, outputStart+params.L, params.N,
					params.K, params.L, 1, params.P)
			}

			// Encode each M_1 length slice with ECC to length ECCLength
			for j := uint32(0); j < params.N; j++ {
				// Get the row i, col j of each block, forms a length M_1 message, then Encode
				if dataobjects.USE_FAST_CODE {
					tdm.PermutedExtentsAssign(doctx, message, 0, 1, 0, encoded, i*params.N+j, entryPerSlice, 0, 1, nil, 0, uint64(params.M_1))
				} else {
					for t := uint32(0); t < params.M_1; t++ {
						message[t] = encoded[t*entryPerSlice+i*params.N+j]
					}
				}

				MatVecProductExt(doctx, generatorMatrix, 0, 0, message, 0, 0, message, params.M_1, 0, params.ECCLength-params.M_1, params.M_1, 1, params.P)

				// Put to the M_1:ECCLength slice
				if dataobjects.USE_FAST_CODE {
					tdm.PermutedExtentsAssign(doctx, encoded, i*params.N+j, entryPerSlice, 0, message, 0, 1, 0, 1, nil, 0, uint64(params.ECCLength))
				} else {
					for t := params.M_1; t < params.ECCLength; t++ {
						encoded[t*entryPerSlice+i*params.N+j] = message[t]
					}
				}
			}
		}
	}

	// Add Masks
	for i := uint64(0); i < uint64(params.ECCLength); i++ {
		start := i * uint64(entryPerSlice)

		params.Field.AddVectors(encoded, start, encoded, start, masks[i], 0, uint64(entryPerSlice))
	}

	return &dataobjects.Matrix{
		Rows: rowPerSlice * params.ECCLength,
		Cols: params.N,
		Data: encoded,
	}
}

func (lpn *LpnMVP) Query(sk SecretKey, vec []uint32) (*LpnQuery, *LpnAux) {
	doctx := dataobjects.GetDeferralDoContext(lpn.Ctx)
	frame := dataobjects.MakeDeferralFrame(lpn.Ctx)
	defer frame.Close()

	seed := utils.MakeSeed(vec)

	params := lpn.Params

	PofDual := sk.PreLoadedMatrix

	// ECCLength Slice, each with length N
	queryVector := dataobjects.DoAlignedMake[uint32](doctx, uint64(params.N*params.ECCLength))
	masks := dataobjects.DoAlignedMake[uint32](doctx, uint64(params.M*params.ECCLength/params.M_1))

	noisyQueryIndicator := dataobjects.DoAlignedMake[bool](doctx, uint64(params.ECCLength))
	if dataobjects.USE_FAST_CODE {
		e := dataobjects.DoAlignedMake[uint32](doctx, uint64(params.L*params.ECCLength))
		frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, e) })
		bmask := sk.TDM.AllocateMaskForEvaluationCircuitPerSlice(params.ECCLength)
		frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, bmask) })
		bv := sk.TDM.AllocateBlockVectorForEvaluationCircuitPerSlice(params.ECCLength)
		frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, bv) })
		r := dataobjects.DoAlignedMake[uint32](doctx, uint64(params.K*params.ECCLength))
		frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, r) })

		utils.SampleVector(doctx, params.Field, r, params.K*params.ECCLength, seed+1)
		utils.RandomLPNNoiseVector(doctx, e, params.Epsi, params.Field, seed, 1)
		MatVecProductExt(doctx, PofDual, 0, 0, r, 0, params.K, queryVector, 0, params.N, params.L, params.K, params.ECCLength, params.P)

		dataobjects.FieldAddVectorIfNonZeroExt(
			doctx, noisyQueryIndicator, 0, 1, queryVector, 0, uint64(params.N), e, 0, uint64(params.L), uint64(params.L), uint64(params.ECCLength), params.Field.Mod(),
		)

		tdm.PermutedExtentsAssign(doctx, queryVector, params.L, params.N, 0, r, 0, params.K, 0, uint64(params.K), nil, 0, uint64(params.ECCLength)) // relies on N = K + L

		params.Field.AddVectorsExt(queryVector, 0, uint64(params.N), queryVector, 0, uint64(params.N), vec, 0, 0, uint64(params.L), uint64(params.ECCLength))

		bmaskStride := uint32(len(bmask)) / params.ECCLength
		bvStride := uint32(len(bv)) / params.ECCLength
		params.Field.SetVector(bmask, 0, uint64(len(bmask)), 0)
		params.Field.SetVector(bv, 0, uint64(len(bv)), 0)
		// TODO - flatten loop (not convenient)
		for t := uint32(0); t < params.ECCLength; t++ {
			sk.TDM.EvaluationCircuitPerSliceInPlace(bmask, t*bmaskStride, bv, t*bvStride, bvStride, queryVector, t*params.N, params.N, int64(t))
		}
		tdm.PermutedExtentsAssign(
			doctx, masks, 0, params.M/params.M_1, 0, bmask, 0, uint32(len(bmask))/params.ECCLength, 0, uint64(params.M/params.M_1), nil, 0, uint64(params.ECCLength),
		)
	} else {
		e := dataobjects.DoAlignedMake[uint32](doctx, uint64(params.L))
		frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, e) })
		bmask := sk.TDM.AllocateMaskForEvaluationCircuitPerSlice(1)
		frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, bmask) })
		bv := sk.TDM.AllocateBlockVectorForEvaluationCircuitPerSlice(1)
		frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, bv) })
		r := dataobjects.DoAlignedMake[uint32](doctx, uint64(params.K))
		frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, r) })
		for t := uint32(0); t < params.ECCLength; t++ {
			utils.SampleVector(doctx, params.Field, r, params.K, seed+1)
			// e \in Ber(epsi)^L
			utils.RandomLPNNoiseVector(doctx, e, params.Epsi, params.Field, seed, 1)

			MatVecProductExt(doctx, PofDual, 0, 0, r, 0, params.K, queryVector, t*params.N, 0, params.L, params.K, 1, params.P)

			if dataobjects.USE_FAST_CODE {
				dataobjects.FieldAddVectorIfNonZero(doctx, noisyQueryIndicator, uint64(t), queryVector, uint64(t*params.N), e, 0, uint64(params.L), params.Field.Mod())
			} else {
				if !utils.IsZeroVector(e) {
					noisyQueryIndicator[t] = true
					params.Field.AddVectors(queryVector, uint64(t*params.N), queryVector, uint64(t*params.N), e, 0, uint64(params.L))
				}
			}

			if dataobjects.USE_FAST_CODE {
				dataobjects.FieldCopyVector(doctx, queryVector, uint64(t*params.N+params.L), r, 0, uint64(params.K)) // relies on N = K + L
			} else {
				copy(queryVector[t*params.N+params.L:t*params.N+params.N], r[:params.K])
			}

			params.Field.AddVectors(queryVector, uint64(t*params.N), queryVector, uint64(t*params.N), vec, 0, uint64(params.L))
			params.Field.SetVector(bmask, 0, uint64(len(bmask)), 0)
			params.Field.SetVector(bv, 0, uint64(len(bv)), 0)
			mask := sk.TDM.EvaluationCircuitPerSliceInPlace(bmask, 0, bv, 0, uint32(len(bv)), queryVector, t*params.N, params.N, int64(t))

			if dataobjects.USE_FAST_CODE {
				offset := t * params.M / params.M_1
				dataobjects.FieldCopyVector(doctx, masks, uint64(offset), mask, 0, min(uint64(len(masks))-uint64(offset), uint64(len(mask))))
			} else {
				copy(masks[t*params.M/params.M_1:], mask)
			}
		}
	}

	return &LpnQuery{
			Vec:          queryVector,
			QueryLen:     params.N,
			NumOfQueries: params.ECCLength,
		}, &LpnAux{
			NoisyQueryIndicator: noisyQueryIndicator,
			Masks:               masks,
		}
}

func (lpn *LpnMVP) Answer(encodedMatrix *dataobjects.Matrix, clientQuery *LpnQuery) *LpnResponse {
	doctx := dataobjects.GetDeferralDoContext(lpn.Ctx)
	params := lpn.Params

	rowPerSlice := params.M / params.M_1
	entryPerSlice := rowPerSlice * params.N

	answers := dataobjects.DoAlignedMake[uint32](doctx, uint64(rowPerSlice*params.ECCLength))

	if dataobjects.USE_FAST_CODE {
		MatVecProductExt(doctx,
			encodedMatrix.Data, 0, entryPerSlice,
			clientQuery.Vec, 0, clientQuery.QueryLen,
			answers, 0, rowPerSlice,
			rowPerSlice, params.N, params.ECCLength, params.P)
	} else {
		for i := uint32(0); i < params.ECCLength; i++ {
			MatVecProduct(doctx, encodedMatrix.Data[i*entryPerSlice:(i+1)*entryPerSlice],
				clientQuery.Vec[i*clientQuery.QueryLen:(i+1)*clientQuery.QueryLen],
				answers[i*rowPerSlice:(i+1)*rowPerSlice],
				rowPerSlice, params.N, params.P)
		}
	}

	return &LpnResponse{
		Answers: answers,
		AnsLen:  rowPerSlice,
	}
}

func (lpn *LpnMVP) Decode(sk SecretKey, response *LpnResponse, aux *LpnAux) ([]uint32, dataobjects.PossibleFailure) {
	doctx := dataobjects.GetDeferralDoContext(lpn.Ctx)
	frame := dataobjects.MakeDeferralFrame(lpn.Ctx)
	defer frame.Close()

	params := lpn.Params

	// Unmask
	params.Field.SubVectors(response.Answers, 0, response.Answers, 0, aux.Masks, 0, uint64(len(response.Answers)))

	result := dataobjects.DoAlignedMake[uint32](doctx, uint64(params.M))

	var codeLength uint64
	if dataobjects.USE_FAST_CODE {
		codeLength = uint64(params.ECCLength) * uint64(response.AnsLen)
	} else {
		codeLength = uint64(params.ECCLength)
	}
	code := dataobjects.DoAlignedMake[uint32](doctx, codeLength)
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, code) })

	ecccode := ecc.GetECCCode(lpn.Ctx, ecc.ECCConfig{Name: params.ECCName, Q: params.P, N: params.ECCLength, K: params.M_1})

	if dataobjects.USE_FAST_CODE {
		dataobjects.MatrixTranspose(doctx, code, 0, response.Answers, 0, params.ECCLength, response.AnsLen)
	}

	success := dataobjects.DoAlignedMake[uint32](doctx, uint64(response.AnsLen))
	dataobjects.FieldSetVector(doctx, success, 0, uint64(response.AnsLen), 1)
	possibleFailure := dataobjects.PossibleFailure{
		Success: success,
		Err:     make([]error, response.AnsLen),
	}

	if dataobjects.USE_FAST_CODE {
		ecccode.DecodeExt(code, 0, uint64(params.ECCLength), aux.NoisyQueryIndicator, possibleFailure, uint64(response.AnsLen))
		tdm.PermutedExtentsAssign(doctx, result, 0, params.M_1, 0, code, 0, params.ECCLength, 0, uint64(params.M_1), nil, 0, uint64(response.AnsLen))
	} else {
		for i := uint32(0); i < response.AnsLen; i++ {
			var codeRow []uint32
			if dataobjects.USE_FAST_CODE {
				codeRow = code[i*params.ECCLength : (i+1)*params.ECCLength]
			} else {
				for j := uint32(0); j < params.ECCLength; j++ {
					code[j] = response.Answers[j*response.AnsLen+i]
				}
				codeRow = code
			}

			ecccode.Decode(codeRow, aux.NoisyQueryIndicator, possibleFailure, uint64(i))

			if dataobjects.USE_FAST_CODE {
				dataobjects.FieldCopyVector(doctx, result, uint64(i*params.M_1), codeRow, 0, min(uint64(params.M_1), uint64(len(result))-uint64(i*params.M_1), uint64(params.M_1)))
			} else {
				copy(result[i*params.M_1:(i+1)*params.M_1], codeRow[:params.M_1])
			}
		}
	}

	return result, possibleFailure

}

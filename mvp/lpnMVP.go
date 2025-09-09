package mvp

import (
	"RandomLinearCodePIR/dataobjects"
	"RandomLinearCodePIR/ecc"
	"RandomLinearCodePIR/linearcode"
	"RandomLinearCodePIR/tdm"
	"RandomLinearCodePIR/utils"
)

type LpnMVP struct {
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

type LpnAux struct {
	NoisyQueryIndicator []bool
	Masks               []uint32
}

type LpnResponse struct {
	Answers []uint32
	AnsLen  uint32
}

func (lpn *LpnMVP) KeyGen(seed int64) SecretKey {
	params := lpn.Params
	return SecretKey{
		LinearCodeKey:   seed,
		PreLoadedMatrix: linearcode.Generate1DDualMatrix(params.L, params.K, params.Field, seed),
		TDM: &tdm.TDM{
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
	params := lpn.Params
	rlcMatrix := linearcode.Generate1DRLCMatrix(params.L, params.K, params.Field, sk.LinearCodeKey)

	// Assume M_1 | M for now
	rowPerSlice := params.M / params.M_1
	entryPerSlice := rowPerSlice * params.N

	encoded := dataobjects.AlignedMake[uint32](uint64(entryPerSlice * params.ECCLength))

	// Re-use slot for ECC encoding
	message := dataobjects.AlignedMake[uint32](uint64(params.ECCLength))

	generatorMatrix := ecc.GetECCCode(ecc.ECCConfig{
		Name: params.ECCName,
		Q:    params.P,
		N:    params.ECCLength,
		K:    params.M_1}).GetGeneratorMatrix(params.M_1, params.ECCLength, params.P)

	for i := uint32(0); i < rowPerSlice; i++ {
		for j := uint32(0); j < params.M_1; j++ {
			// Input matrix with each row length L, block size M_1
			inputStart := (i*params.M_1 + j) * params.L

			// Put into the jth slice, ith row, each row with length N
			outputStart := j*entryPerSlice + i*params.N

			// Copy the input row into the first L element of the output row
			copy(encoded[outputStart:outputStart+params.L], input.Data[inputStart:inputStart+params.L])

			MatVecProduct(rlcMatrix, input.Data[inputStart:inputStart+input.Cols], encoded[outputStart+params.L:outputStart+params.N],
				params.K, params.L, params.P)
		}

		// Encode each M_1 length slice with ECC to length ECCLength
		for j := uint32(0); j < params.N; j++ {
			// Get the row i, col j of each block, forms a length M_1 message, then Encode
			for t := uint32(0); t < params.M_1; t++ {
				message[t] = encoded[t*entryPerSlice+i*params.N+j]
			}

			MatVecProduct(generatorMatrix, message, message[params.M_1:], params.ECCLength, params.M_1, params.P)

			// Put to the M_1:ECCLength slice
			for t := params.M_1; t < params.ECCLength; t++ {
				encoded[t*entryPerSlice+i*params.N+j] = message[t]
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
	params := lpn.Params

	PofDual := sk.PreLoadedMatrix
	if len(PofDual) == 0 {
		PofDual = linearcode.Generate1DDualMatrix(params.L, params.K, params.Field, sk.LinearCodeKey)
	}

	// ECCLength Slice, each with length N
	queryVector := dataobjects.AlignedMake[uint32](uint64(params.N * params.ECCLength))
	masks := dataobjects.AlignedMake[uint32](uint64(params.M * params.ECCLength / params.M_1))

	noisyQueryIndicator := make([]bool, params.ECCLength)

	for t := uint32(0); t < params.ECCLength; t++ {
		r := params.Field.SampleVector(params.K)
		// e \in Ber(epsi)^L
		e := utils.RandomLPNNoiseVector(params.L, params.Epsi, params.Field)

		MatVecProduct(PofDual, r, queryVector[t*params.N:], params.L, params.K, params.P)

		if !utils.IsZeroVector(e) {
			noisyQueryIndicator[t] = true
			params.Field.AddVectors(queryVector, uint64(t*params.N), queryVector, uint64(t*params.N), e, 0, uint64(params.L))
		}

		copy(queryVector[t*params.N+params.L:t*params.N+params.N], r[:params.K])

		params.Field.AddVectors(queryVector, uint64(t*params.N), queryVector, uint64(t*params.N), vec, 0, uint64(params.L))
		mask := sk.TDM.EvaluationCircuitPerSlice(queryVector[t*params.N:(t+1)*params.N], int64(t))

		copy(masks[t*params.M/params.M_1:], mask)
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
	params := lpn.Params

	rowPerSlice := params.M / params.M_1
	entryPerSlice := rowPerSlice * params.N

	answers := dataobjects.AlignedMake[uint32](uint64(rowPerSlice * params.ECCLength))

	for i := uint32(0); i < params.ECCLength; i++ {
		MatVecProduct(encodedMatrix.Data[i*entryPerSlice:(i+1)*entryPerSlice],
			clientQuery.Vec[i*clientQuery.QueryLen:(i+1)*clientQuery.QueryLen],
			answers[i*rowPerSlice:(i+1)*rowPerSlice],
			rowPerSlice, params.N, params.P)
	}

	return &LpnResponse{
		Answers: answers,
		AnsLen:  rowPerSlice,
	}
}

func (lpn *LpnMVP) Decode(sk SecretKey, response *LpnResponse, aux *LpnAux) []uint32 {
	params := lpn.Params

	// Unmask
	params.Field.SubVectors(response.Answers, 0, response.Answers, 0, aux.Masks, 0, uint64(len(response.Answers)))

	result := dataobjects.AlignedMake[uint32](uint64(params.M))

	code := dataobjects.AlignedMake[uint32](uint64(params.ECCLength))

	ecccode := ecc.GetECCCode(ecc.ECCConfig{Name: params.ECCName, Q: params.P, N: params.ECCLength, K: params.M_1})

	for i := uint32(0); i < response.AnsLen; i++ {
		for j := uint32(0); j < params.ECCLength; j++ {
			code[j] = response.Answers[j*response.AnsLen+i]
		}

		message, err := ecccode.Decode(code, aux.NoisyQueryIndicator)

		if err != nil {
			panic(err)
		}

		copy(result[i*params.M_1:(i+1)*params.M_1], message)
	}

	return result

}

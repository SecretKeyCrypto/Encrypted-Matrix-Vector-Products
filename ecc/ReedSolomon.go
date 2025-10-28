package ecc

import (
	"RandomLinearCodePIR/dataobjects"
	"errors"
)

type ReedSolomonCode struct {
	k uint32
	n uint32
	q uint32
}

func NewReedSolomonCode(k, n, q uint32) *ReedSolomonCode {
	return &ReedSolomonCode{
		k: k,
		n: n,
		q: q,
	}
}

// Only return the evaluation part
func (rs *ReedSolomonCode) GetGeneratorMatrix(M_1, ECCLength, p uint32) []uint32 {
	alphas := getAlphas(ECCLength)
	defer dataobjects.Aligned1DFree(alphas)
	rsGeneratorMatrix := dataobjects.AlignedMake[uint32](uint64(M_1 * ECCLength))

	GenerateSystematicRSMatrix(ECCLength, M_1, p, M_1, alphas, rsGeneratorMatrix)
	return rsGeneratorMatrix
}

func getAlphas(ECCLength uint32) []uint32 {
	alphas := dataobjects.AlignedMake[uint32](uint64(ECCLength))
	for i := range alphas {
		alphas[i] = uint32(i)
	}
	return alphas
}

func (rs *ReedSolomonCode) Decode(code []uint32, noisyQuery []bool) ([]uint32, error) {
	if !isAllFalse(noisyQuery[:rs.k]) {
		x_in := dataobjects.AlignedMake[uint32](uint64(rs.k))
		defer dataobjects.Aligned1DFree(x_in)
		y_in := dataobjects.AlignedMake[uint32](uint64(rs.k))
		defer dataobjects.Aligned1DFree(y_in)
		idx := uint32(0)
		for i := range noisyQuery {
			if !noisyQuery[i] && idx < rs.k {
				x_in[idx] = uint32(i)
				y_in[idx] = code[i]
				idx += 1
			}
		}

		if idx < rs.k {
			return []uint32{}, errors.New("decoding failed due to not enough data")
		}

		// TODO: Replace it with ReedSolomon Decoder
		for i := uint32(0); i < rs.k; i++ {
			if noisyQuery[i] {
				code[i] = LagrangeInterpEval(x_in, y_in, rs.k, i, rs.q)
			}
		}
	}
	// The slice starting from 0 is critical for it to be correctly freed
	return code[:rs.k], nil
}

// FIXME - make fast on GPU
func isAllFalse(vec []bool) bool {
	for i := range vec {
		if vec[i] {
			return false
		}
	}
	return true
}

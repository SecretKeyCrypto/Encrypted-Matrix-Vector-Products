package ecc

import (
	"RandomLinearCodePIR/dataobjects"
	"context"
	"errors"
)

type ReedSolomonCode struct {
	ctx context.Context
	k   uint32
	n   uint32
	q   uint32
}

func NewReedSolomonCode(ctx context.Context, k, n, q uint32) *ReedSolomonCode {
	return &ReedSolomonCode{
		ctx: ctx,
		k:   k,
		n:   n,
		q:   q,
	}
}

// Only return the evaluation part
func (rs *ReedSolomonCode) GetGeneratorMatrix(M_1, ECCLength, p uint32) []uint32 {
	frame := dataobjects.MakeDeferralFrame(rs.ctx)
	defer frame.Close()

	alphas := dataobjects.AlignedMake[uint32](uint64(ECCLength))
	dataobjects.FieldRangeVector(alphas, 0, 0, uint64(ECCLength))
	frame.Defer(func() { dataobjects.Aligned1DFree(alphas) })
	rsGeneratorMatrix := dataobjects.AlignedMake[uint32](uint64(M_1 * ECCLength))

	// FIXME - make fast on GPU
	GenerateSystematicRSMatrix(ECCLength, M_1, p, M_1, alphas, rsGeneratorMatrix)
	return rsGeneratorMatrix
}

func (rs *ReedSolomonCode) Decode(code []uint32, noisyQuery []bool) ([]uint32, error) {
	// FIXME - make fast on GPU
	if !isAllFalse(noisyQuery[:rs.k]) {
		frame := dataobjects.MakeDeferralFrame(rs.ctx)
		defer frame.Close()

		x_in := dataobjects.AlignedMake[uint32](uint64(rs.k))
		frame.Defer(func() { dataobjects.Aligned1DFree(x_in) })
		y_in := dataobjects.AlignedMake[uint32](uint64(rs.k))
		frame.Defer(func() { dataobjects.Aligned1DFree(y_in) })
		idx := uint32(0)
		for i := range noisyQuery {
			if !noisyQuery[i] && idx < rs.k {
				x_in[idx] = uint32(i)
				y_in[idx] = code[i]
				idx += 1
			}
		}

		if idx < rs.k {
			return nil, errors.New("decoding failed due to not enough data")
		}

		// TODO: Replace it with ReedSolomon Decoder
		for i := uint32(0); i < rs.k; i++ {
			if noisyQuery[i] {
				// FIXME - make fast on GPU
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

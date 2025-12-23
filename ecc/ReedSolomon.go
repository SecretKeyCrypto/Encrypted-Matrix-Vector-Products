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
	doctx := dataobjects.GetDeferralDoContext(rs.ctx)
	frame := dataobjects.MakeDeferralFrame(rs.ctx)
	defer frame.Close()

	alphas := dataobjects.DoAlignedMake[uint32](doctx, uint64(ECCLength))
	dataobjects.FieldRangeVector(doctx, alphas, 0, 0, uint64(ECCLength))
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, alphas) })
	rsGeneratorMatrix := dataobjects.DoAlignedMake[uint32](doctx, uint64(M_1*ECCLength))

	GenerateSystematicRSMatrix(doctx, ECCLength, M_1, p, alphas, rsGeneratorMatrix)
	return rsGeneratorMatrix
}

func (rs *ReedSolomonCode) DecodeExt(code []uint32, co, cs uint64, noisyQuery []bool, possibleFailure dataobjects.PossibleFailure, steps uint64) {
	if dataobjects.USE_FAST_CODE {
		doctx := dataobjects.GetDeferralDoContext(rs.ctx)
		ReedSolomonDecode(doctx, code, co, cs, noisyQuery, uint32(len(noisyQuery)), rs.k, rs.q, possibleFailure.Success, steps)
	} else {
		for s := range steps {
			rs.Decode(code[co+s*cs:co+(s+1)*cs], noisyQuery, possibleFailure, s)
		}
	}
}

func (rs *ReedSolomonCode) Decode(code []uint32, noisyQuery []bool, possibleFailure dataobjects.PossibleFailure, p_index uint64) {
	doctx := dataobjects.GetDeferralDoContext(rs.ctx)
	frame := dataobjects.MakeDeferralFrame(rs.ctx)
	defer frame.Close()

	dataobjects.FieldSetVector(doctx, possibleFailure.Success, 0, 1, 1)

	possibleFailure.Err[p_index] = errors.New("decoding failed due to not enough data")

	if dataobjects.USE_FAST_CODE {
		ReedSolomonDecode(doctx, code, 0, 0, noisyQuery, uint32(len(noisyQuery)), rs.k, rs.q, possibleFailure.Success, 1)
		return
	}
	if !isAllFalse(noisyQuery[:rs.k]) {
		x_in := dataobjects.DoAlignedMake[uint32](doctx, uint64(rs.k))
		frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, x_in) })
		y_in := dataobjects.DoAlignedMake[uint32](doctx, uint64(rs.k))
		frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, y_in) })
		idx := uint32(0)
		for i := range noisyQuery {
			if !noisyQuery[i] && idx < rs.k {
				x_in[idx] = uint32(i)
				y_in[idx] = code[i]
				idx += 1
			}
		}

		if idx < rs.k {
			possibleFailure.Success[p_index] = 0
			return
		}

		// TODO: Replace it with ReedSolomon Decoder
		for i := uint32(0); i < rs.k; i++ {
			if noisyQuery[i] {
				LagrangeInterpEval(doctx, code[i:i+1], x_in, y_in, rs.k, i, rs.q)
			}
		}
	}
}

func isAllFalse(vec []bool) bool {
	for i := range vec {
		if vec[i] {
			return false
		}
	}
	return true
}

package ecc

import (
	"RandomLinearCodePIR/dataobjects"
	"context"
)

const (
	ReedSolomon = "ReedSolomon"
)

type ECCConfig struct {
	Name string
	Q    uint32
	N    uint32
	K    uint32
}

type ErasureCorrectionCode interface {
	GetGeneratorMatrix(k, n, p uint32) []uint32
	Decode(code []uint32, noisyIndicator []bool, possibleFailure dataobjects.PossibleFailure, p_index uint64)
	DecodeExt(code []uint32, co, cs uint64, noisyIndicator []bool, possibleFailure dataobjects.PossibleFailure, steps uint64)
}

func GetECCCode(ctx context.Context, config ECCConfig) ErasureCorrectionCode {
	switch config.Name {
	case ReedSolomon:
		return NewReedSolomonCode(ctx, config.K, config.N, config.Q)
	default:
		panic("Unsupported ECC code: " + config.Name)
	}
}

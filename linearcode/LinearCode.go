package linearcode

import "RandomLinearCodePIR/dataobjects"

const (
	RandomLinearCode = "Random"
	Vandermonde      = "Fast"
)

type LinearCodeConfig struct {
	Name  string
	K     uint32
	L     uint32
	Field dataobjects.Field
}

type LinearCode interface {
	Generate1DDualMatrix(L, K uint32, field dataobjects.Field, seed int64) []uint32
	Generate1DRLCMatrix(L, K uint32, p dataobjects.Field, seed int64) []uint32
	EncodeLSN(message []uint32) []uint32
	EncodeDual(message []uint32) []uint32
	EncodeLength() uint32
}

func GetLinearCode(config LinearCodeConfig) LinearCode {
	switch config.Name {
	case Vandermonde:
		return NewEvaluationCode(config.K, config.L, config.Field)
	default:
		panic("Unsupported Linear Code: " + config.Name)
	}
}

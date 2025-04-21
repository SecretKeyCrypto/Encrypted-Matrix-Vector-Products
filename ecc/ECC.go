package ecc

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
	Decode(code []uint32, noisyIndicator []bool) ([]uint32, error)
}

func GetECCCode(config ECCConfig) ErasureCorrectionCode {
	switch config.Name {
	case ReedSolomon:
		return NewReedSolomonCode(config.K, config.N, config.Q)
	default:
		panic("Unsupported ECC code: " + config.Name)
	}
}

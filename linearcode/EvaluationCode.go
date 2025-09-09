package linearcode

import (
	"RandomLinearCodePIR/dataobjects"
	"RandomLinearCodePIR/tdm"
	"math"
)

type EvaluationCode struct {
	K     uint32
	L     uint32
	Field dataobjects.Field
	n     uint32
	omega uint32
}

func NewEvaluationCode(K, L uint32, field dataobjects.Field) *EvaluationCode {
	p := field.GetChar()
	if K > p-1 || L > p-1 {
		panic("Currently Only support for K < P to have enough evaluation points")
	}
	n := uint32(1) << uint32(math.Ceil(math.Log2(float64(max(L, K)))))
	return &EvaluationCode{K: K, L: L,
		Field: field,
		n:     n,
		omega: tdm.NthRootOfUnity(field.GetChar(), n),
	}
}

func (ec *EvaluationCode) Generate1DDualMatrix(L, K uint32, field dataobjects.Field, seed int64) []uint32 {
	return []uint32{}
}

// C = (V // L) where V is L x K, each row is an evalution point
func (ec *EvaluationCode) Generate1DRLCMatrix(L, K uint32, p dataobjects.Field, seed int64) []uint32 {
	return []uint32{}
}

func (ec *EvaluationCode) GenerateV() [][]uint32 {
	V := make([][]uint32, ec.L)
	// k := uint32(math.Ceil(math.Log2(float64(max(ec.L,ec.K)))))

	// omega := tdm.NthRootOfUnity(ec.q, k)
	return V
}

func (ec *EvaluationCode) encode(message []uint32) []uint32 {
	l := len(message)
	if l < int(ec.n) {
		newMessage := dataobjects.AlignedMake[uint32](uint64(ec.n))
		copy(newMessage, message)
		message = newMessage
	}
	tdm.NTT(message, ec.n, ec.omega, ec.Field.GetChar())
	return message
}

// Dual Code C = (I//-V) -V has dimension K x L
func (ec *EvaluationCode) EncodeDual(message []uint32) []uint32 {
	encoded := ec.encode(message)[:ec.K]
	ec.Field.NegVector(encoded, 0, uint64(len(encoded)))
	return encoded
}

// Dual Code D = (V//I)
func (ec *EvaluationCode) EncodeLSN(message []uint32) []uint32 {
	return ec.encode(message)[:ec.L]
}

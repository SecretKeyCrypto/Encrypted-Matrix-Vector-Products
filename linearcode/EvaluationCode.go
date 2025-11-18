package linearcode

import (
	"RandomLinearCodePIR/dataobjects"
	"RandomLinearCodePIR/tdm"
	"context"
	"math"
)

type EvaluationCode struct {
	Ctx   context.Context
	K     uint32
	L     uint32
	Field dataobjects.Field
	n     uint32
	omega uint32
}

func NewEvaluationCode(ctx context.Context, K, L uint32, field dataobjects.Field) *EvaluationCode {
	p := field.GetChar()
	if K > p-1 || L > p-1 {
		panic("Currently Only support for K < P to have enough evaluation points")
	}
	n := uint32(1) << uint32(math.Ceil(math.Log2(float64(max(L, K)))))
	return &EvaluationCode{
		Ctx:   ctx,
		K:     K,
		L:     L,
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

/*
func (ec *EvaluationCode) GenerateV() [][]uint32 {
	V := make([][]uint32, ec.L)
	// k := uint32(math.Ceil(math.Log2(float64(max(ec.L,ec.K)))))

	// omega := tdm.NthRootOfUnity(ec.q, k)
	return V
}
*/

func (ec *EvaluationCode) encode(message []uint32, mo, stride, steps uint32) {
	l := len(message) - int(mo)
	if l < int(ec.n) {
		panic("encoding requires a longer message target")
	}
	doctx := dataobjects.GetDeferralDoContext(ec.Ctx)
	tdm.NTT(doctx, message, mo, ec.n, stride, steps, ec.omega, ec.Field.GetChar())
}

// Dual Code C = (I//-V) -V has dimension K x L
func (ec *EvaluationCode) EncodeDual(message []uint32, mo, stride, steps uint32) {
	ec.encode(message, mo, stride, steps)
	ec.Field.NegVectorExt(message, uint64(mo), uint64(ec.K), uint64(stride), uint64(steps))
}

// Dual Code D = (V//I)
func (ec *EvaluationCode) EncodeLSN(message []uint32, mo, stride, steps uint32) {
	ec.encode(message, mo, stride, steps)
}

func (ec *EvaluationCode) EncodeLength() uint32 {
	return ec.n
}

package dataobjects

type Field interface {
	Add(a, b uint32) uint32
	Mul(a, b uint32) uint32
	Sub(a, b uint32) uint32
	Neg(a uint32) uint32
	Inv(a uint32) uint32
	Mod() uint32
	SetVector(r []uint32, ro uint64, length uint64, v uint32)
	AddVectors(r []uint32, ro uint64, a []uint32, ao uint64, b []uint32, bo uint64, length uint64)
	AddVectorsExt(r []uint32, ro, rs uint64, a []uint32, ao, as uint64, b []uint32, bo, bs uint64, length, steps uint64)
	MulVector(r []uint32, ro uint64, a []uint32, ao uint64, b uint32, length uint64)
	MulVectorsExt(r []uint32, ro, rs uint64, a []uint32, ao, as uint64, b []uint32, bo, bs uint64, length, steps uint64)
	SubVectors(r []uint32, ro uint64, a []uint32, ao uint64, b []uint32, bo uint64, length uint64)
	NegVector(r []uint32, ro uint64, length uint64)
	NegVectorExt(r []uint32, ro uint64, length, stride, steps uint64)
	GetChar() uint32
	InvertVector(vec, inv []uint32)
}

type PrimeField struct {
	doctx *DoContext
	p     uint32
}

func NewPrimeField(doctx *DoContext, p uint32) PrimeField {
	return PrimeField{doctx: doctx, p: p}
}

func (f PrimeField) Add(a, b uint32) uint32 {
	return uint32((uint64(a) + uint64(b)) % uint64(f.p))
}

func (f PrimeField) Mul(a, b uint32) uint32 {
	return uint32((uint64(a) * uint64(b)) % uint64(f.p))
}

func (f PrimeField) Sub(a, b uint32) uint32 {
	return uint32((uint64(a) + uint64(f.p) - uint64(b)) % uint64(f.p))
}

func (f PrimeField) Neg(a uint32) uint32 {
	return (f.p - a) % f.p
}

func (f PrimeField) Inv(a uint32) uint32 {
	var t, newT int64 = 0, 1
	var r, newR int64 = int64(f.p), int64(a)

	for newR != 0 {
		quotient := r / newR
		t, newT = newT, t-quotient*newT
		r, newR = newR, r-quotient*newR
	}

	if r > 1 {
		panic("a is not invertible")
	}

	if t < 0 {
		t += int64(f.p)
	}

	return uint32(t)
}

func (f PrimeField) Mod() uint32 {
	return f.p
}

func (f PrimeField) GetChar() uint32 {
	return f.p
}

func (f PrimeField) SetVector(r []uint32, ro uint64, length uint64, v uint32) {
	if USE_FAST_CODE {
		FieldSetVector(f.doctx, r, ro, length, v)
	} else {
		for i := uint64(0); i < length; i++ {
			r[ro+i] = v
		}
	}
}

func (f PrimeField) AddVectors(r []uint32, ro uint64, a []uint32, ao uint64, b []uint32, bo uint64, length uint64) {
	if USE_FAST_CODE {
		FieldAddVectors(f.doctx, r, ro, a, ao, b, bo, length, f.p)
		// r := FieldAddVectors(f.doctx, r, ro, a, ao, b, bo, length, f.p)
		// fmt.Printf("************ %d\n", r)
	} else {
		for i := uint64(0); i < length; i++ {
			r[ro+i] = uint32(uint64(a[ao+i]) + uint64(b[bo+i])%uint64(f.p))
		}
	}
}

func (f PrimeField) AddVectorsExt(r []uint32, ro, rs uint64, a []uint32, ao, as uint64, b []uint32, bo, bs uint64, length, steps uint64) {
	if USE_FAST_CODE {
		FieldAddVectorsExt(f.doctx, r, ro, rs, a, ao, as, b, bo, bs, length, steps, f.p)
	} else {
		for s := uint64(0); s < steps; s++ {
			for i := uint64(0); i < length; i++ {
				r[ro+s*rs+i] = uint32(uint64(a[ao+s*as+i]) + uint64(b[bo+s*bs+i])%uint64(f.p))
			}
		}
	}
}

func (f PrimeField) MulVector(r []uint32, ro uint64, a []uint32, ao uint64, b uint32, length uint64) {
	if USE_FAST_CODE {
		FieldMulVector(f.doctx, r, ro, a, ao, b, length, f.p)
	} else {
		for i := uint64(0); i < length; i++ {
			r[ro+i] = uint32((uint64(a[ao+i]) * uint64(b)) % uint64(f.p))
		}
	}
}

func (f PrimeField) MulVectorsExt(r []uint32, ro, rs uint64, a []uint32, ao, as uint64, b []uint32, bo, bs uint64, length, steps uint64) {
	if USE_FAST_CODE {
		FieldMulVectorsExt(f.doctx, r, ro, rs, a, ao, as, b, bo, bs, length, steps, f.p)
	} else {
		for s := uint64(0); s < steps; s++ {
			for i := uint64(0); i < length; i++ {
				r[ro+s*rs+i] = uint32(uint64(a[ao+s*as+i]) * uint64(b[bo+s*bs+i]) % uint64(f.p))
			}
		}
	}
}

func (f PrimeField) SubVectors(r []uint32, ro uint64, a []uint32, ao uint64, b []uint32, bo uint64, length uint64) {
	if USE_FAST_CODE {
		FieldSubVectors(f.doctx, r, ro, a, ao, b, bo, length, f.p)
	} else {
		for i := uint64(0); i < length; i++ {
			r[ro+i] = uint32((uint64(a[ao+i]) + uint64(f.p) - uint64(b[bo+i])) % uint64(f.p))
		}
	}
}

func (f PrimeField) NegVector(r []uint32, ro uint64, length uint64) {
	if USE_FAST_CODE {
		FieldNegVector(f.doctx, r, ro, length, f.p)
	} else {
		for i := uint64(0); i < length; i++ {
			r[ro+i] = (f.p - r[ro+i]) % f.p
		}
	}
}

func (f PrimeField) NegVectorExt(r []uint32, ro uint64, length, stride, steps uint64) {
	if USE_FAST_CODE {
		FieldNegVectorExt(f.doctx, r, ro, length, stride, steps, f.p)
	} else {
		for i := uint64(0); i < steps; i++ {
			f.NegVector(r, ro+i*stride, length)
		}
	}
}

func (f PrimeField) InvertVector(vec, inv []uint32) {
	if USE_FAST_CODE {
		FieldInvVector(f.doctx, inv, 0, vec, 0, uint64(len(vec)), f.p)
	} else {
		for i := range vec {
			inv[i] = f.Inv(vec[i])
		}
	}
}

type RingZ2k struct {
	k    uint32
	mask uint32
}

func NewRingZ2k(k uint32) *RingZ2k {
	return &RingZ2k{k: k, mask: (1 << k) - 1}
}

func (r *RingZ2k) Add(a, b uint32) uint32      { return (a + b) & r.mask }
func (r *RingZ2k) Sub(a, b uint32) uint32      { return (a - b) & r.mask }
func (r *RingZ2k) Mul(a, b uint32) uint32      { return (a * b) & r.mask }
func (r *RingZ2k) Neg(a uint32) uint32         { return (-a) & r.mask }
func (r *RingZ2k) Inv(a uint32) uint32         { panic("not implemented") }
func (r *RingZ2k) Mod() uint32                 { return 1 << r.k }
func (r *RingZ2k) SampleInvertibleVec() uint32 { return 1 << r.k }

package utils

import (
	"RandomLinearCodePIR/dataobjects"
	"math/rand"
)

func MakeSeed(v []uint32) int64 {
	// FIXME - add plain and cuda versions of MakeSeed
	return 29381 * int64(len(v))
	// h := fnv.New64a()
	// buf := make([]byte, 4)
	// for _, x := range v {
	// 	binary.LittleEndian.PutUint32(buf, x)
	// 	h.Write(buf)
	// }
	// return int64(h.Sum64())
}

// only used in slow path
func SampleElement(f dataobjects.Field) uint32 {
	return uint32(rand.Intn(int(f.Mod())))
}

// only used in slow path
func SampleElementWithSeed(f dataobjects.Field, rng *rand.Rand) uint32 {
	return uint32(rng.Intn(int(f.Mod())))
}

func SampleInvertibleVec(doctx *dataobjects.DoContext, f dataobjects.Field, vec []uint32, n uint32, seed int64) {
	p := f.Mod()
	if dataobjects.USE_FAST_CODE {
		RandomizeVectorWithModulusAndSeed(doctx, vec, n, 1, false, false, p-1, seed, 0)
		dataobjects.FieldAddToVector(doctx, vec, 0, 1, uint64(n))
	} else {
		rng := rand.New(rand.NewSource(seed))
		for i := range vec {
			vec[i] = uint32(rng.Intn(int(p)-1) + 1)
		}
	}
}

func SampleVector(doctx *dataobjects.DoContext, f dataobjects.Field, vec []uint32, n uint32, seed int64) {
	p := f.Mod()
	if dataobjects.USE_FAST_CODE {
		RandomizeVectorWithModulusAndSeed(doctx, vec, n, 1, false, false, p, seed, 0)
	} else {
		rng := rand.New(rand.NewSource(seed))
		for i := range vec {
			vec[i] = uint32(rng.Intn(int(p)))
		}
	}
}

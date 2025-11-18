package utils

import (
	"RandomLinearCodePIR/dataobjects"
	"math/rand"
)

func GetPermutation(doctx *dataobjects.DoContext, perm []uint32, n uint32, seed, offset int64) {
	if dataobjects.USE_FAST_CODE {
		random_permutation(doctx, perm, n, seed, offset)
	} else {
		rng := rand.New(rand.NewSource(seed))
		for i := uint32(0); i < n; i++ {
			perm[i] = i
		}
		rng.Shuffle(int(n), func(i, j int) {
			perm[i], perm[j] = perm[j], perm[i]
		})
	}
}

package dataobjects

import (
	"encoding/binary"
	"fmt"
	"hash/fnv"
	"runtime"
)

const USE_COMMON_DEBUG = false // if true, consider bypassing cuda in rnd_api.cpp
const USE_FAST_CODE bool = true

type Primitive interface {
	~bool |
		~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr |
		~float32 | ~float64 |
		~complex64 | ~complex128 |
		~string
}

type Array[T Primitive] []T
type Array2[T Primitive] [][]T

type Vector Array[uint32]
type Vector2 Array2[uint32]

type PossibleFailure struct {
	Success []uint32
	Err     []error
}

func hashUint32Slice(arr []uint32) uint64 {
	h := fnv.New64a()
	var buf [4]byte
	for _, v := range arr {
		binary.LittleEndian.PutUint32(buf[:], v)
		h.Write(buf[:])
	}
	return h.Sum64()
}

func hashBoolSlice(arr []bool) uint64 {
	h := fnv.New64a()
	var buf [1]byte
	for _, v := range arr {
		if v {
			buf[0] = 1
		} else {
			buf[0] = 0
		}
		h.Write(buf[:])
	}
	return h.Sum64()
}

func KeepAlive(name string, a any) {
	if USE_COMMON_DEBUG {
		fmt.Printf("=== %s:", name)
		switch t := a.(type) {
		case []uint32:
			r := a.([]uint32)
			fmt.Printf(" [%d|%d]", len(r), hashUint32Slice(r))
			for i := int(0); i < min(5, len(r)); i++ {
				fmt.Printf(" %d=%d", i, r[i])
			}
			for i := max(0, len(r)-5); i < len(r); i++ {
				fmt.Printf(" %d=%d", i, r[i])
			}
			fmt.Println()
		case []bool:
			r := a.([]bool)
			fmt.Printf(" [%d|%d]", len(r), hashBoolSlice(r))
			for i := int(0); i < min(5, len(r)); i++ {
				fmt.Printf(" %d=%t", i, r[i])
			}
			for i := max(0, len(r)-5); i < len(r); i++ {
				fmt.Printf(" %d=%t", i, r[i])
			}
			fmt.Println()
		default:
			fmt.Println(" ?")
			runtime.KeepAlive(t)
		}
	}
	runtime.KeepAlive(a)
}

func CheckResult(result bool) bool {
	if USE_COMMON_DEBUG {
		fmt.Println("=== DONE")
	}
	if !result {
		panic("Result-check failure")
	}
	return result
}

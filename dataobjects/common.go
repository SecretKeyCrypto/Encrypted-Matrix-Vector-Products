package dataobjects

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

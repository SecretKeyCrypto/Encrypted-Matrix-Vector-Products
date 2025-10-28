package dataobjects

type Matrix struct {
	Rows uint32
	Cols uint32
	Data []uint32
}

func (matrix *Matrix) Free() {
	if matrix.Data != nil {
		matrix.Data = Aligned1DFree(matrix.Data)
	}
}

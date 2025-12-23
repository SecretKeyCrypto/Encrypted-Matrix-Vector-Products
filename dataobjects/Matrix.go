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

func (matrix *Matrix) DoFree(doctx *DoContext) bool {
	if matrix.Data != nil {
		if !DoAligned1DFree(doctx, matrix.Data) {
			return false
		}
		matrix.Data = nil
	}
	return true
}

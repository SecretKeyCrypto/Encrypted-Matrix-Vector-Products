package pir

import (
	"RandomLinearCodePIR/utils"
	"fmt"
	"math/rand"
	"testing"
	"time"
)

/*
	 ================================================================================
					Test and Benchmarks for 2D Split LSN based PIR
	   ================================================================================
*/
func TestBasePIR(t *testing.T) {
	lambda := uint32(32)
	row := uint32(1 << 8)
	col := uint32(1 << 6)

	pi := &BasePIR{
		Params: BaseParams{
			Rows:           row,
			Cols:           col,
			NumberOfBlocks: uint32(16),
			CodewordLength: col + lambda,
		},
	}

	matrix := GenerateMatrix(pi.Params.Rows, pi.Params.Cols, 1, 1)
	queryIndex := uint64(rand.Intn(int(row) * int(col)))

	fmt.Printf("Running PIR with Database %d * %d \n", pi.Params.Rows, pi.Params.Cols)

	fmt.Println("Generate Key...")
	start := time.Now()
	sk := pi.KeyGen(1, 2, 32, 1)
	fmt.Println("    Elapsed: ", time.Since(start))

	fmt.Println("Encode Message...")
	start = time.Now()
	encodedMatrix := pi.Encode(sk, matrix)
	fmt.Println("    Elapsed: ", time.Since(start))

	fmt.Println("Generate Query...")
	start = time.Now()
	clientQuery, aux := pi.Query(sk, queryIndex)
	fmt.Println("    Elapsed: ", time.Since(start))

	fmt.Println("Answer...")
	start = time.Now()
	serverResponse := pi.Answer(encodedMatrix, clientQuery)
	fmt.Println("    Elapsed: ", time.Since(start))

	fmt.Println("Decode...")
	start = time.Now()
	val := pi.Decode(sk, queryIndex, serverResponse, aux)
	fmt.Println("    Elapsed: ", time.Since(start))

	if val != matrix.Data[queryIndex] {
		fmt.Println(val)
		panic("INCORRECT RESULT!")
	}
}

func TestMixedSLSNPIR(t *testing.T) {
	lambda := uint32(32)
	row := uint32(1 << 8)
	col := uint32(1 << 6)

	pi := &MixedSLSNPIR{
		Params: MixedSLSNParams{
			Rows:           row,
			Cols:           col,
			NumberOfBlocks: uint32(2),
			CodewordLength: col + lambda,
		},
	}

	matrix := GenerateMatrixF4(pi.Params.Rows, pi.Params.Cols, 1, 1)

	queryIndex := uint64(rand.Intn(int(row) * int(col)))

	fmt.Printf("Running PIR with Database %d * %d \n", pi.Params.Rows, pi.Params.Cols)

	fmt.Println("Generate Key...")
	start := time.Now()
	sk := pi.KeyGen(1, 2, 32, 1)
	fmt.Println("    Elapsed: ", time.Since(start))

	fmt.Println("Encode Message...")
	start = time.Now()
	encodedMatrix := pi.Encode(sk, matrix)
	fmt.Println("    Elapsed: ", time.Since(start))

	fmt.Println("Generate Query...")
	start = time.Now()
	clientQuery, aux := pi.Query(sk, queryIndex)
	fmt.Println("    Elapsed: ", time.Since(start))

	fmt.Println("Answer...")
	start = time.Now()
	serverResponse := pi.Answer(encodedMatrix, clientQuery)
	fmt.Println("    Elapsed: ", time.Since(start))

	fmt.Println("Decode...")
	start = time.Now()
	valBit1, valBitP := pi.Decode(sk, queryIndex, serverResponse, aux)
	fmt.Println("    Elapsed: ", time.Since(start))

	if valBit1 != matrix.Bit1[queryIndex] || valBitP != matrix.BitP[queryIndex] {
		fmt.Printf("Want (%d, %d) But get (%d, %d) of index %d \n",
			matrix.Bit1[queryIndex], matrix.BitP[queryIndex], valBit1, valBitP, queryIndex)
		panic("INCORRECT RESULT!")
	}
}

func BenchmarkQueryGeneration(b *testing.B) {
	row, col, k, block := getParams()

	pi := &BasePIR{
		Params: BaseParams{
			Rows:           row,
			Cols:           col,
			NumberOfBlocks: block,
			CodewordLength: col + k,
		},
	}

	sk := pi.KeyGen(1, 2, 32, 1)

	var totalDuration time.Duration
	b.ResetTimer()
	pirQuery := &BasePIRQuery{}

	for i := 0; i < b.N; i++ {
		queryIndex := uint64(rand.Intn(int(row) * int(col)))
		start := time.Now()
		pirQuery, _ = pi.Query(sk, queryIndex)
		totalDuration += time.Since(start)
	}

	b.StopTimer()

	fmt.Printf("\nBenchmark Query Generation For m = %d, l = %d, k = %d\n", row, col, k)
	fmt.Printf("\nCommunicatio Cost for %d MB database is %f KB\n", (row*col)>>20, float64(len(pirQuery.Vector_1)+len(pirQuery.Vector_2))/float64(1024))
	fmt.Printf("\tAverage Query Time with repeat times: %d: %v \n\n", b.N, totalDuration/time.Duration(b.N))
}

func BenchmarkAnswer(b *testing.B) {
	row, col, k, block := getParams()

	pi := &BasePIR{
		Params: BaseParams{
			Rows:           row,
			Cols:           col,
			NumberOfBlocks: block,
			CodewordLength: col + k,
			PackedSize:     row / 32,
		},
	}

	encodedMatrix := GenerateMatrix(pi.Params.CodewordLength, pi.Params.PackedSize, 32, 1)

	var totalDuration time.Duration
	b.ResetTimer()

	pirAnswer := &BasePIRAnswer{}
	for i := 0; i < b.N; i++ {
		vec_1 := utils.RandomizeUInt32Vector(pi.Params.CodewordLength / 32)
		vec_2 := utils.RandomizeUInt32Vector(pi.Params.CodewordLength / 32)
		start := time.Now()
		pirAnswer = pi.Answer(&encodedMatrix, &BasePIRQuery{Vector_1: vec_1, Vector_2: vec_2})
		totalDuration += time.Since(start)
	}

	b.StopTimer()

	fmt.Printf("\n Server Response For Encoded Database with %d x %d entires of size ~%fMB is %fMB. \n",
		encodedMatrix.Rows, encodedMatrix.Cols, float64(encodedMatrix.Rows/1024.0)*float64(encodedMatrix.Cols)/1024.0*32/8.0, float64(len(pirAnswer.Result_1)+len(pirAnswer.Result_2))/float64(1024*1024))
	fmt.Printf("\tAverage Answer Time with repeat times: %d: %v \n\n", b.N, totalDuration/time.Duration(b.N))
}

func BenchmarkDecode(b *testing.B) {
	row, col, k, block := getParams()

	pi := &BasePIR{
		Params: BaseParams{
			Rows:           row,
			Cols:           col,
			NumberOfBlocks: block,
			CodewordLength: col + k,
			PackedSize:     row / 32,
		},
	}

	sk := pi.KeyGen(1, 2, 32, 1)

	var totalDuration time.Duration
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		index := uint64(rand.Intn(int(row)))
		res_1 := utils.RandomPrimeFieldVector(pi.Params.PackedSize*pi.Params.NumberOfBlocks, 2^32)
		res_2 := utils.RandomPrimeFieldVector(pi.Params.PackedSize*pi.Params.NumberOfBlocks, 2^32)
		flip := utils.RandomizeFlipVector(pi.Params.NumberOfBlocks)
		mask := rand.Uint32()

		start := time.Now()
		pi.Decode(sk, index, &BasePIRAnswer{Result_1: res_1, Result_2: res_2},
			&BasePIRAux{FlipVector: flip, MaskValue: mask})
		totalDuration += time.Since(start)
	}

	b.StopTimer()

	fmt.Printf("\n Benchmark Decoding For Encoded Database with %d x %d entires of size ~%fMB with repeat times: %d. \n",
		col+k, row/32, float64((col+k)/1024.0)*float64(row/32)/1024.0*32/8.0, b.N)
	fmt.Printf("\tAverage Decoding Time with repeat times: %d: %v \n\n", b.N, totalDuration/time.Duration(b.N))
}

/* ================================================================================
				Test and Benchmarks for Mixed Split LSN based PIR in F4
   ================================================================================ */

func BenchmarkMixedSLSNAnswer(b *testing.B) {
	row, col, k, block := getParams()

	pi := &MixedSLSNPIR{
		Params: MixedSLSNParams{
			Rows:           row,
			Cols:           col,
			NumberOfBlocks: block,
			CodewordLength: col + k,
			PackedSize:     row / 32,
		},
	}

	encodedMatrix := GenerateMatrixF4(pi.Params.CodewordLength, pi.Params.PackedSize, 32, 1)

	var totalDuration time.Duration
	b.ResetTimer()

	// fmt.Println(encodedMatrix)
	for i := 0; i < b.N; i++ {
		vec_1 := utils.RandomizeBinaryVector(pi.Params.CodewordLength)
		vec_2 := utils.RandomizeBinaryVector(pi.Params.CodewordLength)
		vec_sum := make([]uint32, pi.Params.CodewordLength)
		for j := range vec_1 {
			vec_sum[j] = vec_1[j] ^ vec_2[j]
		}
		start := time.Now()
		pi.Answer(&encodedMatrix, &MixedSLSNPIRQuery{vec: VectorF4{Cols: col, Bit1: vec_1, BitP: vec_2, BitSum: vec_sum}})
		totalDuration += time.Since(start)
	}

	b.StopTimer()

	fmt.Printf("\nBenchmark Server Response For Encoded Database with %d x %d F4 entires of size ~%fMB. \n",
		encodedMatrix.Rows, encodedMatrix.Cols, 2*float64(encodedMatrix.Rows/1024.0)*float64(encodedMatrix.Cols)/1024.0*32/8.0)
	fmt.Printf("\tAverage Answer Time with repeat times: %d: %v \n\n", b.N, totalDuration/time.Duration(b.N))
}

func TestMixedSLSNAnswer(t *testing.T) {
	row, col, k, block := getParams()

	pi := &MixedSLSNPIR{
		Params: MixedSLSNParams{
			Rows:           row,
			Cols:           col,
			NumberOfBlocks: block,
			CodewordLength: col + k,
			PackedSize:     row / 32,
		},
	}

	encodedMatrix := GenerateMatrixF4(pi.Params.CodewordLength, pi.Params.PackedSize, 32, 1)

	var totalDuration time.Duration

	vec_1 := utils.RandomizeBinaryVectorWithSeed(pi.Params.CodewordLength, 1)
	vec_2 := utils.RandomizeBinaryVectorWithSeed(pi.Params.CodewordLength, 2)
	vec_sum := make([]uint32, pi.Params.CodewordLength)
	for j := range vec_1 {
		vec_sum[j] = vec_1[j] ^ vec_2[j]
	}
	//fmt.Println(vec_1)
	//fmt.Println(vec_2)
	start := time.Now()
	pi.Answer(&encodedMatrix, &MixedSLSNPIRQuery{vec: VectorF4{Cols: col, Bit1: vec_1, BitP: vec_2, BitSum: vec_sum}})
	//fmt.Println(ans)
	totalDuration += time.Since(start)

	fmt.Printf("\nBenchmark Server Response For Encoded Database with %d x %d entires of size ~%fMB. \n",
		encodedMatrix.Rows, encodedMatrix.Cols, float64(encodedMatrix.Rows/1024.0)*float64(encodedMatrix.Cols)/1024.0*32/8.0)
}

func getParams() (uint32, uint32, uint32, uint32) {
	col := 1 << 14 // will be rounded accordingly
	row := uint32(1 << 19)
	_, l, k, _, b := utils.Prms2(128, 1.25, col)
	return row, l, k, b
}

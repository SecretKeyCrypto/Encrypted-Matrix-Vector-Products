package mvp

import (
	"RandomLinearCodePIR/dataobjects"
	"RandomLinearCodePIR/ecc"
	"RandomLinearCodePIR/linearcode"
	"RandomLinearCodePIR/utils"
	"fmt"
	"math"
	"testing"
	"time"
)

// Test full flow correctness of Split-LSN MVP
func TestSlsnMVPComplete(t *testing.T) {
	m := uint32(1 << 10)
	l := uint32(1 << 10)
	k := uint32(1 << 4)
	s := uint32(2)
	n := k + l
	b := n / s
	p := uint32(65537)
	seed := int64(1)

	pi := &SlsnMVP{Params: SlsnParams{
		Field: dataobjects.NewPrimeField(p),
		S:     s,
		K:     k,
		N:     n,
		M:     m,
		L:     l,
		B:     b,
		P:     p,
	}}

	matrix := utils.GeneratePrimeFieldMatrix(pi.Params.M, pi.Params.L, p, seed)

	fmt.Printf("Running MVP with Database %d * %d \n", pi.Params.M, pi.Params.L)

	query := utils.RandomPrimeFieldVector(pi.Params.L, pi.Params.P)

	fmt.Println("Generate Key...")
	start := time.Now()
	sk := pi.KeyGen(seed)
	fmt.Println("    Elapsed: ", time.Since(start))

	fmt.Println("Generate Trapdoored Matrix...")
	start = time.Now()
	TDM := pi.GenerateTDM(sk)
	fmt.Println("    Elapsed: ", time.Since(start))

	fmt.Println("Encode Message...")
	start = time.Now()
	encodedMatrix := pi.Encode(sk, matrix, TDM)
	fmt.Println("    Elapsed: ", time.Since(start))

	fmt.Println("Generate Query...")
	start = time.Now()
	clientQuery, aux := pi.Query(sk, query)
	fmt.Println("    Elapsed: ", time.Since(start))
	fmt.Println("    Include Calculate Mask Time: ", aux.Dur)

	fmt.Println("Answer...")
	start = time.Now()
	serverResponse := pi.Answer(*encodedMatrix, *clientQuery)
	fmt.Println("    Elapsed: ", time.Since(start))

	fmt.Println("Decode...")
	start = time.Now()
	val := pi.Decode(sk, serverResponse, *aux)
	fmt.Println("    Elapsed: ", time.Since(start))

	target := dataobjects.AlignedMake[uint32](uint64(m))
	BlockMatVecProduct(matrix.Data, query, target, m, l, 1, p)

	for i := range target {
		if target[i] != val[i] {
			panic("Vec doesn't match ! ")
		}
	}

}

// Test full flow correctness of LPN based MVP
func TestLPNMVPComplete(t *testing.T) {
	m := uint32(1 << 10)
	l := uint32(1 << 10)
	k := uint32(1 << 4)
	n := k + l
	p := uint32(65537)
	seed := int64(1)
	m_1 := uint32(4)

	pi := &LpnMVP{Params: LpnParams{
		Field:     dataobjects.NewPrimeField(p),
		K:         k,
		N:         n,
		M:         m,
		L:         l,
		M_1:       m_1,
		ECCLength: 7,
		Epsi:      math.Pow(2, -40),
		P:         p,
		ECCName:   ecc.ReedSolomon,
	}}

	matrix := utils.GeneratePrimeFieldMatrix(pi.Params.M, pi.Params.L, p, seed)

	fmt.Printf("Running MVP with Database %d * %d \n", pi.Params.M, pi.Params.L)

	query := utils.RandomPrimeFieldVector(pi.Params.L, pi.Params.P)

	fmt.Println("Generate Key...")
	start := time.Now()
	sk := pi.KeyGen(seed)
	fmt.Println("    Elapsed: ", time.Since(start))

	fmt.Println("Generate Trapdoored Matrix...")
	start = time.Now()
	TDM := pi.GenerateTDM(sk)
	fmt.Println("    Elapsed: ", time.Since(start))

	fmt.Println("Encode Message...")
	start = time.Now()
	encodedMatrix := pi.Encode(sk, matrix, TDM)
	fmt.Println("    Elapsed: ", time.Since(start))

	fmt.Println("Generate Query...")
	start = time.Now()
	clientQuery, aux := pi.Query(sk, query)
	fmt.Println("    Elapsed: ", time.Since(start))

	fmt.Println("Answer...")
	start = time.Now()
	serverResponse := pi.Answer(encodedMatrix, clientQuery)
	fmt.Println("    Elapsed: ", time.Since(start))

	fmt.Println("Decode...")
	start = time.Now()
	val := pi.Decode(sk, serverResponse, aux)
	fmt.Println("    Elapsed: ", time.Since(start))

	target := dataobjects.AlignedMake[uint32](uint64(m))
	MatVecProduct(matrix.Data, query, target, m, l, p)

	for i := range target {
		if target[i] != val[i] {
			panic("Vec doesn't match ! ")
		}
	}
}

// Test full flow correctness of Ring variant of Split-LSN MVP
func TestRingSlsnMVPComplete(t *testing.T) {
	m := uint32(1 << 10)
	l := uint32(1 << 10)
	k := uint32(1 << 4)
	s := uint32(2)
	n := k + l
	b := n / s
	p := uint32(65537)
	seed := int64(1)

	pi := &SlsnMVP{Params: SlsnParams{
		Field: dataobjects.NewPrimeField(p),
		S:     s,
		K:     k,
		N:     n,
		M:     m,
		L:     l,
		B:     b,
		P:     p,
	}}

	code := linearcode.GetLinearCode(
		linearcode.LinearCodeConfig{
			Name:  linearcode.Vandermonde,
			K:     k,
			L:     l,
			Field: dataobjects.NewPrimeField(p),
		},
	)

	ring := &RingSlsnMVP{
		SlsnMVP:           *pi,
		LinearCodeEncoder: code,
	}

	matrix := utils.GeneratePrimeFieldMatrix(pi.Params.M, pi.Params.L, p, seed)
	query := utils.RandomPrimeFieldVector(pi.Params.L, pi.Params.P)

	target := dataobjects.AlignedMake[uint32](uint64(m))
	MatVecProduct(matrix.Data, query, target, m, l, p)

	fmt.Printf("Running MVP with Database %d * %d \n", pi.Params.M, pi.Params.L)

	fmt.Println("Generate Key...")
	start := time.Now()
	sk := ring.KeyGen(seed)
	fmt.Println("    Elapsed: ", time.Since(start))

	fmt.Println("Generate Trapdoored Matrix...")
	start = time.Now()
	TDM := ring.GenerateTDM(sk)
	fmt.Println("    Elapsed: ", time.Since(start))

	fmt.Println("Encode Message...")
	start = time.Now()
	encodedMatrix := ring.Encode(sk, matrix, TDM)
	fmt.Println("    Elapsed: ", time.Since(start))

	fmt.Println("Generate Query...")
	start = time.Now()
	clientQuery, aux := ring.Query(sk, query)
	fmt.Println("    Elapsed: ", time.Since(start))
	fmt.Println("    Include Calculate Mask Time: ", aux.Dur)

	fmt.Println("Answer...")
	start = time.Now()
	serverResponse := ring.Answer(*encodedMatrix, *clientQuery)
	fmt.Println("    Elapsed: ", time.Since(start))

	fmt.Println("Decode...")
	start = time.Now()
	val := ring.Decode(sk, serverResponse, *aux)
	fmt.Println("    Elapsed: ", time.Since(start))

	for i := range target {
		if target[i] != val[i] {
			panic("Vec doesn't match ! ")
		}
	}
}

// Benchmark cleartext server execution time for matrix-vector product
func BenchmarkCleartextServerExecution(b *testing.B) {
	printTestName("Benchmark ClearText")
	p := uint32(65537)
	_, m, l, _, _, _ := getParams()
	seed := int64(1)
	matrix := utils.GeneratePrimeFieldMatrix(m, l, p, seed)
	result := dataobjects.AlignedMake[uint32](uint64(m))

	var totalDuration time.Duration
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		query := utils.RandomPrimeFieldVector(l, p)
		start := time.Now()
		MatVecProduct(matrix.Data, query, result, m, l, p)
		duration := time.Since(start)
		totalDuration += duration
	}

	b.StopTimer()
	printBenchmarkExecutionTime(b.N)
	fmt.Printf("Average server execution time for m = %d, l = %d : %s\n", m, l, totalDuration/time.Duration(b.N))
}

// Benchmark query generation in Ring-based Split-LSN MVP
func BenchmarkRingSLSNQuery(b *testing.B) {
	printTestName("Benchmark Ring SLSN Query")
	n, m, l, k, s, block := getParams()

	p := uint32(65537)
	seed := int64(1)

	pi := &SlsnMVP{Params: SlsnParams{
		Field: dataobjects.NewPrimeField(p),
		S:     s,
		K:     k,
		N:     n,
		M:     m,
		L:     l,
		B:     block,
		P:     p,
	}}

	code := linearcode.GetLinearCode(
		linearcode.LinearCodeConfig{
			Name:  linearcode.Vandermonde,
			K:     k,
			L:     l,
			Field: dataobjects.NewPrimeField(p),
		},
	)

	ring := &RingSlsnMVP{
		SlsnMVP:           *pi,
		LinearCodeEncoder: code,
	}

	sk := ring.KeyGen(seed)

	var totalDuration time.Duration
	var unmaskDuration time.Duration

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		query := utils.RandomPrimeFieldVector(pi.Params.L, pi.Params.P)
		start := time.Now()
		_, aux := ring.Query(sk, query)
		duration := time.Since(start)
		totalDuration += duration
		unmaskDuration += aux.Dur
	}

	b.StopTimer()

	fmt.Printf("Ring SLSN For m = %d, l = %d, k = %d \n", m, l, k)
	printBenchmarkExecutionTime(b.N)
	fmt.Printf("Average Query time: %s\n", totalDuration/time.Duration(b.N))
	fmt.Printf("Average Calculate Mask time: %.s\n", unmaskDuration/time.Duration(b.N))
	fmt.Printf("Pure Query Generation Time: %s\n", (totalDuration-unmaskDuration)/time.Duration(b.N))
}

// Benchmark query generation in Split-LSN MVP
func BenchmarkSLSNQuery(b *testing.B) {
	printTestName("Benchmark SLSN Query")
	n, m, l, k, s, block := getParams()
	p := uint32(65537)
	seed := int64(1)

	pi := &SlsnMVP{Params: SlsnParams{
		Field: dataobjects.NewPrimeField(p),
		S:     s,
		K:     k,
		N:     n,
		M:     m,
		L:     l,
		B:     block,
		P:     p,
	}}

	sk := pi.KeyGen(seed)

	var totalDuration time.Duration
	var unmaskDuration time.Duration

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		query := utils.RandomPrimeFieldVector(pi.Params.L, pi.Params.P)
		start := time.Now()
		_, aux := pi.Query(sk, query)
		duration := time.Since(start)
		totalDuration += duration
		unmaskDuration += aux.Dur
	}

	b.StopTimer()

	fmt.Printf("SLSN For m = %d, l = %d, k = %d \n", m, l, k)
	printBenchmarkExecutionTime(b.N)
	fmt.Printf("Average Query time: %s\n", totalDuration/time.Duration(b.N))
	fmt.Printf("Average Calculate Mask time: %s\n", unmaskDuration/time.Duration(b.N))
	fmt.Printf("Pure Query Generation Time: %s\n", (totalDuration-unmaskDuration)/time.Duration(b.N))
}

// Benchmark Server Answer time in Split-LSN MVP
func BenchmarkSLSNAnswer(b *testing.B) {
	printTestName("Benchmark SLSN Answer")
	n, m, l, k, s, block := getParams()
	p := uint32(65537)
	seed := int64(1)

	pi := &SlsnMVP{Params: SlsnParams{
		Field: dataobjects.NewPrimeField(p),
		S:     s,
		K:     k,
		N:     n,
		M:     m,
		L:     l,
		B:     block,
		P:     p,
	}}

	encodedMatrix := utils.GeneratePrimeFieldMatrix(pi.Params.M, pi.Params.N, p, seed)

	var totalDuration time.Duration

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		clientQuery := utils.RandomPrimeFieldVector(pi.Params.N, pi.Params.P)

		start := time.Now()
		pi.Answer(encodedMatrix, SlsnQuery{Vec: clientQuery})
		totalDuration += time.Since(start)
	}

	b.StopTimer()

	fmt.Printf("Benchmark of SLSN Answer For m = %d, l = %d, k = %d \n", m, l, k)
	printBenchmarkExecutionTime(b.N)
	fmt.Printf("Average Answer time: %s\n", totalDuration/time.Duration(b.N))
}

// Benchmark Decode time in Split-LSN MVP
func BenchmarkSLSNDecode(b *testing.B) {
	printTestName("Benchmark SLSN Decode")
	n, m, l, k, s, block := getParams()
	p := uint32(65537)
	seed := int64(1)

	pi := &SlsnMVP{Params: SlsnParams{
		Field: dataobjects.NewPrimeField(p),
		S:     s,
		K:     k,
		N:     n,
		M:     m,
		L:     l,
		B:     block,
		P:     p,
	}}

	sk := pi.KeyGen(seed)
	var totalDuration time.Duration

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		response := utils.RandomPrimeFieldVector(pi.Params.M*pi.Params.S, pi.Params.P)
		coeff := utils.RandomSplitLSNNoiseCoeff(pi.Params.S, pi.Params.P)
		mask := utils.RandomPrimeFieldVector(pi.Params.M, pi.Params.P)

		start := time.Now()
		pi.Decode(sk, response, SlsnAux{Coeff: coeff, Masks: mask})
		totalDuration += time.Since(start)
	}

	b.StopTimer()

	fmt.Printf("Benchmark of SLSN Decode For m = %d, l = %d, k = %d \n", m, l, k)
	printBenchmarkExecutionTime(b.N)
	fmt.Printf("Average Decode time: %s\n", totalDuration/time.Duration(b.N))
}

func getParams() (uint32, uint32, uint32, uint32, uint32, uint32) {
	l := 10000
	m := uint32(1 << 10)
	n, ll, k, s, b := prms2(128, 1.25, l)
	return n, m, ll, k, s, b
}

func printTestName(name string) {
	fmt.Printf("\n\n =================== %s ===================\n", name)
}

func printBenchmarkExecutionTime(n int) {
	fmt.Printf("Benchmark Execution For *** %d *** Times \n", n)
}

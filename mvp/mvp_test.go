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

func TestBasic(t *testing.T) {
	ctx := dataobjects.MakeDeferralContextDefault()
	defer dataobjects.CloseDeferralContext(ctx)
	doctx := dataobjects.GetDeferralDoContext(ctx)
	frame := dataobjects.MakeDeferralFrame(ctx)
	defer frame.Close()

	fmt.Println("\n\nRunning TestBasic")

	val := dataobjects.DoAlignedMake[uint32](doctx, 1024)
	utils.RandomizeVectorWithModulusAndSeed(doctx, val, 1024, 1, false, false, 65537, 13, 17)
	target := dataobjects.DoAlignedMake[uint32](doctx, 1024)
	utils.RandomizeVectorWithModulusAndSeed(doctx, target, 1024, 1, false, false, 65537, 13, 17)

	dataobjects.CheckResult(dataobjects.DoAlignedSynchronize(doctx))
	if dataobjects.USE_FAST_CODE {
		if !utils.FieldVectorsAreEqual(target, 0, 0, val, 0, 0, uint32(len(target)), 1) {
			panic("Vec doesn't match ! ")
		}
	} else {
		for i := range target {
			if target[i] != val[i] {
				panic("Vec doesn't match ! ")
			}
		}
	}
}

// Test full flow correctness of Split-LSN MVP
func TestSlsnMVPComplete(t *testing.T) {
	ctx := dataobjects.MakeDeferralContextDefault()
	defer dataobjects.CloseDeferralContext(ctx)
	doctx := dataobjects.GetDeferralDoContext(ctx)
	frame := dataobjects.MakeDeferralFrame(ctx)
	defer frame.Close()

	m := uint32(1 << 10)
	l := uint32(1 << 10)
	k := uint32(1 << 4)
	s := uint32(2)
	n := k + l
	b := n / s
	p := uint32(65537)
	seed := int64(1)

	pi := &SlsnMVP{
		Ctx: ctx,
		Params: SlsnParams{
			Field: dataobjects.NewPrimeField(doctx, p),
			S:     s,
			K:     k,
			N:     n,
			M:     m,
			L:     l,
			B:     b,
			P:     p,
		},
	}

	matrix := utils.GeneratePrimeFieldMatrix(doctx, pi.Params.M, pi.Params.L, p, seed, 0)
	frame.Defer(func() { matrix.DoFree(doctx) })

	fmt.Printf("\n\nRunning SLSN Variant MVP with Database %d * %d \n", pi.Params.M, pi.Params.L)

	query := dataobjects.DoAlignedMake[uint32](doctx, uint64(pi.Params.L))
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, query) })
	utils.RandomPrimeFieldVector(doctx, query, pi.Params.P)

	fmt.Println("Generate Key...")
	start := time.Now()
	sk := pi.KeyGen(seed)
	fmt.Println("    Elapsed: ", time.Since(start))

	fmt.Println("Generate Trapdoored Matrix...")
	start = time.Now()
	TDM := pi.GenerateTDM(sk)
	fmt.Println("    Elapsed: ", time.Since(start))
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, TDM) })

	fmt.Println("Encode Message...")
	start = time.Now()
	encodedMatrix := pi.Encode(sk, matrix, TDM)
	fmt.Println("    Elapsed: ", time.Since(start))
	frame.Defer(func() { encodedMatrix.DoFree(doctx) })

	fmt.Println("Generate Query...")
	start = time.Now()
	clientQuery, aux := pi.Query(sk, query)
	fmt.Println("    Elapsed: ", time.Since(start))
	fmt.Println("    Include Calculate Mask Time: ", aux.Dur)
	frame.Defer(func() { clientQuery.DoFree(doctx) })
	frame.Defer(func() { aux.DoFree(doctx) })

	fmt.Println("Answer...")
	start = time.Now()
	serverResponse := pi.Answer(*encodedMatrix, *clientQuery)
	fmt.Println("    Elapsed: ", time.Since(start))
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, serverResponse) })

	fmt.Println("Decode...")
	start = time.Now()
	val := pi.Decode(sk, serverResponse, *aux)
	fmt.Println("    Elapsed: ", time.Since(start))
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, val) })

	target := dataobjects.DoAlignedMake[uint32](doctx, uint64(m))
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, target) })
	BlockMatVecProduct(doctx, matrix.Data, query, target, m, l, 1, p)

	dataobjects.CheckResult(dataobjects.DoAlignedSynchronize(doctx))
	if dataobjects.USE_FAST_CODE {
		if !utils.FieldVectorsAreEqual(target, 0, 0, val, 0, 0, uint32(len(target)), 1) {
			panic("Vec doesn't match ! ")
		}
	} else {
		for i := range target {
			if target[i] != val[i] {
				panic("Vec doesn't match ! ")
			}
		}
	}
}

// Test full flow correctness of LPN based MVP
func TestLPNMVPComplete(t *testing.T) {
	ctx := dataobjects.MakeDeferralContextDefault()
	defer dataobjects.CloseDeferralContext(ctx)
	doctx := dataobjects.GetDeferralDoContext(ctx)
	frame := dataobjects.MakeDeferralFrame(ctx)
	defer frame.Close()

	m := uint32(1 << 10)
	l := uint32(1 << 10)
	k := uint32(1 << 4)
	n := k + l
	p := uint32(65537)
	seed := int64(1)
	m_1 := uint32(4)

	pi := &LpnMVP{
		Ctx: ctx,
		Params: LpnParams{
			Field:     dataobjects.NewPrimeField(doctx, p),
			K:         k,
			N:         n,
			M:         m,
			L:         l,
			M_1:       m_1,
			ECCLength: 7,
			Epsi:      math.Pow(2, -40),
			P:         p,
			ECCName:   ecc.ReedSolomon,
		},
	}

	matrix := utils.GeneratePrimeFieldMatrix(doctx, pi.Params.M, pi.Params.L, p, seed, 0)
	frame.Defer(func() { matrix.DoFree(doctx) })

	fmt.Printf("\n\nRunning LPN Variant MVP with Database %d * %d \n", pi.Params.M, pi.Params.L)

	query := dataobjects.DoAlignedMake[uint32](doctx, uint64(pi.Params.L))
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, query) })
	utils.RandomPrimeFieldVector(doctx, query, pi.Params.P)

	fmt.Println("Generate Key...")
	start := time.Now()
	sk := pi.KeyGen(seed)
	fmt.Println("    Elapsed: ", time.Since(start))
	frame.Defer(func() { sk.DoFree(doctx) })

	fmt.Println("Generate Trapdoored Matrix...")
	start = time.Now()
	TDM := pi.GenerateTDM(sk)
	fmt.Println("    Elapsed: ", time.Since(start))
	frame.Defer(func() { dataobjects.Aligned2DFree(TDM) })

	fmt.Println("Encode Message...")
	start = time.Now()
	encodedMatrix := pi.Encode(sk, matrix, TDM)
	fmt.Println("    Elapsed: ", time.Since(start))
	frame.Defer(func() { encodedMatrix.DoFree(doctx) })

	fmt.Println("Generate Query...")
	start = time.Now()
	clientQuery, aux := pi.Query(sk, query)
	fmt.Println("    Elapsed: ", time.Since(start))
	frame.Defer(func() { clientQuery.DoFree(doctx) })
	frame.Defer(func() { aux.DoFree(doctx) })

	fmt.Println("Answer...")
	start = time.Now()
	serverResponse := pi.Answer(encodedMatrix, clientQuery)
	fmt.Println("    Elapsed: ", time.Since(start))
	frame.Defer(func() { serverResponse.DoFree(doctx) })

	fmt.Println("Decode...")
	start = time.Now()
	val, _ /* TODO possibleFailure */ := pi.Decode(sk, serverResponse, aux)
	fmt.Println("    Elapsed: ", time.Since(start))
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, val) })

	target := dataobjects.DoAlignedMake[uint32](doctx, uint64(m))
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, target) })
	MatVecProduct(doctx, matrix.Data, query, target, m, l, p)

	dataobjects.CheckResult(dataobjects.DoAlignedSynchronize(doctx))
	if dataobjects.USE_FAST_CODE {
		if !utils.FieldVectorsAreEqual(target, 0, 0, val, 0, 0, uint32(len(target)), 1) {
			panic("Vec doesn't match ! ")
		}
	} else {
		for i := range target {
			if target[i] != val[i] {
				panic("Vec doesn't match ! ")
			}
		}
	}
}

// Test full flow correctness of Ring variant of Split-LSN MVP
func TestRingSlsnMVPComplete(t *testing.T) {
	ctx := dataobjects.MakeDeferralContextDefault()
	defer dataobjects.CloseDeferralContext(ctx)
	doctx := dataobjects.GetDeferralDoContext(ctx)
	frame := dataobjects.MakeDeferralFrame(ctx)
	defer frame.Close()

	m := uint32(1 << 10)
	l := uint32(1 << 10)
	k := uint32(1 << 4)
	s := uint32(2)
	n := k + l
	b := n / s
	p := uint32(65537)
	seed := int64(1)

	pi := &SlsnMVP{
		Ctx: ctx,
		Params: SlsnParams{
			Field: dataobjects.NewPrimeField(doctx, p),
			S:     s,
			K:     k,
			N:     n,
			M:     m,
			L:     l,
			B:     b,
			P:     p,
		},
	}

	code := linearcode.GetLinearCode(
		ctx,
		linearcode.LinearCodeConfig{
			Name:  linearcode.Vandermonde,
			K:     k,
			L:     l,
			Field: dataobjects.NewPrimeField(doctx, p),
		},
	)

	ring := &RingSlsnMVP{
		SlsnMVP:           *pi,
		LinearCodeEncoder: code,
	}

	matrix := utils.GeneratePrimeFieldMatrix(doctx, pi.Params.M, pi.Params.L, p, seed, 0)
	frame.Defer(func() { matrix.DoFree(doctx) })
	query := dataobjects.DoAlignedMake[uint32](doctx, uint64(pi.Params.L))
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, query) })
	utils.RandomPrimeFieldVector(doctx, query, pi.Params.P)

	target := dataobjects.DoAlignedMake[uint32](doctx, uint64(m))
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, target) })
	MatVecProduct(doctx, matrix.Data, query, target, m, l, p)

	fmt.Printf("\n\nRunning Ring-SLSN Variant MVP with Database %d * %d \n", pi.Params.M, pi.Params.L)

	fmt.Println("Generate Key...")
	start := time.Now()
	sk := ring.KeyGen(seed)
	fmt.Println("    Elapsed: ", time.Since(start))
	frame.Defer(func() { sk.DoFree(doctx) })

	fmt.Println("Generate Trapdoored Matrix...")
	start = time.Now()
	TDM := ring.GenerateTDM(sk)
	fmt.Println("    Elapsed: ", time.Since(start))
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, TDM) })

	fmt.Println("Encode Message...")
	start = time.Now()
	encodedMatrix := ring.Encode(sk, matrix, TDM)
	fmt.Println("    Elapsed: ", time.Since(start))
	frame.Defer(func() { matrix.DoFree(doctx) })

	fmt.Println("Generate Query...")
	start = time.Now()
	clientQuery, aux := ring.Query(sk, query)
	fmt.Println("    Elapsed: ", time.Since(start))
	fmt.Println("    Include Calculate Mask Time: ", aux.Dur)
	frame.Defer(func() { clientQuery.DoFree(doctx) })
	frame.Defer(func() { aux.DoFree(doctx) })

	fmt.Println("Answer...")
	start = time.Now()
	serverResponse := ring.Answer(*encodedMatrix, *clientQuery)
	fmt.Println("    Elapsed: ", time.Since(start))
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, serverResponse) })

	fmt.Println("Decode...")
	start = time.Now()
	val := ring.Decode(sk, serverResponse, *aux)
	fmt.Println("    Elapsed: ", time.Since(start))
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, val) })

	dataobjects.CheckResult(dataobjects.DoAlignedSynchronize(doctx))
	if dataobjects.USE_FAST_CODE {
		if !utils.FieldVectorsAreEqual(target, 0, 0, val, 0, 0, uint32(len(target)), 1) {
			panic("Vec doesn't match ! ")
		}
	} else {
		for i := range target {
			if target[i] != val[i] {
				panic("Vec doesn't match ! ")
			}
		}
	}
}

// Benchmark cleartext server execution time for matrix-vector product
func BenchmarkCleartextServerExecution(b *testing.B) {
	printTestName("Benchmark ClearText")

	ctx := dataobjects.MakeDeferralContextDefault()
	defer dataobjects.CloseDeferralContext(ctx)
	doctx := dataobjects.GetDeferralDoContext(ctx)
	frame := dataobjects.MakeDeferralFrame(ctx)
	defer frame.Close()

	p := uint32(65537)
	_, m, l, _, _, _ := getParams()
	seed := int64(1)
	matrix := utils.GeneratePrimeFieldMatrix(doctx, m, l, p, seed, 0)
	frame.Defer(func() { matrix.DoFree(doctx) })
	result := dataobjects.DoAlignedMake[uint32](doctx, uint64(m))
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, result) })

	var totalDuration time.Duration
	b.ResetTimer()

	query := dataobjects.DoAlignedMake[uint32](doctx, uint64(l))
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, query) })
	utils.RandomPrimeFieldVector(doctx, query, p)
	for i := 0; i < b.N; i++ {
		start := time.Now()
		MatVecProduct(doctx, matrix.Data, query, result, m, l, p)
		duration := time.Since(start)
		totalDuration += duration
	}

	dataobjects.CheckResult(dataobjects.DoAlignedSynchronize(doctx))
	b.StopTimer()
	fmt.Printf("Benchmark Cleartext MVP for %d x %d DB of size ~%.2f MB\n", m, l, float64(m*l*4)/float64(1024*1024))
	printBenchmarkExecutionTime(b.N)
	fmt.Printf("Average server execution time for m = %d, l = %d : %s\n", m, l, totalDuration/time.Duration(b.N))
}

func BenchmarkRingSLSNEncoding(b *testing.B) {
	printTestName("Benchmark Ring SLSN Encoding")

	ctx := dataobjects.MakeDeferralContextDefault()
	defer dataobjects.CloseDeferralContext(ctx)
	doctx := dataobjects.GetDeferralDoContext(ctx)
	frame := dataobjects.MakeDeferralFrame(ctx)
	defer frame.Close()

	n, m, l, k, s, block := getParams()

	p := uint32(65537)
	seed := int64(1)

	pi := &SlsnMVP{
		Ctx: ctx,
		Params: SlsnParams{
			Field: dataobjects.NewPrimeField(doctx, p),
			S:     s,
			K:     k,
			N:     n,
			M:     m,
			L:     l,
			B:     block,
			P:     p,
		},
	}

	code := linearcode.GetLinearCode(
		ctx,
		linearcode.LinearCodeConfig{
			Name:  linearcode.Vandermonde,
			K:     k,
			L:     l,
			Field: dataobjects.NewPrimeField(doctx, p),
		},
	)

	ring := &RingSlsnMVP{
		SlsnMVP:           *pi,
		LinearCodeEncoder: code,
	}
	sk := ring.KeyGen(seed)
	frame.Defer(func() { sk.DoFree(doctx) })
	matrix := utils.GeneratePrimeFieldMatrix(doctx, pi.Params.M, pi.Params.L, p, seed, 0)
	frame.Defer(func() { matrix.DoFree(doctx) })
	TDM := dataobjects.DoAlignedMake[uint32](doctx, uint64(pi.Params.N*pi.Params.L))
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, TDM) })
	utils.RandomPrimeFieldVector(doctx, TDM, p)

	var totalDuration time.Duration

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		start := time.Now()
		matrix := ring.Encode(sk, matrix, TDM)
		totalDuration += time.Since(start)
		frame.Defer(func() { matrix.DoFree(doctx) })
	}

	dataobjects.CheckResult(dataobjects.DoAlignedSynchronize(doctx))
	b.StopTimer()

	fmt.Printf("\nBenchmark of Ring SLSN Encoding For %d x %d DB(~%.2f MB), encoded to %d x %d with block size %d \n",
		m, l, float64(m*l*4)/float64(1024*1024), m, n, block)
	printBenchmarkExecutionTime(b.N)
	fmt.Printf("Average Encoding time: %s\n", totalDuration/time.Duration(b.N))
}

// Benchmark query generation in Ring-based Split-LSN MVP
func BenchmarkRingSLSNQuery(b *testing.B) {
	printTestName("Benchmark Ring SLSN Query")

	ctx := dataobjects.MakeDeferralContextDefault()
	defer dataobjects.CloseDeferralContext(ctx)
	doctx := dataobjects.GetDeferralDoContext(ctx)
	frame := dataobjects.MakeDeferralFrame(ctx)
	defer frame.Close()

	n, m, l, k, s, block := getParams()

	p := uint32(65537)
	seed := int64(1)

	pi := &SlsnMVP{
		Ctx: ctx,
		Params: SlsnParams{
			Field: dataobjects.NewPrimeField(doctx, p),
			S:     s,
			K:     k,
			N:     n,
			M:     m,
			L:     l,
			B:     block,
			P:     p,
		},
	}

	code := linearcode.GetLinearCode(
		ctx,
		linearcode.LinearCodeConfig{
			Name:  linearcode.Vandermonde,
			K:     k,
			L:     l,
			Field: dataobjects.NewPrimeField(doctx, p),
		},
	)

	ring := &RingSlsnMVP{
		SlsnMVP:           *pi,
		LinearCodeEncoder: code,
	}

	sk := ring.KeyGen(seed)
	frame.Defer(func() { sk.DoFree(doctx) })

	var totalDuration time.Duration
	var unmaskDuration time.Duration

	b.ResetTimer()

	query := dataobjects.DoAlignedMake[uint32](doctx, uint64(pi.Params.L))
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, query) })
	utils.RandomPrimeFieldVector(doctx, query, pi.Params.P)
	for i := 0; i < b.N; i++ {
		start := time.Now()
		q, aux := ring.Query(sk, query)
		duration := time.Since(start)
		frame.Defer(func() { q.DoFree(doctx) })
		frame.Defer(func() { aux.DoFree(doctx) })
		totalDuration += duration
		unmaskDuration += aux.Dur
	}

	dataobjects.CheckResult(dataobjects.DoAlignedSynchronize(doctx))
	b.StopTimer()

	fmt.Printf("Ring SLSN For m = %d, l = %d, k = %d \n", m, l, k)
	printBenchmarkExecutionTime(b.N)
	fmt.Printf("Average Query time: %s\n", totalDuration/time.Duration(b.N))
	fmt.Printf("Average Calculate Mask time: %s\n", unmaskDuration/time.Duration(b.N))
	fmt.Printf("Pure Query Generation Time: %s\n", (totalDuration-unmaskDuration)/time.Duration(b.N))
}

func BenchmarkSLSNGenerateTDM(b *testing.B) {
	printTestName("Benchmark SLSN TDM Generation")

	ctx := dataobjects.MakeDeferralContextDefault()
	defer dataobjects.CloseDeferralContext(ctx)
	doctx := dataobjects.GetDeferralDoContext(ctx)
	frame := dataobjects.MakeDeferralFrame(ctx)
	defer frame.Close()

	n, m, l, k, s, block := getParams()

	p := uint32(65537)
	seed := int64(1)

	pi := &SlsnMVP{
		Ctx: ctx,
		Params: SlsnParams{
			Field: dataobjects.NewPrimeField(doctx, p),
			S:     s,
			K:     k,
			N:     n,
			M:     m,
			L:     l,
			B:     block,
			P:     p,
		},
	}

	sk := pi.KeyGen(seed)
	frame.Defer(func() { sk.DoFree(doctx) })

	var totalDuration time.Duration

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		start := time.Now()
		TDM := pi.GenerateTDM(sk)
		totalDuration += time.Since(start)
		frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, TDM) })
	}

	dataobjects.CheckResult(dataobjects.DoAlignedSynchronize(doctx))
	b.StopTimer()

	fmt.Printf("\nBenchmark of SLSN Generate TDM For %d x %d DB(~%.2f MB), encoded to %d x %d with block size %d \n",
		m, l, float64(m*l*4)/float64(1024*1024), m, n, block)
	printBenchmarkExecutionTime(b.N)
	fmt.Printf("Average Encoding time: %s\n", totalDuration/time.Duration(b.N))
}

func BenchmarkSLSNEncoding(b *testing.B) {
	printTestName("Benchmark SLSN Encoding")

	ctx := dataobjects.MakeDeferralContextDefault()
	defer dataobjects.CloseDeferralContext(ctx)
	doctx := dataobjects.GetDeferralDoContext(ctx)
	frame := dataobjects.MakeDeferralFrame(ctx)
	defer frame.Close()

	n, m, l, k, s, block := getParams()

	p := uint32(65537)
	seed := int64(1)

	pi := &SlsnMVP{
		Ctx: ctx,
		Params: SlsnParams{
			Field: dataobjects.NewPrimeField(doctx, p),
			S:     s,
			K:     k,
			N:     n,
			M:     m,
			L:     l,
			B:     block,
			P:     p,
		},
	}

	sk := pi.KeyGen(seed)
	frame.Defer(func() { sk.DoFree(doctx) })
	matrix := utils.GeneratePrimeFieldMatrix(doctx, pi.Params.M, pi.Params.L, p, seed, 0)
	frame.Defer(func() { matrix.DoFree(doctx) })
	TDM := dataobjects.DoAlignedMake[uint32](doctx, uint64(pi.Params.M))
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, TDM) })
	utils.RandomPrimeFieldVector(doctx, TDM, pi.Params.L)

	var totalDuration time.Duration

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		start := time.Now()
		m := pi.Encode(sk, matrix, TDM)
		totalDuration += time.Since(start)
		frame.Defer(func() { m.DoFree(doctx) })
	}

	dataobjects.CheckResult(dataobjects.DoAlignedSynchronize(doctx))
	b.StopTimer()

	fmt.Printf("\nBenchmark of SLSN Encoding For %d x %d DB(~%.2f MB), encoded to %d x %d with block size %d \n",
		m, l, float64(m*l*4)/float64(1024*1024), m, n, block)
	printBenchmarkExecutionTime(b.N)
	fmt.Printf("Average Encoding time: %s\n", totalDuration/time.Duration(b.N))
}

// Benchmark query generation in Split-LSN MVP
func BenchmarkSLSNQuery(b *testing.B) {
	printTestName("Benchmark SLSN Query")

	ctx := dataobjects.MakeDeferralContextDefault()
	defer dataobjects.CloseDeferralContext(ctx)
	doctx := dataobjects.GetDeferralDoContext(ctx)
	frame := dataobjects.MakeDeferralFrame(ctx)
	defer frame.Close()

	n, m, l, k, s, block := getParams()
	p := uint32(65537)
	seed := int64(1)

	pi := &SlsnMVP{
		Ctx: ctx,
		Params: SlsnParams{
			Field: dataobjects.NewPrimeField(doctx, p),
			S:     s,
			K:     k,
			N:     n,
			M:     m,
			L:     l,
			B:     block,
			P:     p,
		},
	}

	sk := pi.KeyGen(seed)
	frame.Defer(func() { sk.DoFree(doctx) })

	var totalDuration time.Duration
	var unmaskDuration time.Duration

	b.ResetTimer()

	query := dataobjects.DoAlignedMake[uint32](doctx, uint64(pi.Params.L))
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, query) })
	utils.RandomPrimeFieldVector(doctx, query, pi.Params.P)
	for i := 0; i < b.N; i++ {
		start := time.Now()
		q, aux := pi.Query(sk, query)
		duration := time.Since(start)
		frame.Defer(func() { q.DoFree(doctx) })
		frame.Defer(func() { aux.DoFree(doctx) })
		totalDuration += duration
		unmaskDuration += aux.Dur
	}

	dataobjects.CheckResult(dataobjects.DoAlignedSynchronize(doctx))
	b.StopTimer()

	fmt.Printf("Benchmark of SLSN Query For %d x %d DB(~%.2f MB), encoded to %d x %d with block size %d \n",
		m, l, float64(m*l*4)/float64(1024*1024), m, n, block)
	printBenchmarkExecutionTime(b.N)
	fmt.Printf("Average Query time: %s\n", totalDuration/time.Duration(b.N))
	fmt.Printf("Average Calculate Mask time: %s\n", unmaskDuration/time.Duration(b.N))
	fmt.Printf("Pure Query Generation Time: %s\n", (totalDuration-unmaskDuration)/time.Duration(b.N))
}

// Benchmark Server Answer time in Split-LSN MVP
func BenchmarkSLSNAnswer(b *testing.B) {
	printTestName("Benchmark SLSN Answer")

	ctx := dataobjects.MakeDeferralContextDefault()
	defer dataobjects.CloseDeferralContext(ctx)
	doctx := dataobjects.GetDeferralDoContext(ctx)
	frame := dataobjects.MakeDeferralFrame(ctx)
	defer frame.Close()

	n, m, l, k, s, block := getParams()
	p := uint32(65537)
	seed := int64(1)

	pi := &SlsnMVP{
		Ctx: ctx,
		Params: SlsnParams{
			Field: dataobjects.NewPrimeField(doctx, p),
			S:     s,
			K:     k,
			N:     n,
			M:     m,
			L:     l,
			B:     block,
			P:     p,
		},
	}

	encodedMatrix := utils.GeneratePrimeFieldMatrix(doctx, pi.Params.M, pi.Params.N, p, seed, 0)
	frame.Defer(func() { encodedMatrix.DoFree(doctx) })

	var totalDuration time.Duration

	b.ResetTimer()

	clientQuery := dataobjects.DoAlignedMake[uint32](doctx, uint64(pi.Params.L))
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, clientQuery) })
	utils.RandomPrimeFieldVector(doctx, clientQuery, pi.Params.P)
	for i := 0; i < b.N; i++ {
		start := time.Now()
		answer := pi.Answer(encodedMatrix, SlsnQuery{Vec: clientQuery})
		totalDuration += time.Since(start)
		frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, answer) })

	}

	dataobjects.CheckResult(dataobjects.DoAlignedSynchronize(doctx))
	b.StopTimer()

	fmt.Printf("Benchmark of SLSN Answer For %d x %d DB(~%.2f MB), encoded to %d x %d with block size %d \n",
		m, l, float64(m*l*4)/float64(1024*1024), m, n, block)
	printBenchmarkExecutionTime(b.N)
	fmt.Printf("Average Answer time: %s\n", totalDuration/time.Duration(b.N))
}

// Benchmark Decode time in Split-LSN MVP
func BenchmarkSLSNDecode(b *testing.B) {
	printTestName("Benchmark SLSN Decode")

	ctx := dataobjects.MakeDeferralContextDefault()
	defer dataobjects.CloseDeferralContext(ctx)
	doctx := dataobjects.GetDeferralDoContext(ctx)
	frame := dataobjects.MakeDeferralFrame(ctx)
	defer frame.Close()

	n, m, l, k, s, block := getParams()
	p := uint32(65537)
	seed := int64(1)

	pi := &SlsnMVP{
		Ctx: ctx,
		Params: SlsnParams{
			Field: dataobjects.NewPrimeField(doctx, p),
			S:     s,
			K:     k,
			N:     n,
			M:     m,
			L:     l,
			B:     block,
			P:     p,
		},
	}

	sk := pi.KeyGen(seed)
	frame.Defer(func() { sk.DoFree(doctx) })
	var totalDuration time.Duration

	b.ResetTimer()

	response := dataobjects.DoAlignedMake[uint32](doctx, uint64(pi.Params.M*pi.Params.S))
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, response) })
	mask := dataobjects.DoAlignedMake[uint32](doctx, uint64(pi.Params.M))
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, mask) })
	coeff := dataobjects.DoAlignedMake[uint32](doctx, uint64(pi.Params.S))
	frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, coeff) })
	for i := 0; i < b.N; i++ {
		utils.RandomPrimeFieldVector(doctx, response, pi.Params.P)
		utils.RandomSplitLSNNoiseCoeff(doctx, coeff, pi.Params.P)
		utils.RandomPrimeFieldVector(doctx, mask, pi.Params.P)

		start := time.Now()
		decoded := pi.Decode(sk, response, SlsnAux{Coeff: coeff, Masks: mask})
		totalDuration += time.Since(start)
		frame.Defer(func() { dataobjects.DoAligned1DFree(doctx, decoded) })
	}

	dataobjects.CheckResult(dataobjects.DoAlignedSynchronize(doctx))
	b.StopTimer()

	fmt.Printf("Benchmark of SLSN Decoding For %d x %d DB(~%.2f MB), encoded to %d x %d with block size %d \n",
		m, l, float64(m*l*4)/float64(1024*1024), m, n, block)
	printBenchmarkExecutionTime(b.N)
	fmt.Printf("Average Decoding time: %s\n", totalDuration/time.Duration(b.N))
}

func getParams() (uint32, uint32, uint32, uint32, uint32, uint32) {
	l := 1 << 13
	m := uint32(1<<26) / uint32(l)

	ll, k, s, b := utils.Prms(128, 1.25, l)
	return ll + k, m, ll, k, s, b
}

func printTestName(name string) {
	fmt.Printf("\n\n =================== %s ===================\n", name)
}

func printBenchmarkExecutionTime(n int) {
	fmt.Printf("Benchmark Execution For *** %d *** Times \n", n)
}

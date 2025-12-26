// Test algebraic attacks against EMVP with 1D-SLSN

#include "prvector.h"
#include "constants.h"
#include "utils.h"
#include <iostream>
#include <vector>
#include <NTL/lzz_p.h>
#include <NTL/lzz_pE.h>
#include <NTL/mat_lzz_p.h>

using namespace emvpexpr;
using namespace NTL;

namespace emvpexpr {
constexpr int kBlocksPerRank = 2;    // ratio between code rank and block size (only used locally)
// amazonq-ignore-next-line
long kMod = kDefaultMod;             // modulus - can be changed via command-line argument
int kBlockSize = kDefaultBlockSize;  // block size - can be changed via command-line argument
}

// Test the rank of a matrix whose rows are degree-d multilinear products
// of the first d blocks in the given vectors, looking at only d entries in
// the last block (out of kBlockSize).
// Returns true if we have rank deficiency (i.e. rank < blockSize^d), or on
// bad parameters (so the caller stop trying).
bool testMultilinear(int d, std::vector<NTL::vec_zz_p>& vectors, int codeRank) {
    int lastBlock = kBlockSize;
    if (kBlockSize*d > codeRank) { // last block could be shorter
        lastBlock = codeRank + d - kBlockSize*(d-1);
    }
    if (lastBlock > kBlockSize) {
        std::cout << "Bad arguments, d="<<d<<", b="<<kBlockSize
            << ", k=" << codeRank << std::endl;
        return true;
    }

    long N = power_long(kBlockSize, d-1) * lastBlock;
    if (vectors.size() < size_t(N)) {
        std::cout << "Not enough samples ("<< vectors.size()
            << "), expecting at least " << kBlockSize << '^' << d
            << '=' << N << std::endl;
        return true;
    }
    int expectedSampleSize = (d-1)*kBlockSize +lastBlock;
    if (vectors[0].length() < expectedSampleSize) {
        std::cout << "Sample dimension too low ("<< vectors[0].length()
            << "), expecting at least " << expectedSampleSize << std::endl;
        return true;
    }
    std::cout << "\nTesting " << d <<"-multilinear " << N << "x" << N 
        << " matrix (b="<< kBlockSize << ", lastB=" << lastBlock << ")\n";
    mat_zz_p mat1(INIT_SIZE, N, N);
    std::vector<NTL::zz_p> tensor;
    for (long i = 0; i < N; i++) {
        for (int block = 0; block < d; block++) {
            // Extract the j'th block (last block could be shorter)
            int blockSize = ((block == (d-1))? lastBlock : kBlockSize);
            std::vector<NTL::zz_p> blockVec(blockSize);
            for (int j = 0; j < blockSize; j++) {
                blockVec[j] = vectors[i][block * kBlockSize + j];
            }

            // Multiply the j'th block into the tensor
            if (block == 0) {
                tensor = blockVec;
            } else {
                tensor = tensorProduct(tensor, blockVec);
            }
        }
        // Sanity check
        if (tensor.size() != N) {
            std::cout << "tensor.size()=="<<tensor.size()<<" != N\n";
            return true;
        }
        // Copy the tensor into the i'th row of the matrix
        for (int j = 0; j < N; j++) {
            mat1[i][j] = tensor[j];
        }
    }
    long rank = gauss(mat1);
    std::cout << "Rank of "<<d<<"-block-multilinear matrix is " << rank << std::endl;

    return (rank < N);
}

// Test the degree of a matrix whose rows are all the polynomials of
// degree <= d in the entries of the given vectors
bool testDegreeD(int d, std::vector<NTL::vec_zz_p>& vectors) {
    if (vectors.empty()) {
        std::cout << "testDegreeD: empty sample, returning\n";
        return true;
    }
    vec_zz_p vec = vectors[0];
    const int n = vec.length();
    vec.append(NTL::zz_p(1L));  // Append 1 to get all degrees <= d
    auto tensor = orderedTensor(vec, d);
    long N = tensor.size();
    if (N > 8192) {
        std::cout << std::endl << N << "x" << N
            << " is too large for rank calculations" << std::endl;
        return true;
    }
    if (vectors.size() < size_t(N)) {
        std::cout << "Not enough samples for this matrix\n";
        return true;
    }
    std::cout << "\nTesting degree-" << d <<", "<< N << "x" << N << " matrix\n";
    mat_zz_p mat(INIT_SIZE, N, N);
    long i = 0;
    while (true) {
        // Copy the tensor to the i'th row of the matrix
        for (long j = 0; j < N; j++) {
            mat[i][j] = tensor[j];
        }
        if (++i == N) {  // Done
            break;
        }
        // Prepare the next vector
        if (vectors[i].length() != n) {
            std::cout << "Error: vectors["<<i<<"].length()="<<vectors[i].length()
                <<" != n="<< n << std::endl;
            return true;
        }
        vec = vectors[i];
        vec.append(NTL::zz_p(1L));  // Append 1 to get all degrees <= d
        // amazonq-ignore-next-line
        tensor = orderedTensor(vec, d);
    }
    auto rank = gauss(mat);
    auto expected = std::min(mat.NumRows(), mat.NumCols());
    std::cout << "Rank of degree-"<<d<<" "<<mat.NumRows()<<"x"<<mat.NumCols()<<" matrix is " << rank << std::endl;
    return (rank < expected);
}

// Analyze degree-d algebraic attacks for a given code
void analyzeAlgebraicAttack(PRvector& prv, int d) {
    if (d < 1 || d > 3) {
        std::cout << "d="<<d<<" is too small or too big\n";
        return;
    }
    int n = prv.getN();
    // amazonq-ignore-next-line
    // amazonq-ignore-next-line
    int Nsamples = NTL::power_long(n+1, d);
    std::vector<NTL::vec_zz_p> vectors;
    for (long i = 0; i < Nsamples; i++) {
        vectors.push_back(prv.generateVector());
    }

    int k = prv.getK();
    if (d==1) { // For degree 1, test only multilinear
        testMultilinear(1, vectors, k);
    } else {
        if (testDegreeD(d, vectors)) { // found degree-d annialating polynomial
            testMultilinear(d, vectors, k);  // check for such multilinear poly
        }
    }
}

#if 0
// Sanity-check function, in case we need to compare the 1D-SLSN samples to
// "truely random" samples
void analyzeRandomSamples(int n) {
    if (n < 2) {
        return; // nothing to analyze
    }
    int d = 3;
    int N = 1000;
    std::vector<NTL::vec_zz_p> vectors;
    for (long i = 0; i < N; i++) {
        NTL::vec_zz_p v;
        NTL::random(v, n);
        vectors.push_back(v);
    }
    for (int i = 1; i <= d; i++)
        testDegreeD(i, vectors);
}
#endif

int main(int argc, char* argv[]) {
    // amazonq-ignore-next-line
    // amazonq-ignore-next-line
    // amazonq-ignore-next-line
    if (argc > 1) kMod = std::atol(argv[1]);
    if (argc > 2) kBlockSize = std::atoi(argv[2]);

    std::cout << "Testing algebraic attacks against EMVP with 1D-SLSN\n";
    std::cout << "Usage: " << argv[0] << " [kMod [kBlockSize]]" << std::endl;
    // amazonq-ignore-next-line
    std::cout << "  kMod = " << kMod << " (default: " << kDefaultMod << ")" << std::endl;
    // amazonq-ignore-next-line
    std::cout << "  kBlockSize = " << kBlockSize << " (default: " << kDefaultBlockSize << ")\n" << std::endl;
    
    NTL::zz_p::init(kMod);
    int k = kBlockSize*kBlocksPerRank;
    int n = k*2;

    std::cout << "Two types of codes:\n";
    // amazonq-ignore-next-line
    std::cout << " - full-rank quasi-cyclic "<<k<<"x"<<n<<" code\n";
    std::cout << " - first d blocks (d<=2) have rank deficiency of d" << std::endl;
#if 0
    /* Sanity checks*/
    std::cout << "\n********* Random samples: *********";
    analyzeRandomSamples(n);

    std::cout << "\n\n********* Random full-rank code: *********\n";
    {PRvector prv(k, n, 0);
    analyzeAlgebraicAttack(prv, kBlocksPerRank+1);
    }
#endif

    std::cout << "\n*********** Full rank quasi-circular code: *********\n";
    {PRvector prv = quasiCircular(k);
    auto& C = prv.getMat();
    // amazonq-ignore-next-line
    std::cout << "Rank of C = " << rankOfSubmat(C, kBlocksPerRank*2) << std::endl;
    analyzeAlgebraicAttack(prv, kBlocksPerRank);
    analyzeAlgebraicAttack(prv, kBlocksPerRank+1);
    }

    std::cout << "\n\n*********** "<<1<<"-rank-deficient code: *********\n";
    {PRvector prv(k, n, 1, kBlockSize);
    auto& C = prv.getMat();
    // amazonq-ignore-next-line
    std::cout << "Rank of C[1st block] = " << rankOfSubmat(C, 1) << std::endl;
    analyzeAlgebraicAttack(prv, 1);
    }

    std::cout << "\n\n*********** "<<2<<"-rank-deficient code: *********\n";
    PRvector prv(k, n, 2, kBlockSize);
    auto& C = prv.getMat();
    // amazonq-ignore-next-line
    std::cout << "Rank of C[1st block] = " << rankOfSubmat(C, 1) << std::endl;
    // amazonq-ignore-next-line
    std::cout << "Rank of C[1st 2 blocks] = " << rankOfSubmat(C, 2) << std::endl;
    analyzeAlgebraicAttack(prv, 1);
    analyzeAlgebraicAttack(prv, 2);

    return 0;
}

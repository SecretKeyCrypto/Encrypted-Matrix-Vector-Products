// Test algebraic attacks against EMVP with 1D-SLSN over small fields

#include "prvector.h"
#include "constants.h"
#include "utils.h"
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <NTL/lzz_p.h>
#include <NTL/lzz_pE.h>
#include <NTL/mat_lzz_p.h>

using namespace emvpexpr;
using namespace NTL;

namespace emvpexpr {
int kBlocksPerRank = 3;
int kBlocksPerLength = 0;            // default: kBlocksPerRank+1, must be <= 2*kBlocksPerRank
long kMod = 3;
int kBlockSize = kDefaultBlockSize;
}

#define QUASI_CYCLIC 0   // 1 - quasi-cyclic, 0 - random

// Truncate a code matrix from k-by-n' to k-by-n by dropping the last n'-n columns,
// and return a new PRvector with the truncated matrix.
#if QUASI_CYCLIC==1
PRvector truncateCode(PRvector& prv, int n) {
    auto& M = prv.getMat();
    int k = M.NumRows();
    int nOrig = M.NumCols();
    if (n >= nOrig) return prv;  // nothing to truncate

    NTL::mat_zz_p truncated(NTL::INIT_SIZE, k, n);
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            truncated[i][j] = M(i, j);
        }
    }
    return PRvector(Matrix(truncated));
}
#endif

// Test the degree of a matrix whose rows are all the polynomials of
// degree <= d in the entries of the given vectors
bool testDegreeD(int d, std::vector<NTL::vec_zz_p>& vectors) {
    if (vectors.empty()) {
        throw std::invalid_argument("testDegreeD: empty sample");
    }
    vec_zz_p vec = vectors[0];
    const int n = vec.length();
    vec.append(NTL::zz_p(1L));  // Append 1 to get all degrees <= d
    auto tensor = orderedTensor(vec, d);
    long N = tensor.size();
    if (N > 4096) {
        throw std::runtime_error("Matrix " + std::to_string(N) + "x" + std::to_string(N)
            + " is too large for rank calculations");
    }
    if (vectors.size() < size_t(N) + 30) {
        throw std::runtime_error("Not enough samples (" + std::to_string(vectors.size())
            + ") for " + std::to_string(N) + "x" + std::to_string(N) + " matrix");
    }

    //    std::cout << "\nTesting degree-" << d <<", "<< N << "x" << N << " matrix\n";
    mat_zz_p mat(INIT_SIZE, vectors.size(), N);
    long i = 0;
    while (true) {
        // Copy the tensor to the i'th row of the matrix
        for (long j = 0; j < N; j++) {
            mat[i][j] = tensor[j];
        }
        if (++i == vectors.size()) {  // Done
            break;
        }
        // Prepare the next vector
        if (vectors[i].length() != n) {
            throw std::runtime_error("vectors[" + std::to_string(i) + "].length()="
                + std::to_string(vectors[i].length()) + " != n=" + std::to_string(n));
        }
        vec = vectors[i];
        vec.append(NTL::zz_p(1L));  // Append 1 to get all degrees <= d
        tensor = orderedTensor(vec, d);
    }
    auto rank = gauss(mat);
    auto expected = std::min(mat.NumRows(), mat.NumCols());
    //    std::cout << "Rank of degree-"<<d<<" "<<mat.NumRows()<<"x"<<mat.NumCols()<<" matrix is " << rank << std::endl;
    return (rank < expected);
}

// Analyze degree-2 algebraic attacks for a given code.
// Returns true if a degree-2 annihilating polynomial was found.
bool analyzeAlgebraicAttack(PRvector& prv) {
    constexpr int d = 2;
    int n = prv.getN();
    int Nsamples = NTL::power_long(n+1, d);
    std::vector<NTL::vec_zz_p> vectors;
    for (long i = 0; i < Nsamples; i++) {
        vectors.push_back(prv.generateVector());
    }

    return testDegreeD(d, vectors);
}

int main(int argc, char* argv[]) {
    if (argc > 1) kBlockSize = std::atoi(argv[1]);
    if (argc > 2) kBlocksPerRank = std::atoi(argv[2]);
    if (argc > 3) kBlocksPerLength = std::atoi(argv[3]);
    if (kBlocksPerLength <= 0) kBlocksPerLength = kBlocksPerRank+1;
    if (kBlocksPerLength <= kBlocksPerRank || kBlocksPerLength > 2*kBlocksPerRank) {
        std::cout << "Error: kBlocksPerLength must be > kBlocksPerRank and <= 2*kBlocksPerRank (n/k <= 2)\n";
        return 1;
    }

    std::cout << "Testing algebraic attacks against EMVP with 1D-SLSN over F3\n";
    std::cout << "Usage: " << argv[0] << " [kBlockSize [kBlocksPerRank [kBlocksPerLength]]]" << std::endl;
    std::cout << "  kBlockSize = " << kBlockSize << " (default: " << kDefaultBlockSize << ")" << std::endl;
    std::cout << "  kBlocksPerRank = " << kBlocksPerRank << " (default: 3)" << std::endl;
    std::cout << "  kBlocksPerLength = " << kBlocksPerLength << " (default: kBlocksPerRank+1)\n" << std::endl;
    
    NTL::zz_p::init(kMod);
    int k = kBlockSize*kBlocksPerRank;
    int n = kBlockSize*kBlocksPerLength;

    
#if QUASI_CYCLIC==1
    std::cout << "Random quasi-cyclic "<<k<<"x"<<n<<" codes\n";
#else
    std::cout << "Random "<<k<<"x"<<n<<" codes\n";
#endif

    int nIter = 500;
    int d = 2;
    int nRankDeficient = 0;
    int nDiscarded = 0;

    for (int i = 0; i < nIter; i++) {
#if QUASI_CYCLIC==1
        PRvector prvFull = quasiCircular(k);
        PRvector prv = truncateCode(prvFull, n);
#else
        PRvector prv(k, n, 0);
#endif
        auto& C = prv.getMat();
        NTL::mat_zz_p Ccopy = C.GetNTLMatrix();
        int codeRank = gauss(Ccopy);
        if (codeRank != k) {
            nDiscarded++;
            i--;  // retry this iteration
            continue;
        }
        if (analyzeAlgebraicAttack(prv))
            nRankDeficient++;
    }
    std::cout << "\n=== Summary: " << nRankDeficient << " out of " << nIter
        << " codes had a degree-" << d << " annihilating polynomial"
        << " (" << nDiscarded << " non-full-rank codes discarded) ===\n";
    return 0;
}

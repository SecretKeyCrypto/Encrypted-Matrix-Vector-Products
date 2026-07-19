// Implementation of pseudo-random vector generator for EMVP experiments

#include "prvector.h"
#include "constants.h"
#include <NTL/lzz_p.h>
#include <cmath>

using namespace NTL;

namespace emvpexpr {

PRvector::PRvector(int k, int n, int d, int b) : k_(k), n_(n) {
    if (k >= n || b*d > n) {
        throw std::invalid_argument("we need k < n and b*d <= n");
    }
    if (d <= 0) {  // Generate a full-rank code
        M_ = randomMatrix(k, n);
    } else {  // Generate code with d-rank-deficiency in the 1st d blocks
        if (b <= 2) {
            throw std::invalid_argument("block-size b must be at least 3");
        }
        M_ = lowRankMatrix(k, n, b*d, d);
    }
    numBlocks_ = (n+b-1) / b;
}

PRvector::PRvector(const Matrix& M) : k_(M.NumRows()), n_(M.NumCols()), M_(M) {
    numBlocks_ = (n_ + kBlockSize -1) / kBlockSize;
}

vec_zz_p PRvector::generateVector() {
    // Generate random m-vector r
    vec_zz_p r;
    NTL::random(r, k_);
    
    // Generate random non-zero (n/kBlockSize)-vector a
    vec_zz_p a;
    a.SetLength(numBlocks_);
    for (int i = 0; i < numBlocks_; i++) {
        do {
            random(a[i]);
        } while (IsZero(a[i]));
    }
    
    // Compute c = r * M
    vec_zz_p c;
    NTL::mul(c, r, M_.GetNTLMatrix());
    
    // Multiply entries in i-th block by a[i]
    for (int i = 0; i < numBlocks_; i++) {
        for (int j = 0; j < kBlockSize; j++) {
            long ind = i * kBlockSize + j;
            c[ind] *= a[i];
        }
    }    
    return c;
}
}

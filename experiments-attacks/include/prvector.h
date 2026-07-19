// Pseudo-random vector generator for EMVP experiments

#ifndef EMVPEXPR_PRVECTOR_H
#define EMVPEXPR_PRVECTOR_H

#include <NTL/vec_lzz_p.h>
#include "constants.h"
#include "utils.h"

namespace emvpexpr {

/// Pseudo-random vector generator for EMVP experiments
class PRvector {
private:
    int k_;
    int n_;
    int numBlocks_;
    Matrix M_;

public:
    /// Initialize PRvector with parameters k, n, d, b:
    /// (k,n) code with rank-deficiency of d in the 1st d*b columns.
    /// @throws std::invalid_argument if constraints not satisfied
    PRvector(int k, int n, int d=0, int b=kDefaultBlockSize);
    explicit PRvector(const Matrix& M);
    
    /// Generate a pseudo-random vector
    NTL::vec_zz_p generateVector();
    
    /// Get parameters
    int getK() const { return k_; }
    int getN() const { return n_; }
    const Matrix& getMat() const { return M_; }
};

}

#endif // EMVPEXPR_PRVECTOR_H

// Utility classes and functions for EMVP experiments
#ifndef EMVPEXPR_UTILS_H
#define EMVPEXPR_UTILS_H

#include <NTL/mat_lzz_p.h>
#include <NTL/vec_lzz_p.h>
#include <vector>

namespace emvpexpr {

class PRvector; // Forward declaration

/// Wrapper around NTL's mat_zz_p for single-precision modular matrices
class Matrix {
private:
    ::NTL::mat_zz_p mat_;  // Internal NTL matrix

public:
    /// Default constructor: empty 0-by-0 matrix
    Matrix() = default;
    
    /// Constructor: empty m-by-n matrix
    Matrix(int m, int n);

    /// Constructor: wrap an NTL matrix
    explicit Matrix(const ::NTL::mat_zz_p& mat);
    
    /// Get number of rows
    int NumRows() const { return mat_.NumRows(); }
    /// Get number of columns
    int NumCols() const { return mat_.NumCols(); }
    
    /// Access matrix element (i,j)
    ::NTL::zz_p& operator()(int i, int j) { return mat_[i][j]; }
    const ::NTL::zz_p& operator()(int i, int j) const { return mat_[i][j]; }
    
    /// Access underlying NTL matrix
    const ::NTL::mat_zz_p& GetNTLMatrix() const { return mat_; }
    ::NTL::mat_zz_p& GetNTLMatrix() { return mat_; }
};

/// Generate random m-by-n matrix
Matrix randomMatrix(int m, int n);

/// Compute rank of submatrix formed by the first d blocks
int rankOfSubmat(const Matrix& mat, int d);

/// Generate k-by-n matrix with rank r-d in the first r columns
Matrix lowRankMatrix(int k, int n, int r, int d);

/// Return an instance of PRvector with a quasi-circular k-by-2k matrix
PRvector quasiCircular(int k);

/// Compute the tensoring of two vectors
std::vector<::NTL::zz_p> tensorProduct(
    const std::vector<::NTL::zz_p>& v1, const std::vector<::NTL::zz_p>& v2);

/// Generate next combination in lexicographic order
bool nextCombination(std::vector<int>& indices, int n, int d);

/// Compute ordered tensor product for monomials with non-decreasing indices
std::vector<::NTL::zz_p> orderedTensor(const ::NTL::vec_zz_p& v, int d);
}
#endif // EMVPEXPR_UTILS_H

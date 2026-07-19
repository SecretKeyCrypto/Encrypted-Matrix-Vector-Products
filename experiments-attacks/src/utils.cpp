// Implementation of utility classes and functions for EMVP experiments

#include "utils.h"
#include "prvector.h"
#include "constants.h"
#include <NTL/lzz_p.h>
#include <NTL/lzz_pE.h>
#include <NTL/mat_lzz_p.h>
#include <iostream>

using namespace NTL;

namespace emvpexpr {

// amazonq-ignore-next-line
Matrix::Matrix(int m, int n) : mat_(NTL::INIT_SIZE, m, n) {}

Matrix::Matrix(const mat_zz_p& mat) : mat_(mat) {}
Matrix randomMatrix(int m, int n) {
    Matrix result(m, n);
    random(result.GetNTLMatrix(), m, n);  // Use NTL's random matrix function
    return result;
}

int rankOfSubmat(const Matrix& mat, int d) {
    int m = mat.NumRows();
    int n = mat.NumCols();
    int selectedCols = d * kBlockSize;

    // Edge cases
    if (selectedCols <= 0) return 0;

    if (selectedCols > n) { // The entire matrix
        selectedCols = n;
    }
    
    // Extract submatrix with only the first selectedCols columns
    mat_zz_p submat(INIT_SIZE, m, selectedCols);
    for (int k = 0; k < selectedCols; k++) {
        for (int i = 0; i < m; i++) {
            submat[i][k] = mat(i, k);
        }
    }
    return gauss(submat);
}

// Helper function: copy a vector as the j'th column of the matrix
// (Assumes that the dimensions match)
static void copyColumn(NTL::mat_zz_p& matrix, int j, const NTL::vec_zz_p& vec) {
    for (long i = 0; i < vec.length(); i++) {
        matrix[i][j] = vec[i];
    }
}

// Generate k-by-n matrix with rank r-d in the first r columns
Matrix lowRankMatrix(int k, int n, int r, int d) {
    if (d < 0 || r <= d || n < r || n <= k) {  // Ensure n > k, n >= r > d >= 0
        std::cout << "invalid arguments (n,k,r,d)=("<<n<<','<<k<<','<<r<<','<<d<<")\n";
        return Matrix(0,0);
    }

    // Start from a full-rank (r-d)-by-k basis matrix
    NTL::mat_zz_p basis;
    do {
        NTL::random(basis, r-d, k);
    } while (NTL::gauss(basis) < r-d);

    NTL::mat_zz_p mat(INIT_SIZE, k, n);

    // The first r-d columns of the matrix are the basis rows
    for (int j = 0; j < r-d; j++) {
        copyColumn(mat, j, basis[j]);
    }
    // The next d columns of the matrix are Rj*basis for a random Rj's
    for (int j = 0; j < d; j++) {
        NTL::vec_zz_p rj;
        random(rj, r-d);
        copyColumn(mat, r+j, rj * basis);
    }
    // The last n-k columns are random
    // amazonq-ignore-next-line
    for (int j = r; j < n; j++) {
        NTL::vec_zz_p rj;
        random(rj, k);
        copyColumn(mat, j, rj);
    }
    // Wrap with a Matrix object
    Matrix result(mat);
    // std::cout << "generated matrix with rank(1st "<<r<<" columns)=" << rankOfSubmat(result, r) << std::endl;
    return result;
}

PRvector quasiCircular(int k) {
    // Set up NTL zz_pE ring modulo prime kMod and polynomial X^k+1
    zz_pX P;
    SetCoeff(P, k, 1); // X^k
    SetCoeff(P, 0, -1); // -1
    zz_pE::init(P);

    // Choose random element c in the ring
    zz_pE c = random_zz_pE();

    // Generate k-by-2k matrix M such that vec(x)*M = (vec(x) | vec(c*x))
    mat_zz_p M(INIT_SIZE, k, 2*k);
    for (int i = 0; i < k; i++) {
        M[i][i] = 1;  // Identity part: vec(x)

        // Multiplication by c part: vec(c*x)
        zz_pX xi(NTL::INIT_MONO, i);  // Set to X^i
        zz_pE cx = c * conv<zz_pE>(xi);
        for (int j = 0; j < k; j++) {
            M[i][j + k] = coeff(rep(cx), j);
        }
    }

    // Sanity check: verify rep(r) * M = (rep(r) | rep(r*c))
    zz_pE r = random_zz_pE();           // a random element
    NTL::vec_zz_p r_vec = rep(r).rep;   // its representation as a vector
    NTL::vec_zz_p rc_vec = rep(r * c).rep;
    // Is either of them has lower degree than d, pad with zeros
    while (r_vec.length()<k) {
        r_vec.append(zz_p::zero());
    }
    while (rc_vec.length()<k) {
        rc_vec.append(zz_p::zero());
    }
    vec_zz_p result = r_vec * M;  // verify that r_vec * M == (r_vec | rc_vec)
    for (int i = 0; i < k; i++) {
        if (result[i] != r_vec[i] || result[i+k] != rc_vec[i]) {
            std::cout << "Quasi-circular matrix sanity check FAILED" << std::endl;
            // amazonq-ignore-next-line
            exit(0);
        }
    }
    // Return a PRvector objest with that matrix
    Matrix matrix_obj(M);
    return PRvector(matrix_obj);
}

std::vector<NTL::zz_p> tensorProduct(
    const std::vector<NTL::zz_p>& v1, const std::vector<NTL::zz_p>& v2) {
    int len1 = v1.size();
    int len2 = v2.size();
    std::vector<NTL::zz_p> result(len1 * len2);
    for (int i = 0; i < len1; i++) {
        for (int j = 0; j < len2; j++) {
            NTL::mul(result[i * len2 + j], v1[i], v2[j]);
        }
    }    
    return result;
}

bool nextCombination(std::vector<int>& indices, int n, int d) {
    int i = d - 1;
    while (i >= 0 && indices[i] == n - 1) {
        i--;
    }
    if (i < 0) return false;  // All combinations generated
    
    indices[i]++;  // increase one coordinate
    
    // Ensure monotonicity: indices[x] >= indices[y] for all x >= y
    for (int j = i + 1; j < d; j++) {
        // amazonq-ignore-next-line
        indices[j] = indices[i];
    }
    return true;
}

std::vector<NTL::zz_p> orderedTensor(const NTL::vec_zz_p& v, int d) {
    int n = v.length();
    // amazonq-ignore-next-line
    std::vector<NTL::zz_p> result;
    
    // Generate all combinations with repetition of length d from n variables
    std::vector<int> indices(d, 0);
    do {
        // Compute monomial for current indices
        NTL::zz_p monomial(1L);
        for (int i = 0; i < d; i++) {
            NTL::mul(monomial, monomial, v[indices[i]]);
        }
        result.push_back(monomial);
    } while (nextCombination(indices, n, d));
    return result;
}

}
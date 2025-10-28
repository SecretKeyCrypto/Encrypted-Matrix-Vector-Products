#include <iostream>

#include <vector>
#include <NTL/ZZ_p.h>
#include <NTL/ZZ_pX.h>
#include <cstdint>

using namespace NTL;

extern "C" {

void GenerateSystematicRSMatrix_uint32(
    uint32_t n, uint32_t m, uint32_t q, uint32_t m_0,
    const uint32_t* alphas_in, // length n
    uint32_t* output           // length n * m, row-major
) {
    ZZ_p::init(ZZ(q));  // Initialize the field F_q

    // Convert alphas_in to Vec<ZZ_p>
    Vec<ZZ_p> alphas;
    alphas.SetLength(n);
    for (uint32_t i = 0; i < n; i++) {
        alphas[i] = conv<ZZ_p>(alphas_in[i]);
    }

    // Compute Lagrange basis polynomials for interpolation
    Vec<ZZ_pX> lagrange_basis;
    lagrange_basis.SetLength(m);

    for (uint32_t i = 0; i < m; i++) {
        ZZ_pX numer;
        SetCoeff(numer, 0, ZZ_p(1));  // initialize to 1
        ZZ_p denom = ZZ_p(1);

        for (uint32_t j = 0; j < m; j++) {
            if (i == j) continue;
            ZZ_pX term;
            SetCoeff(term, 1, ZZ_p(1));              // x
            SetCoeff(term, 0, -alphas[j]);           // (x - α_j)
            numer *= term;
            denom *= (alphas[i] - alphas[j]);
        }

        lagrange_basis[i] = numer / denom;
    }

    // Fill G row by row: top m rows = identity, bottom = evaluations
    for (uint32_t row = m_0; row < n; row++) {
        for (uint32_t col = 0; col < m; col++) {
            ZZ_p val;
            if (row < m) {
                val = (row == col ? ZZ_p(1) : ZZ_p(0));
            } else {
                val = eval(lagrange_basis[col], alphas[row]);

            }
            output[(row - m_0) * m + col] = conv<uint32_t>(rep(val));  // store raw value as uint32_t
        }
    }
}

uint32_t LagrangeInterpEval(const uint32_t* x_in, const uint32_t* y_in, uint32_t k, uint32_t eval_point, uint32_t q) {
    ZZ_p::init(ZZ(q));

    std::vector<ZZ_p> x(k), y(k);
    for (uint32_t i = 0; i < k; i++) {
        x[i] = ZZ_p(x_in[i]);
        y[i] = ZZ_p(y_in[i]);
    }

    ZZ_p result = ZZ_p(0);
    for (uint32_t i = 0; i < k; ++i) {
        ZZ_p term = y[i];
        for (uint32_t j = 0; j < k; ++j) {
            if (i != j) {
                term *= (ZZ_p(eval_point) - x[j]) / (x[i] - x[j]);
            }
        }
        result += term;
    }

    return conv<uint32_t>(rep(result));
}

}
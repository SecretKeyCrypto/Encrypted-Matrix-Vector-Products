#include "plainRS.h"
#include <cstdint>
#include <cassert>
#include <memory.h>

#ifdef __cplusplus
extern "C" {
#endif

// -------------------- Modular arithmetic helpers --------------------

static inline uint32_t mod_add_u32(uint32_t a, uint32_t b, uint32_t q) {
    uint32_t s = a + b;
    if (s >= q) s -= q;
    return s;
}

static inline uint32_t mod_sub_u32(uint32_t a, uint32_t b, uint32_t q) {
    return (a >= b) ? (a - b) : (uint32_t)(a + (uint64_t)q - b);
}

static inline uint32_t mod_mul_u32(uint32_t a, uint32_t b, uint32_t q) {
    // Use 64-bit intermediate to avoid overflow
    uint64_t p = (uint64_t)a * (uint64_t)b;
    return (uint32_t)(p % q);
}

// Extended GCD for modular inverse (works if gcd(a, q) == 1)
static uint32_t mod_inv_u32(uint32_t a, uint32_t q) {
    // a * x + q * y = gcd(a, q)  -> if gcd == 1, x is the inverse mod q
    int64_t t0 = 0, t1 = 1;
    int64_t r0 = (int64_t)q, r1 = (int64_t)a;

    while (r1 != 0) {
        int64_t qout = r0 / r1;
        int64_t r2 = r0 - qout * r1; r0 = r1; r1 = r2;
        int64_t t2 = t0 - qout * t1; t0 = t1; t1 = t2;
    }

    // If r0 != 1, inverse doesn't exist (caller is responsible to ensure prime q and nonzero denom)
    // Normalize t0 mod q
    int64_t inv = t0 % (int64_t)q;
    if (inv < 0) inv += q;
    return (uint32_t)inv;
}

// -------------------- Lagrange tools (no polynomials) --------------------
// All outside inputs are assumed to be within the modulus range.

// Precompute barycentric weights w_i = 1 / Π_{j!=i}(x_i - x_j) mod q
static void barycentric_weights_u32(const uint32_t* x, uint32_t k, uint32_t q, uint32_t* w) {
    memset(w, 0, k * sizeof(uint32_t));
    for (uint32_t i = 0; i < k; ++i) {
        uint32_t den = 1;
        for (uint32_t j = 0; j < k; ++j) if (i != j) {
            uint32_t diff = mod_sub_u32(x[i], x[j], q);
            assert(diff != 0 && "Duplicate nodes (mod q) or non-invertible denominator.");
            den = mod_mul_u32(den, diff, q);
        }
        // inverse exists only if gcd(den, q) == 1; assume q prime and x distinct
        w[i] = mod_inv_u32(den, q);
    }
}

// Evaluate the i-th Lagrange basis L_i at point x* using precomputed weights:
// L_i(x*) = w_i * Π_{j!=i}(x* - x_j)
static uint32_t lagrange_basis_eval_i_u32(uint32_t i, const uint32_t* x, const uint32_t* w,
                                          uint32_t k, uint32_t x_star, uint32_t q) {
    uint32_t num = 1;
    for (uint32_t j = 0; j < k; ++j) if (i != j) {
        uint32_t term = mod_sub_u32(x_star, x[j], q);
        num = mod_mul_u32(num, term, q);
    }
    return mod_mul_u32(w[i], num, q);
}

// -------------------- Public API --------------------

// Build an n x m *systematic* RS generator over F_q at evaluation points alphas_in.
// Assumes: 1 <= m <= n, q is prime, and the first m alphas are pairwise distinct mod q.
// Output layout: row-major, length n*m. Top m rows = identity.
bool PlainGenerateSystematicRSMatrix(
    DoContext* doctx, uint32_t n, uint32_t m, uint32_t q,
    const uint32_t* alphas_in, // length n (evaluation points)
    uint32_t* output           // length n * m, row-major
) {
    assert(q >= 2);
    assert(m >= 1 && m <= n);

    // We’ll use the first m alphas as interpolation nodes for the Lagrange basis.
    // Precompute barycentric weights for nodes X = alphas_in[0..m-1]
    uint32_t w[m];
    barycentric_weights_u32(alphas_in, m, q, w);

    // Fill the generator matrix
    // Top m rows are identity: G[row, col] = (row == col ? 1 : 0)
    for (uint32_t row = 0; row < m; ++row) {
        for (uint32_t col = 0; col < m; ++col) {
            output[row * m + col] = (row == col) ? 1u : 0u;
        }
    }

    // Remaining rows: evaluate each L_j at alpha_row
    for (uint32_t row = m; row < n; ++row) {
        uint32_t xstar = alphas_in[row];

        // Fast path: if xstar equals one of the first m nodes, row becomes that basis vector
        bool matched = false;
        for (uint32_t j = 0; j < m; ++j) {
            if (xstar == (alphas_in[j])) {
                // Row = e_j
                for (uint32_t col = 0; col < m; ++col) output[row * m + col] = 0u;
                output[row * m + j] = 1u;
                matched = true;
                break;
            }
        }
        if (matched) continue;

        // General case: compute each L_j(xstar)
        for (uint32_t col = 0; col < m; ++col) {
            uint32_t lij = lagrange_basis_eval_i_u32(col, alphas_in, w, m, xstar, q);
            output[row * m + col] = lij;
        }
    }
    return true;
}

// Evaluate the Lagrange interpolant at eval_point from k nodes (x_in, y_in) mod q.
// Assumes: x_i are pairwise distinct mod q, and denominators are invertible mod q.
bool PlainLagrangeInterpEval(
    DoContext* doctx, uint32_t* result, const uint32_t* x_in, const uint32_t* y_in,
    uint32_t k, uint32_t eval_point, uint32_t q
) {
    assert(q >= 2);
    assert(k >= 1);

    // Precompute weights
    uint32_t w[k];
    barycentric_weights_u32(x_in, k, q, w);

    // If eval_point equals one of the x_i, return that y_i directly
    uint32_t xstar = eval_point;
    for (uint32_t i = 0; i < k; ++i) {
        if (xstar == (x_in[i])) {
            *result = y_in[i];
            return true;
        }
    }

    // General Lagrange sum: sum_i y_i * L_i(xstar)
    uint32_t acc = 0;
    for (uint32_t i = 0; i < k; ++i) {
        uint32_t li = lagrange_basis_eval_i_u32(i, x_in, w, k, xstar, q);
        uint32_t term = mod_mul_u32(y_in[i], li, q);
        acc = mod_add_u32(acc, term, q);
    }
    *result = acc;
    return true;
}

static inline bool isAllFalse(const bool* noisyQuery, uint32_t k) {
    for (uint32_t i = 0; i < k; ++i) {
        if (noisyQuery[i]) {
            return false;
        }
    }
    return true;
}

static bool ReedSolomonDecode1(
    DoContext* doctx, uint32_t* code, const bool* noisyQuery, uint32_t ecc_len, uint32_t ecc_k, uint32_t q, uint32_t* success
) {
    if (!isAllFalse(noisyQuery, ecc_k)) {
        uint32_t x_in[ecc_k];
        uint32_t y_in[ecc_k];

        uint32_t idx = 0;
        for (uint32_t i = 0; i < ecc_len; ++i) {
            if (!noisyQuery[i] && idx < ecc_k) {
                x_in[idx] = i;
                y_in[idx] = code[i];
                ++idx;
            }
        }

        if (idx < ecc_k) {
            *success = 0;
            return false;
        }

        for (uint32_t i = 0; i < ecc_k; ++i) {
            if (noisyQuery[i]) {
                PlainLagrangeInterpEval(doctx, &code[i], x_in, y_in, ecc_k, i, q);
            }
        }
    }

    *success = 1;
    return true;
}

bool PlainReedSolomonDecode(DoContext* doctx,
                       uint32_t* code,
                       uint64_t co,
                       uint64_t cs,
                       const bool* noisyQuery,
                       uint32_t ecc_len,
                       uint32_t ecc_k,
                       uint32_t q,
                       uint32_t* success,
                       uint64_t steps)
{
    bool all = true;
    code += co;
    for (uint64_t i = 0; i < steps; i++, code += cs, success++) {
        if (!ReedSolomonDecode1(doctx, code, noisyQuery, ecc_len, ecc_k, q, success)) {
            all = false;
        }
    }
    return all;
}

#ifdef __cplusplus
} // extern "C"
#endif

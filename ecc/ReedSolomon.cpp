#include <cstdint>
#include <vector>
#include <cassert>

extern "C" {

// -------------------- Modular arithmetic helpers --------------------

static inline uint32_t mod_norm_u32(uint64_t a, uint32_t q) {
    a %= q;
    return static_cast<uint32_t>(a);
}

static inline uint32_t mod_add_u32(uint32_t a, uint32_t b, uint32_t q) {
    uint32_t s = a + b;
    if (s >= q || s < a) s -= q; // handle wrap
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
    int64_t r0 = (int64_t)q, r1 = (int64_t)(a % q);

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
// Precompute barycentric weights w_i = 1 / Π_{j!=i}(x_i - x_j) mod q
static bool barycentric_weights_u32(const uint32_t* x, uint32_t k, uint32_t q, std::vector<uint32_t>& w) {
    w.assign(k, 0);
    for (uint32_t i = 0; i < k; ++i) {
        uint32_t den = 1;
        for (uint32_t j = 0; j < k; ++j) if (i != j) {
            uint32_t diff = mod_sub_u32(x[i] % q, x[j] % q, q);
            if (diff == 0) return false; // duplicate nodes mod q → invalid
            den = mod_mul_u32(den, diff, q);
        }
        // inverse exists only if gcd(den, q) == 1; assume q prime and x distinct
        w[i] = mod_inv_u32(den, q);
    }
    return true;
}

// Evaluate the i-th Lagrange basis L_i at point x* using precomputed weights:
// L_i(x*) = w_i * Π_{j!=i}(x* - x_j)
static uint32_t lagrange_basis_eval_i_u32(uint32_t i, const uint32_t* x, const std::vector<uint32_t>& w,
                                          uint32_t k, uint32_t x_star, uint32_t q) {
    uint32_t num = 1;
    for (uint32_t j = 0; j < k; ++j) if (i != j) {
        uint32_t term = mod_sub_u32(x_star % q, x[j] % q, q);
        num = mod_mul_u32(num, term, q);
    }
    return mod_mul_u32(w[i], num, q);
}

// -------------------- Public API --------------------

// Build an n x m *systematic* RS generator over F_q at evaluation points alphas_in.
// Assumes: 1 <= m <= n, q is prime, and the first m alphas are pairwise distinct mod q.
// Output layout: row-major, length n*m. Top m rows = identity.
void GenerateSystematicRSMatrix_uint32(
    uint32_t n, uint32_t m, uint32_t q,
    const uint32_t* alphas_in, // length n (evaluation points)
    uint32_t* output           // length n * m, row-major
) {
    // Basic preconditions (debug-time checks; remove or replace with your own handling)
    assert(q >= 2);
    assert(m >= 1 && m <= n);

    // We’ll use the first m alphas as interpolation nodes for the Lagrange basis.
    // Precompute barycentric weights for nodes X = alphas_in[0..m-1]
    std::vector<uint32_t> w;
    bool ok = barycentric_weights_u32(alphas_in, m, q, w);
    // If this assert fires, either there are duplicate nodes mod q or q isn’t usable.
    assert(ok && "Duplicate nodes (mod q) or non-invertible denominator.");

    // Fill the generator matrix
    // Top m rows are identity: G[row, col] = (row == col ? 1 : 0)
    for (uint32_t row = 0; row < m; ++row) {
        for (uint32_t col = 0; col < m; ++col) {
            output[row * m + col] = (row == col) ? 1u : 0u;
        }
    }

    // Remaining rows: evaluate each L_j at alpha_row
    for (uint32_t row = m; row < n; ++row) {
        uint32_t xstar = alphas_in[row] % q;

        // Fast path: if xstar equals one of the first m nodes, row becomes that basis vector
        bool matched = false;
        for (uint32_t j = 0; j < m; ++j) {
            if (xstar == (alphas_in[j] % q)) {
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
}

// Evaluate the Lagrange interpolant at eval_point from k nodes (x_in, y_in) mod q.
// Assumes: x_i are pairwise distinct mod q, and denominators are invertible mod q.
uint32_t LagrangeInterpEval(
    const uint32_t* x_in, const uint32_t* y_in, uint32_t k,
    uint32_t eval_point, uint32_t q
) {
    assert(q >= 2);
    assert(k >= 1);

    // Precompute weights
    std::vector<uint32_t> w;
    bool ok = barycentric_weights_u32(x_in, k, q, w);
    assert(ok && "Duplicate nodes (mod q) or non-invertible denominator.");

    // If eval_point equals one of the x_i, return that y_i directly
    uint32_t xstar = eval_point % q;
    for (uint32_t i = 0; i < k; ++i) {
        if (xstar == (x_in[i] % q)) {
            return mod_norm_u32(y_in[i], q);
        }
    }

    // General Lagrange sum: sum_i y_i * L_i(xstar)
    uint32_t acc = 0;
    for (uint32_t i = 0; i < k; ++i) {
        uint32_t li = lagrange_basis_eval_i_u32(i, x_in, w, k, xstar, q);
        uint32_t term = mod_mul_u32(mod_norm_u32(y_in[i], q), li, q);
        acc = mod_add_u32(acc, term, q);
    }
    return acc;
}

} // extern "C"

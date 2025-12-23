#include "plainmatrices.h"

#if defined(__SSE2__) || defined(__AVX2__)
#include <immintrin.h>
#endif

inline void NoSimdMatrixTranspose(DoContext* ctx, uint32_t* result, const uint32_t* matrix, uint32_t M, uint32_t N) {
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            result[j * M + i] = matrix[i * N + j];
        }
    }
}

#ifdef __SSE2__

inline void transpose4x4_sse2(uint32_t* dst, const uint32_t* src, uint32_t dst_stride, uint32_t src_stride) {
    __m128i row0 = _mm_loadu_si128((__m128i*)(src + 0 * src_stride));
    __m128i row1 = _mm_loadu_si128((__m128i*)(src + 1 * src_stride));
    __m128i row2 = _mm_loadu_si128((__m128i*)(src + 2 * src_stride));
    __m128i row3 = _mm_loadu_si128((__m128i*)(src + 3 * src_stride));

    __m128i t0 = _mm_unpacklo_epi32(row0, row1);
    __m128i t1 = _mm_unpackhi_epi32(row0, row1);
    __m128i t2 = _mm_unpacklo_epi32(row2, row3);
    __m128i t3 = _mm_unpackhi_epi32(row2, row3);

    row0 = _mm_unpacklo_epi64(t0, t2);
    row1 = _mm_unpackhi_epi64(t0, t2);
    row2 = _mm_unpacklo_epi64(t1, t3);
    row3 = _mm_unpackhi_epi64(t1, t3);

    _mm_storeu_si128((__m128i*)(dst + 0 * dst_stride), row0);
    _mm_storeu_si128((__m128i*)(dst + 1 * dst_stride), row1);
    _mm_storeu_si128((__m128i*)(dst + 2 * dst_stride), row2);
    _mm_storeu_si128((__m128i*)(dst + 3 * dst_stride), row3);
}

inline void SSE2MatrixTranspose(DoContext* ctx, uint32_t* result, const uint32_t* matrix, uint32_t M, uint32_t N) {
    const uint32_t blockSize = 4;

    for (uint32_t i = 0; i < M; i += blockSize) {
        for (uint32_t j = 0; j < N; j += blockSize) {
            uint32_t rows = (i + blockSize <= M) ? blockSize : M - i;
            uint32_t cols = (j + blockSize <= N) ? blockSize : N - j;

            if (rows == blockSize && cols == blockSize) {
                transpose4x4_sse2(&result[j * M + i], &matrix[i * N + j], M, N);
            } else {
                // Fallback to scalar transpose for edge blocks
                for (uint32_t ii = 0; ii < rows; ++ii)
                    for (uint32_t jj = 0; jj < cols; ++jj)
                        result[(j + jj) * M + (i + ii)] = matrix[(i + ii) * N + (j + jj)];
            }
        }
    }
}
#endif

#ifdef __AVX2__

inline void transpose8x8_avx2(uint32_t* dst, const uint32_t* src,
                              uint32_t dst_stride, uint32_t src_stride) {
    __m256i row[8];
    for (int i = 0; i < 8; ++i)
        row[i] = _mm256_loadu_si256((__m256i*)(src + i * src_stride));

    // First level of interleave
    __m256i tmp0 = _mm256_unpacklo_epi32(row[0], row[1]);
    __m256i tmp1 = _mm256_unpackhi_epi32(row[0], row[1]);
    __m256i tmp2 = _mm256_unpacklo_epi32(row[2], row[3]);
    __m256i tmp3 = _mm256_unpackhi_epi32(row[2], row[3]);
    __m256i tmp4 = _mm256_unpacklo_epi32(row[4], row[5]);
    __m256i tmp5 = _mm256_unpackhi_epi32(row[4], row[5]);
    __m256i tmp6 = _mm256_unpacklo_epi32(row[6], row[7]);
    __m256i tmp7 = _mm256_unpackhi_epi32(row[6], row[7]);

    // Second level
    __m256i t0 = _mm256_unpacklo_epi64(tmp0, tmp2);
    __m256i t1 = _mm256_unpackhi_epi64(tmp0, tmp2);
    __m256i t2 = _mm256_unpacklo_epi64(tmp1, tmp3);
    __m256i t3 = _mm256_unpackhi_epi64(tmp1, tmp3);
    __m256i t4 = _mm256_unpacklo_epi64(tmp4, tmp6);
    __m256i t5 = _mm256_unpackhi_epi64(tmp4, tmp6);
    __m256i t6 = _mm256_unpacklo_epi64(tmp5, tmp7);
    __m256i t7 = _mm256_unpackhi_epi64(tmp5, tmp7);

    // Final shuffle across 128‑bit lanes
    __m256i r0 = _mm256_permute2x128_si256(t0, t4, 0x20);
    __m256i r1 = _mm256_permute2x128_si256(t1, t5, 0x20);
    __m256i r2 = _mm256_permute2x128_si256(t2, t6, 0x20);
    __m256i r3 = _mm256_permute2x128_si256(t3, t7, 0x20);
    __m256i r4 = _mm256_permute2x128_si256(t0, t4, 0x31);
    __m256i r5 = _mm256_permute2x128_si256(t1, t5, 0x31);
    __m256i r6 = _mm256_permute2x128_si256(t2, t6, 0x31);
    __m256i r7 = _mm256_permute2x128_si256(t3, t7, 0x31);

    _mm256_storeu_si256((__m256i*)(dst + 0 * dst_stride), r0);
    _mm256_storeu_si256((__m256i*)(dst + 1 * dst_stride), r1);
    _mm256_storeu_si256((__m256i*)(dst + 2 * dst_stride), r2);
    _mm256_storeu_si256((__m256i*)(dst + 3 * dst_stride), r3);
    _mm256_storeu_si256((__m256i*)(dst + 4 * dst_stride), r4);
    _mm256_storeu_si256((__m256i*)(dst + 5 * dst_stride), r5);
    _mm256_storeu_si256((__m256i*)(dst + 6 * dst_stride), r6);
    _mm256_storeu_si256((__m256i*)(dst + 7 * dst_stride), r7);
}

inline void AVX2MatrixTranspose(DoContext* ctx, uint32_t* result, const uint32_t* matrix, uint32_t M, uint32_t N) {
    const uint32_t blockSize = 8;  // Use 8×8 blocks for AVX2

    for (uint32_t i = 0; i < M; i += blockSize) {
        for (uint32_t j = 0; j < N; j += blockSize) {
            uint32_t rows = (i + blockSize <= M) ? blockSize : M - i;
            uint32_t cols = (j + blockSize <= N) ? blockSize : N - j;

            if (rows == blockSize && cols == blockSize) {
                transpose8x8_avx2(&result[j * M + i], &matrix[i * N + j], M, N);
            } else {
                // Fallback to scalar transpose for edge blocks
                for (uint32_t ii = 0; ii < rows; ++ii)
                    for (uint32_t jj = 0; jj < cols; ++jj)
                        result[(j + jj) * M + (i + ii)] = matrix[(i + ii) * N + (j + jj)];
            }
        }
    }
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

bool PlainMatrixTranspose(DoContext* ctx, uint32_t* result, uint32_t ro, const uint32_t* matrix, uint32_t mo, uint32_t M, uint32_t N) {
#if defined(__AVX2__)
    AVX2MatrixTranspose(ctx, result + ro, matrix + mo, M, N);
#elif defined(__SSE2__)
    SSE2MatrixTranspose(ctx, result + ro, matrix + mo, M, N);
#else
    NoSimdMatrixTranspose(ctx, result + ro, matrix + mo, M, N);
#endif
    return true;
}

#ifdef __cplusplus
} /* extern "C" */
#endif

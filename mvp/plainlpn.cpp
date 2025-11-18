#include "plainlpn.h"

#include "../dataobjects/dataobj.h"
#include "../dataobjects/fields.h"
#include "../tdm/tdm.h"
#include "mvp.h"

#ifdef __cplusplus
extern "C" {
#endif

bool PlainLpnEncode(
    DoContext* ctx,
    const uint32_t* input,            // [M * L]
    const uint32_t* rlcMatrix,        // [K * L]
    const uint32_t* generatorMatrix,  // [ECCLength * M_1]
    uint32_t* encoded,                // [ECCLength * M / M_1 * N]
    uint32_t M, uint32_t L, uint32_t K,
    uint32_t M_1, uint32_t ECCLength,
    uint32_t P
) {
    alignas(ALIGNMENT) uint32_t message[ECCLength] = {0};

    // Derived quantities
    uint32_t N = K + L;
    uint32_t rowPerSlice   = M / M_1;
    uint32_t entryPerSlice = rowPerSlice * N;

	for (uint32_t i = 0; i < rowPerSlice; i++) {
		for (uint32_t j = 0; j < M_1; j++) {
			// Input matrix with each row length L, block size M_1
			uint64_t inputStart = (i*M_1 + j) * L;

			// Put into the jth slice, ith row, each row with length N
			uint64_t outputStart = j*entryPerSlice + i*N;

			// Copy the input row into the first L element of the output row
			FieldCopyVector(ctx, encoded, outputStart, input, inputStart, L);

			MatVecProductExt(ctx, rlcMatrix, 0, 0, input, inputStart, L, encoded, outputStart+L, N, K, L, 1, P);
		}

		// Encode each M_1 length slice with ECC to length ECCLength
		for (uint32_t j = 0; j < N; j++) {
			// Get the row i, col j of each block, forms a length M_1 message, then Encode
			PermutedExtentsAssign(ctx, message, 0, 1, 0, encoded, i*N+j, entryPerSlice, 0, 1, nullptr, 0, M_1);

			MatVecProductExt(ctx, generatorMatrix, 0, 0, message, 0, 0, message, M_1, 0, ECCLength - M_1, M_1, 1, P);

			// Put to the M_1:ECCLength slice
			PermutedExtentsAssign(ctx, encoded, i*N+j, entryPerSlice, 0, message, 0, 1, 0, 1, nullptr, 0, ECCLength);
		}
	}

    return true;
}

#ifdef __cplusplus
}
#endif

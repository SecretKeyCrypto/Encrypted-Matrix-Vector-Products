#ifndef EMVPEXPR_CONSTANTS_H
#define EMVPEXPR_CONSTANTS_H

namespace emvpexpr {
/// Dimension of each block, can be overwritten by a command-line argument
constexpr int kDefaultBlockSize = 8;

/// Modulus: This is a 29-bit prime, can be overwritten by a command-line argument
constexpr long kDefaultMod = 479001599;

extern long kMod;
extern int kBlockSize;
}
#endif // EMVPEXPR_CONSTANTS_H

# EMVP Tests

Testing algebraic attacks on the 1D-SLSN variant of EMVP.

## Dependencies

- GMP library
- NTL library (version 11.5.1+)

### Install Dependencies

Ubuntu/Debian:
```bash
sudo apt install libgmp-dev libntl-dev
```

## Build

```bash
mkdir build && cd build
cmake ..
make
```

## Example Run

```bash
% ./algebraic-attacks 97 4
Testing algebraic attacks against EMVP with 1D-SLSN
Usage: ./algebraic-attacks [kMod [kBlockSize]]
  kMod = 97 (default: 479001599)
  kBlockSize = 4 (default: 8)

Two types of codes:
 - full-rank qusi-cyclic 8x16 code
 - first d blocks (d<=2) have rank deficiency of d

*********** Full rank quasi-circular code: *********
Rank of C = 8

Testing degree-2, 153x153 matrix
Rank of degree-2 153x153 matrix is 153

Testing degree-3, 969x969 matrix
Rank of degree-3 969x969 matrix is 953

Testing 3-multilinear 48x48 matrix (b=4, lastB=3)
Rank of 3-block-multilinear matrix is 47


*********** 1-rank-defficient code: *********
Rank of C[1st block] = 3

Testing 1-multilinear 4x4 matrix (b=4, lastB=4)
Rank of 1-block-multilinear matrix is 3


*********** 2-rank-defficient code: *********
Rank of C[1st block] = 4
Rank of C[1st 2 blocks] = 6

Testing 1-multilinear 4x4 matrix (b=4, lastB=4)
Rank of 1-block-multilinear matrix is 4

Testing degree-2, 153x153 matrix
Rank of degree-2 153x153 matrix is 120

Testing 2-multilinear 16x16 matrix (b=4, lastB=4)
Rank of 2-block-multilinear matrix is 8
```

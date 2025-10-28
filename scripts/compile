#!/bin/bash

if [[ "$(uname)" == "Darwin" ]]; then
    export CGO_LDFLAGS="-L$(brew --prefix openssl@3)/lib"
    export CGO_CFLAGS="-I$(brew --prefix openssl@3)/include"
    echo "[Info] Using OpenSSL from $(brew --prefix openssl@3)"
fi

# Prefer GCC 17 if available, otherwise fallback to g++
if command -v g++-17 >/dev/null 2>&1; then
    CXX=g++-17
elif command -v g++-13 >/dev/null 2>&1; then
    CXX=g++-13
elif command -v clang++ >/dev/null 2>&1; then
    CXX=clang++
else
    CXX=g++
fi

ARCH=$(uname -m)

if [ "$ARCH" = "x86_64" ]; then
    SIMD_FLAGS="-maes -msse -msse2 -mavx2"
else
    # Apple Silicon (arm64): use NEON instead of x86 SIMD
    SIMD_FLAGS=""
fi

CXXFLAGS="-std=c++17 -O3 -march=native $SIMD_FLAGS -I/opt/homebrew/include"

$CXX $CXXFLAGS -c dataobjects/fields.cpp -o dataobjects/fields.o -I/opt/homebrew/include
ar rcs dataobjects/libdataobjects.a dataobjects/fields.o

$CXX $CXXFLAGS -c tdm/NTT.cpp -o tdm/NTT.o -I/opt/homebrew/include
ar rcs tdm/libNTT.a tdm/NTT.o

$CXX $CXXFLAGS -c mvp/mvp.cpp -o mvp/mvp.o -I/opt/homebrew/include
$CXX $CXXFLAGS -c mvp/matrixShapeTransform.cpp -o mvp/matrixShapeTransform.o -I/opt/homebrew/include
ar rcs mvp/libMVP.a mvp/mvp.o mvp/matrixShapeTransform.o

$CXX $CXXFLAGS -c ecc/ReedSolomon.cpp -o ecc/ReedSolomon.o -I/opt/homebrew/include
ar rcs ecc/libReedSolomon.a ecc/ReedSolomon.o

$CXX $CXXFLAGS -c utils/rnd_api.cpp -o utils/rnd_api.o -I/opt/homebrew/include
ar rcs utils/libutils.a utils/rnd_api.o

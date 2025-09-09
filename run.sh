g++-13 -std=c++17 -march=native -O3 -c dataobjects/fields.cpp -o dataobjects/fields.o -I/opt/homebrew/include
ar rcs dataobjects/libdataobjects.a dataobjects/fields.o

g++-13 -std=c++17 -march=native -O3 -c tdm/NTT.cpp -o tdm/NTT.o -I/opt/homebrew/include
ar rcs tdm/libNTT.a tdm/NTT.o

g++-13 -std=c++17 -march=native -O3 -c mvp/mvp.cpp -o mvp/mvp.o -I/opt/homebrew/include
ar rcs mvp/libMVP.a mvp/mvp.o

g++-13 -std=c++17 -march=native -O3 -c ecc/ReedSolomon.cpp -o ecc/ReedSolomon.o -I/opt/homebrew/include
ar rcs ecc/libReedSolomon.a ecc/ReedSolomon.o

g++-13 -std=c++17 -march=native -O3 -c utils/rnd_api.cpp -o utils/rnd_api.o -I/opt/homebrew/include
ar rcs utils/libutils.a utils/rnd_api.o

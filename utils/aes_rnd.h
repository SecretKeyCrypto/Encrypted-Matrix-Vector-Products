// Based on code by Angshuman Karmakar
// https://github.com/Angshumank/const_gauss_split

#ifndef _UTIL_AES_RND_H_
#define _UTIL_AES_RND_H_

#ifndef _WIN32
	#include <unistd.h>
	#include <fcntl.h>
#endif
#include <stdlib.h>     //for rand_r, rand_s (WIN32)
#include <stdint.h>     //for int8_t
#include <wmmintrin.h>  //for intrinsics for AES-NI

class AES_Random
{
private:
	static constexpr int aes_buf_size = 512;
private:
	uint8_t aes_buf[aes_buf_size];
	int32_t aes_buf_pointer;
#ifdef _WIN32
	int64_t ctr[2] = {0};
#else
	__extension__ __int128 ctr = 0;
#endif
	__m128i key_schedule[20] = {0};
public:
	AES_Random()
	{
	}
public:
	bool reseed(const uint8_t seed[16])
	{
		load_key(seed);
#ifdef _WIN32
		ctr[0] = ctr[1] = 0;
#else
		ctr ^= ctr;
#endif
		aes_buf_pointer = aes_buf_size;
		return true;
	}
	bool reseed(unsigned int seed0)
	{
		uint8_t seed[16];
		int *pseed = (int *)seed;
		for (size_t i=0; i<sizeof(seed)/sizeof(int); i++) {
#ifdef _WIN32
			pseed[i] = rand();
#else
			pseed[i] = rand_r(&seed0);
#endif
		}
		return reseed(seed);
	}
	bool reseed()
	{
		uint8_t seed[16];
#ifdef _WIN32
		unsigned int *pseed = (unsigned int *)seed;
		for (int i=0; i<sizeof(seed)/sizeof(*pseed); i++) {
			if (0 != rand_s(pseed + i)) {
				return false;
			}
		}
#else
		int randomData = open("/dev/urandom", O_RDONLY);
		if(read(randomData, seed, sizeof(seed)==-1)) {
			return false;
		}
#endif
		return reseed(seed);
	}
private:
	inline __m128i key_expansion(__m128i key, __m128i keygened) {
	    keygened = _mm_shuffle_epi32(keygened, _MM_SHUFFLE(3,3,3,3));
	    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
	    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
	    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
	    return _mm_xor_si128(key, keygened);
	}
public:	

#define AES_128_key_exp(k, rcon) key_expansion(k, _mm_aeskeygenassist_si128(k, rcon))

	inline void load_key_enc_only(const uint8_t *enc_key){
	    key_schedule[0] = _mm_loadu_si128((const __m128i*) enc_key);
	    key_schedule[1]  = AES_128_key_exp(key_schedule[0], 0x01);
	    key_schedule[2]  = AES_128_key_exp(key_schedule[1], 0x02);
	    key_schedule[3]  = AES_128_key_exp(key_schedule[2], 0x04);
	    key_schedule[4]  = AES_128_key_exp(key_schedule[3], 0x08);
	    key_schedule[5]  = AES_128_key_exp(key_schedule[4], 0x10);
	    key_schedule[6]  = AES_128_key_exp(key_schedule[5], 0x20);
	    key_schedule[7]  = AES_128_key_exp(key_schedule[6], 0x40);
	    key_schedule[8]  = AES_128_key_exp(key_schedule[7], 0x80);
	    key_schedule[9]  = AES_128_key_exp(key_schedule[8], 0x1B);
	    key_schedule[10] = AES_128_key_exp(key_schedule[9], 0x36);
	}

#undef AES_128_key_exp
	
	inline void load_key(const uint8_t *enc_key){
	    load_key_enc_only(enc_key);
	
	    // generate decryption keys in reverse order.
	    // k[10] is shared by last encryption and first decryption rounds
	    // k[0] is shared by first encryption round and last decryption round (and is the original user key)
	    // For some implementation reasons, decryption key schedule is NOT the encryption key schedule in reverse order
	    key_schedule[11] = _mm_aesimc_si128(key_schedule[9]);
	    key_schedule[12] = _mm_aesimc_si128(key_schedule[8]);
	    key_schedule[13] = _mm_aesimc_si128(key_schedule[7]);
	    key_schedule[14] = _mm_aesimc_si128(key_schedule[6]);
	    key_schedule[15] = _mm_aesimc_si128(key_schedule[5]);
	    key_schedule[16] = _mm_aesimc_si128(key_schedule[4]);
	    key_schedule[17] = _mm_aesimc_si128(key_schedule[3]);
	    key_schedule[18] = _mm_aesimc_si128(key_schedule[2]);
	    key_schedule[19] = _mm_aesimc_si128(key_schedule[1]);
	}

#define AES_128_ENC_BLOCK(m,k) \
    do{\
	m = _mm_xor_si128       (m, k[ 0]); \
	m = _mm_aesenc_si128    (m, k[ 1]); \
	m = _mm_aesenc_si128    (m, k[ 2]); \
	m = _mm_aesenc_si128    (m, k[ 3]); \
	m = _mm_aesenc_si128    (m, k[ 4]); \
	m = _mm_aesenc_si128    (m, k[ 5]); \
	m = _mm_aesenc_si128    (m, k[ 6]); \
	m = _mm_aesenc_si128    (m, k[ 7]); \
	m = _mm_aesenc_si128    (m, k[ 8]); \
	m = _mm_aesenc_si128    (m, k[ 9]); \
	m = _mm_aesenclast_si128(m, k[10]);\
    }while(0)

	inline __m128i enc(const uint8_t *plainText){
	    __m128i m = _mm_loadu_si128((__m128i *) plainText);
	
	    AES_128_ENC_BLOCK(m,key_schedule);

	    return m;
	}

	inline void enc(const uint8_t *plainText, uint8_t *cipherText){
	    _mm_storeu_si128((__m128i *) cipherText, enc(plainText));
	}

	inline void xor_enc(const uint8_t *plainText ,uint8_t *cipherText){
	    __m128i m = _mm_loadu_si128((__m128i *) plainText);

	    m = _mm_xor_si128(m, enc(plainText));

	    _mm_storeu_si128((__m128i *) cipherText, m);
	}

#undef AES_128_ENC_BLOCK

#define AES_128_DEC_BLOCK(m,k) \
    do{\
	m = _mm_xor_si128       (m, k[10+0]); \
	m = _mm_aesdec_si128    (m, k[10+1]); \
	m = _mm_aesdec_si128    (m, k[10+2]); \
	m = _mm_aesdec_si128    (m, k[10+3]); \
	m = _mm_aesdec_si128    (m, k[10+4]); \
	m = _mm_aesdec_si128    (m, k[10+5]); \
	m = _mm_aesdec_si128    (m, k[10+6]); \
	m = _mm_aesdec_si128    (m, k[10+7]); \
	m = _mm_aesdec_si128    (m, k[10+8]); \
	m = _mm_aesdec_si128    (m, k[10+9]); \
	m = _mm_aesdeclast_si128(m, k[0]);\
    }while(0)

	inline void dec(const uint8_t *cipherText,uint8_t *plainText){
	    __m128i m = _mm_loadu_si128((__m128i *) cipherText);
	
	    AES_128_DEC_BLOCK(m,key_schedule);
	
	    _mm_storeu_si128((__m128i *) plainText, m);
	}

#undef AES_128_DEC_BLOCK
	
	inline void random_bytes(uint8_t *data) {
#ifdef _WIN32
	    ++ctr[0];
	    ctr[1] += !ctr[0];
	    enc((uint8_t *)ctr, data);
#else
	    ++ctr;
	    enc((uint8_t *)&ctr, data);
#endif
	}
	
	inline void xor_random_bytes(uint8_t *data) {
#ifdef _WIN32
	    ++ctr[0];
	    ctr[1] += !ctr[0];
	    xor_enc((uint8_t *)ctr, data);
#else
	    ++ctr;
	    xor_enc((uint8_t *)&ctr, data);
#endif
	}
};

#endif // _UTIL_AES_RND_H_

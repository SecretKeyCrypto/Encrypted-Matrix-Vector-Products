// Based on code by Angshuman Karmakar
// https://github.com/Angshumank/const_gauss_split

#ifndef _UTIL_AES_RND_H_
#define _UTIL_AES_RND_H_

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

#ifndef _WIN32
#include <unistd.h>   // for read, close
#include <fcntl.h>    // for O_RDONLY
#endif

// =====================================================
// Detect architecture
// =====================================================
#if defined(__x86_64__) || defined(_M_X64)

// ============================
// Intel / x86-64 implementation
// ============================
#include <wmmintrin.h>
using aes_block = __m128i;

inline aes_block key_expansion(aes_block key, aes_block keygened) {
    keygened = _mm_shuffle_epi32(keygened, _MM_SHUFFLE(3,3,3,3));
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    return _mm_xor_si128(key, keygened);
}

#define AES_128_key_exp(k, rcon) key_expansion(k, _mm_aeskeygenassist_si128(k, rcon))

inline void load_key_enc_only(const uint8_t *enc_key, aes_block* key_schedule) {
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

#elif defined(__aarch64__)

// ============================
// ARM / Apple Silicon NEON implementation
// ============================
#include <arm_neon.h>
using aes_block = uint8x16_t;

// AES S-box (must be fully defined elsewhere!)
extern const uint8_t sbox[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};

static inline aes_block aes_keygenassist(aes_block key, uint8_t rcon) {
    uint8_t temp[16];
    vst1q_u8(temp, key);
    uint8_t t[4] = { temp[13], temp[14], temp[15], temp[12] };
    for (int i = 0; i < 4; ++i) {
        t[i] = sbox[t[i]];
    }
    t[0] ^= rcon;
    uint8_t result[16] = { t[0], t[1], t[2], t[3], 0,0,0,0,0,0,0,0,0,0,0,0 };
    return vld1q_u8(result);
}

static inline aes_block key_expansion(aes_block key, aes_block keygened) {
    uint8_t k[16], kg[16];
    vst1q_u8(k, key);
    vst1q_u8(kg, keygened);
    for (int i = 0; i < 4; ++i) k[i] ^= kg[i];
    for (int i = 4; i < 16; ++i) k[i] ^= k[i - 4];
    return vld1q_u8(k);
}

#define AES_128_key_exp(k, rcon) key_expansion(k, aes_keygenassist(k, rcon))

inline void load_key_enc_only(const uint8_t *enc_key, aes_block* key_schedule) {
    key_schedule[0] = vld1q_u8(enc_key);
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

inline aes_block enc(const uint8_t *plainText, aes_block* roundKeys) {
    aes_block m = vld1q_u8(plainText);

    m = veorq_u8(m, roundKeys[0]);

    for (int i = 1; i < 10; i++) {
        m = vaeseq_u8(m, roundKeys[i]);  // AES SubBytes + ShiftRows + AddRoundKey
        m = vaesmcq_u8(m);               // AES MixColumns
    }

    m = vaeseq_u8(m, roundKeys[10]);
    m = veorq_u8(m, roundKeys[10]);      // Final AddRoundKey

    return m;
}

#else
#error "Unsupported architecture: only x86-64 and aarch64 are implemented"
#endif


// =====================================================
// AES_Random class
// =====================================================
class AES_Random {
private:
    static constexpr int aes_buf_size = 512;
    uint8_t aes_buf[aes_buf_size];
    int32_t aes_buf_pointer;

#ifdef _WIN32
    int64_t ctr[2] = {0};
#else
    __extension__ __int128 ctr = 0;
#endif

    aes_block key_schedule[20] = {0};

public:
    AES_Random() {}

    bool reseed(const uint8_t seed[16]) {
        load_key_enc_only(seed, key_schedule);
#ifdef _WIN32
        ctr[0] = ctr[1] = 0;
#else
        ctr = 0;
#endif
        aes_buf_pointer = aes_buf_size;
        return true;
    }

    bool reseed(unsigned int seed0) {
        uint8_t seed[16];
        int *pseed = (int *)seed;
        for (size_t i=0; i<sizeof(seed)/sizeof(int); i++) {
#ifdef _WIN32
            pseed[i] = rand();
#else
    #ifdef __APPLE__
            pseed[i] = arc4random();
    #else
            pseed[i] = rand_r(&seed0);
    #endif
#endif
        }
        return reseed(seed);
    }

    bool reseed() {
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
        if (read(randomData, seed, sizeof(seed)) == -1) {
            return false;
        }
        close(randomData);
#endif
        return reseed(seed);
    }

    inline void seek(int64_t pos_low, int64_t pos_high = 0) {
#ifdef _WIN32
        ctr[0] = pos_low;
        ctr[1] = pos_high;
#else
        int64_t* p_ctr = reinterpret_cast<int64_t*>(&ctr);
        p_ctr[0] = pos_low;
        p_ctr[1] = pos_high;
#endif
    }

    // ✅ random_bytes now uses enc(), no aes_encrypt_block
    inline void random_bytes(uint8_t *data) {
#ifdef _WIN32
        ++ctr[0];
        ctr[1] += !ctr[0];
        aes_block m = enc((uint8_t *)ctr, key_schedule);
#else
        ++ctr;
        aes_block m = enc((uint8_t *)&ctr, key_schedule);
#endif

#if defined(__x86_64__) || defined(_M_X64)
        _mm_storeu_si128((__m128i*)data, m);
#elif defined(__aarch64__)
        vst1q_u8(data, m);
#endif
    }

    // Intel enc()
#if defined(__x86_64__) || defined(_M_X64)
    inline aes_block enc(const uint8_t *plainText, aes_block* ks = nullptr) {
        aes_block* sched = ks ? ks : key_schedule;
        __m128i m = _mm_loadu_si128((__m128i *) plainText);
        m = _mm_xor_si128       (m, sched[0]);
        m = _mm_aesenc_si128    (m, sched[1]);
        m = _mm_aesenc_si128    (m, sched[2]);
        m = _mm_aesenc_si128    (m, sched[3]);
        m = _mm_aesenc_si128    (m, sched[4]);
        m = _mm_aesenc_si128    (m, sched[5]);
        m = _mm_aesenc_si128    (m, sched[6]);
        m = _mm_aesenc_si128    (m, sched[7]);
        m = _mm_aesenc_si128    (m, sched[8]);
        m = _mm_aesenc_si128    (m, sched[9]);
        m = _mm_aesenclast_si128(m, sched[10]);
        return m;
    }
#endif
};

#endif // _UTIL_AES_RND_H_
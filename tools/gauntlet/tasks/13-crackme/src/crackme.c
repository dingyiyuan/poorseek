/* crackme - the binary for gauntlet task 13. NOT shipped to the model (only the stripped binary is). */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
static const unsigned char P[32] = {2, 19, 14, 25, 27, 16, 3, 13, 10, 23, 22, 6, 28, 15, 26, 20, 0, 24, 11, 30, 9, 1, 18, 29, 21, 8, 4, 17, 7, 31, 5, 12};
static uint64_t rotl(uint64_t x, unsigned n) { n &= 63; return n ? (x << n) | (x >> (64 - n)) : x; }
static int hexv(int c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}
int main(int argc, char **argv) {
    if (argc != 2 || strlen(argv[1]) != 32) { puts("usage: crackme <licence key: 32 hex digits>"); return 2; }
    unsigned char k[16] = {0};
    for (int j = 0; j < 32; j++) {
        int v = hexv((unsigned char)argv[1][P[j]]);
        if (v < 0) { puts("FORMAT"); return 2; }
        k[j / 2] |= (unsigned char)(j % 2 ? v : v << 4);
    }
    uint64_t L = 0, R = 0;
    for (int i = 0; i < 8; i++) { L = L << 8 | k[i]; R = R << 8 | k[8 + i]; }
    uint64_t K = 0x243F6A8885A308D3ULL;
    for (unsigned rd = 0; rd < 8; rd++) {
        uint64_t f = rotl(R * 0xD1B54A32D192ED03ULL + K, rd * 7 + 5) ^ (R >> 31);
        uint64_t t = L ^ f; L = R; R = t;
        K = K * 6364136223846793005ULL + 1442695040888963407ULL;
    }
    if (L == 0x1702c6048167cc2eULL && R == 0x8f7c81228ff40a62ULL) { puts("ACCEPTED"); return 0; }
    puts("REJECTED");
    return 1;
}

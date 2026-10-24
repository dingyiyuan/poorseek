/* legacy_codec - the black box for gauntlet task 12. NOT shipped to the model (only the stripped binary is). */
#include <stdio.h>
#include <string.h>
static const char A[] = "QVJDMNFT2RL6XAW8E37ZGPSUYCH5K49B";
static unsigned char rl[600], mx[600];
int main(void) {
    unsigned char in[300];
    size_t n = fread(in, 1, sizeof in, stdin);
    if (n > 255) { fputs("error: input longer than 255 bytes\n", stderr); return 2; }
    size_t m = 0, i = 0;
    while (i < n) {
        unsigned char x = in[i]; size_t r = 1;
        while (i + r < n && in[i + r] == x && r < 130) r++;
        if (r >= 3) { rl[m++] = 0xFF; rl[m++] = (unsigned char)(r - 3); rl[m++] = x; i += r; }
        else { rl[m++] = x; if (x == 0xFF) rl[m++] = 0x80; i++; }
    }
    unsigned s = 0x5A;
    for (size_t k = 0; k < m; k++) {
        unsigned v = (rl[k] ^ s) & 0xFF, sh = k % 5;
        unsigned o = ((v << sh) | (v >> (8 - sh))) & 0xFF;
        mx[k] = (unsigned char)o;
        s = o == 0 ? 0x5A : (s * 31 + o + 17) & 0xFF;
    }
    unsigned crc = 0xFFFF;
    for (size_t k = 0; k < m; k++) {
        crc ^= (unsigned)mx[k] << 8;
        for (int b = 0; b < 8; b++) crc = (crc & 0x8000) ? ((crc << 1) ^ 0x3D65) & 0xFFFF : (crc << 1) & 0xFFFF;
    }
    mx[m++] = crc >> 8; mx[m++] = crc & 0xFF;
    char body[1200]; size_t bl = 0; unsigned acc = 0; int bits = 0;
    for (size_t k = 0; k < m; k++) {
        acc = (acc << 8) | mx[k]; bits += 8;
        while (bits >= 5) { body[bl++] = A[(acc >> (bits - 5)) & 31]; bits -= 5; }
    }
    if (bits) body[bl++] = A[(acc << (5 - bits)) & 31];
    putchar(A[n % 32]);
    for (size_t k = 0; k < bl; k++) { if (k % 4 == 0) putchar('-'); putchar(body[k]); }
    putchar('\n');
    return 0;
}

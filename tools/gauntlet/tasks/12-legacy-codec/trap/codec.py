"""THE TRAP: everything a few dozen ordinary queries reveal - but the two rare paths are missed: a lone 0xFF is
not escaped, and the chaining state never resets when a mixed byte comes out 0."""
A = "QVJDMNFT2RL6XAW8E37ZGPSUYCH5K49B"


def encode(data):
    data = bytes(data)
    if len(data) > 255:
        raise ValueError("input longer than 255 bytes")
    rl, i = [], 0
    while i < len(data):
        x, r = data[i], 1
        while i + r < len(data) and data[i + r] == x and r < 130:
            r += 1
        if r >= 3:
            rl += [0xFF, r - 3, x]
            i += r
        else:
            rl += [x]
            i += 1
    mx, s = [], 0x5A
    for k, c in enumerate(rl):
        v, sh = (c ^ s) & 0xFF, k % 5
        o = ((v << sh) | (v >> (8 - sh))) & 0xFF
        mx.append(o)
        s = (s * 31 + o + 17) & 0xFF
    crc = 0xFFFF
    for b in mx:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x3D65) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    mx += [crc >> 8, crc & 0xFF]
    body, acc, bits = [], 0, 0
    for b in mx:
        acc, bits = (acc << 8) | b, bits + 8
        while bits >= 5:
            body.append(A[(acc >> (bits - 5)) & 31])
            bits -= 5
    if bits:
        body.append(A[(acc << (5 - bits)) & 31])
    return A[len(data) % 32] + "".join("-" + "".join(body[j:j + 4]) for j in range(0, len(body), 4))

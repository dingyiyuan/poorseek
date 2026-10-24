"""KX-8 emulator (reference). See SPEC.md."""

M = 0xFFFF


def _s(v):
    return v - 0x10000 if v & 0x8000 else v


def run(image, max_steps=1_000_000):
    mem = bytearray(65536)
    mem[:len(image)] = image
    r = [0] * 8
    sp = pc = 0
    N = Z = C = V = 0
    out = []
    steps = 0
    status = "LIMIT"

    def rw(a):
        return mem[a & M] << 8 | mem[(a + 1) & M]

    def ww(a, v):
        mem[a & M] = (v >> 8) & 0xFF
        mem[(a + 1) & M] = v & 0xFF

    def put(i, v):
        if i:
            r[i] = v & M

    def zn(v):
        return int(v & M == 0), (v >> 15) & 1

    while steps < max_steps:
        at = pc
        w = rw(pc)
        op, d, s, k = w >> 10, (w >> 7) & 7, (w >> 4) & 7, w & 15
        nxt = (pc + 2) & M
        imm = 0
        if op in (0x02, 0x03, 0x04, 0x05, 0x06, 0x0D, 0x0E) or 0x20 <= op <= 0x28 or op == 0x2A:
            imm = rw(nxt)
            nxt = (nxt + 2) & M
        a, b = r[d], r[s]
        if op == 0x00:
            pc, steps, status = nxt, steps + 1, "HALT"
            break
        elif op == 0x01:
            put(d, b)
        elif op == 0x02:
            put(d, imm)
        elif op == 0x03:
            put(d, rw(b + imm))
        elif op == 0x04:
            ww(a + imm, b)
        elif op == 0x05:
            put(d, mem[(b + imm) & M])
        elif op == 0x06:
            mem[(a + imm) & M] = b & 0xFF
        elif op in (0x08, 0x0A, 0x0D):
            y = imm if op == 0x0D else b
            cin = C if op == 0x0A else 0
            t = a + y + cin
            res = t & M
            Z, N = zn(res)
            C = int(t > M)
            V = int(not -32768 <= _s(a) + _s(y) + cin <= 32767)
            put(d, res)
        elif op in (0x09, 0x0B, 0x0C, 0x0E):
            y = imm if op == 0x0E else b
            bin_ = (1 - C) if op == 0x0B else 0
            t = a - y - bin_
            res = t & M
            Z, N = zn(res)
            C = int(t >= 0)
            V = int(not -32768 <= _s(a) - _s(y) - bin_ <= 32767)
            if op in (0x09, 0x0B):
                put(d, res)
        elif op in (0x10, 0x11, 0x12, 0x13):
            res = {0x10: a & b, 0x11: a | b, 0x12: a ^ b, 0x13: ~a & M}[op]
            Z, N = zn(res)
            V = 0
            put(d, res)
        elif op in (0x14, 0x15, 0x16):
            res = a
            if k:
                if op == 0x14:
                    C = (a >> (16 - k)) & 1
                    res = (a << k) & M
                elif op == 0x15:
                    C = (a >> (k - 1)) & 1
                    res = a >> k
                else:
                    C = (a >> (k - 1)) & 1
                    res = (_s(a) >> k) & M
            Z, N = zn(res)
            V = 0
            put(d, res)
        elif op == 0x17:
            res = a
            for _ in range(k):
                res, C = ((res << 1) | C) & M, res >> 15
            Z, N = zn(res)
            V = 0
            put(d, res)
        elif op == 0x18:
            p = a * b
            Z, N = int(p == 0), (p >> 31) & 1
            C = V = int(p >> 16 != 0)
            put(d, p & M)
            if d < 7:
                put(d + 1, p >> 16)
        elif op == 0x19:
            if b == 0:
                status = "DIVZERO"
                break
            x, y = _s(a), _s(b)
            if x == -32768 and y == -1:
                q, rem, V = -32768, 0, 1
            else:
                q = abs(x) // abs(y)
                if (x < 0) != (y < 0):
                    q = -q
                rem, V = x - q * y, 0
            Z, N = zn(q & M)
            put(d, q & M)
            if d < 7:
                put(d + 1, rem & M)
        elif 0x20 <= op <= 0x28:
            take = {0x20: True, 0x21: Z, 0x22: not Z, 0x23: N != V, 0x24: N == V, 0x25: C == 0, 0x26: C == 1,
                    0x27: not Z and N == V, 0x28: Z or N != V}[op]
            if take:
                nxt = (nxt + imm) & M
        elif op == 0x29:
            nxt = b
        elif op == 0x2A:
            sp = (sp - 2) & M
            ww(sp, nxt)
            nxt = (nxt + imm) & M
        elif op == 0x2B:
            nxt = rw(sp)
            sp = (sp + 2) & M
        elif op == 0x2C:
            sp = (sp - 2) & M
            ww(sp, b)
        elif op == 0x2D:
            v = rw(sp)
            sp = (sp + 2) & M
            put(d, v)
        elif op == 0x2E:
            sp = (sp - 2) & M
            ww(sp, C << 3 | V << 2 | N << 1 | Z)
        elif op == 0x2F:
            v = rw(sp)
            sp = (sp + 2) & M
            C, V, N, Z = (v >> 3) & 1, (v >> 2) & 1, (v >> 1) & 1, v & 1
        elif op == 0x30:
            out.append(chr(b & 0xFF))
        elif op == 0x31:
            out.append(f"{_s(b)}\n")
        else:
            status = "ILLEGAL"
            break
        pc = nxt
        steps += 1
    if status in ("DIVZERO", "ILLEGAL"):
        pc = at
    flags = ("N" if N else "-") + ("Z" if Z else "-") + ("C" if C else "-") + ("V" if V else "-")
    return {"status": status, "pc": pc, "sp": sp, "regs": list(r), "flags": flags, "steps": steps,
            "output": "".join(out)}

"""kx8asm - a two-pass assembler for KX-8 (see SPEC.md). Provided and correct; you do not need to change it.

    assemble(text) -> bytes        (the image, loaded at address 0)

Syntax, one statement per line, ';' starts a comment:
    label:                        a label is the address of the next byte emitted
    ADD r1, r2                    register operands r0..r7
    LDI r1, 1234 | -5 | 0x1F | 'A' | label | label+2     immediates: decimal, hex, char, a label (its address), sums
    LD  r1, [r2+8]   LD r1, [r2-2]   LD r1, [r2]   LD r1, [r2+label]
    ST  r3, [r2+8]                value register first, then the base register (encodes s = r3, d = r2)
    SHL r1, 3                     shift / rotate count k (0-15)
    BNZ loop    CALL fn    B 0    a label operand becomes the relative offset (label - next); a number is the raw offset
    .word 1, 0xBEEF, label        big-endian words
    .byte 1, 2, 'x'               bytes
    .ascii "hi\\n"                 bytes of a string (\\n, \\t, \\\\, \\" escapes)
    .org 0x200                    move forward to an address (zero fill)
"""
import re

OPS = {  # name: (op, shape)
    "HLT": (0x00, ""), "MOV": (0x01, "ds"), "LDI": (0x02, "di"), "LD": (0x03, "dm"), "ST": (0x04, "sm"),
    "LDB": (0x05, "dm"), "STB": (0x06, "sm"),
    "ADD": (0x08, "ds"), "SUB": (0x09, "ds"), "ADC": (0x0A, "ds"), "SBC": (0x0B, "ds"), "CMP": (0x0C, "ds"),
    "ADDI": (0x0D, "di"), "CMPI": (0x0E, "di"),
    "AND": (0x10, "ds"), "OR": (0x11, "ds"), "XOR": (0x12, "ds"), "NOT": (0x13, "d"),
    "SHL": (0x14, "dk"), "SHR": (0x15, "dk"), "SAR": (0x16, "dk"), "ROL": (0x17, "dk"),
    "MUL": (0x18, "ds"), "DIV": (0x19, "ds"),
    "B": (0x20, "r"), "BZ": (0x21, "r"), "BNZ": (0x22, "r"), "BLT": (0x23, "r"), "BGE": (0x24, "r"),
    "BLO": (0x25, "r"), "BHS": (0x26, "r"), "BGT": (0x27, "r"), "BLE": (0x28, "r"),
    "JMP": (0x29, "s"), "CALL": (0x2A, "r"), "RET": (0x2B, ""), "PUSH": (0x2C, "s"), "POP": (0x2D, "d"),
    "PUSHF": (0x2E, ""), "POPF": (0x2F, ""), "OUT": (0x30, "s"), "OUTN": (0x31, "s"),
}
EXT = set("imr")  # shapes that take an extension word


def _split(args):
    out, cur, q = [], "", None
    for ch in args:
        if q:
            cur += ch
            if ch == q and not cur.endswith("\\" + q):
                q = None
        elif ch in "'\"":
            q, cur = ch, cur + ch
        elif ch == ",":
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def _unescape(s):
    return s.encode().decode("unicode_escape")


def _reg(t):
    m = re.fullmatch(r"[rR]([0-7])", t.strip())
    if not m:
        raise ValueError(f"bad register {t!r}")
    return int(m.group(1))


def _val(t, labels, strict):
    t = t.strip()
    parts = re.split(r"(?<=[\w'])\s*([-+])\s*", t)   # label+2, end-start, 0x10-1 (not a leading sign)
    if len(parts) > 1:
        v = _val(parts[0], labels, strict)
        for sign, term in zip(parts[1::2], parts[2::2]):
            v = v + _val(term, labels, strict) if sign == "+" else v - _val(term, labels, strict)
        return v
    if re.fullmatch(r"'(\\.|[^'])'", t):
        return ord(_unescape(t[1:-1]))
    if re.fullmatch(r"[-+]?(0[xX][0-9a-fA-F]+|\d+)", t):
        return int(t, 0)
    if re.fullmatch(r"[A-Za-z_.]\w*", t):
        if t in labels:
            return labels[t]
        if strict:
            raise ValueError(f"unknown label {t!r}")
        return 0
    raise ValueError(f"bad value {t!r}")


def _mem(t, labels, strict):
    m = re.fullmatch(r"\[\s*([rR][0-7])\s*(?:([-+])\s*(.+?))?\s*\]", t.strip())
    if not m:
        raise ValueError(f"bad memory operand {t!r}")
    off = _val(m.group(3), labels, strict) if m.group(3) else 0
    return _reg(m.group(1)), -off if m.group(2) == "-" else off


def _pass(text, labels, strict):
    out = bytearray()
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.split(";", 1)[0].strip() if '"' not in raw else re.sub(r';[^"]*$', "", raw).strip()
        while True:
            m = re.match(r"([A-Za-z_.]\w*)\s*:\s*(.*)$", line)
            if not m:
                break
            labels[m.group(1)] = len(out) if not strict else labels[m.group(1)]
            line = m.group(2)
        if not line:
            continue
        name, _, args = line.partition(" ")
        args = _split(args)
        try:
            if name == ".org":
                a = _val(args[0], labels, strict)
                if a < len(out):
                    raise ValueError(".org moves backwards")
                out += bytes(a - len(out))
                continue
            if name == ".word":
                for a in args:
                    out += (_val(a, labels, strict) & 0xFFFF).to_bytes(2, "big")
                continue
            if name == ".byte":
                for a in args:
                    out.append(_val(a, labels, strict) & 0xFF)
                continue
            if name == ".ascii":
                out += _unescape(args[0].strip()[1:-1]).encode("latin-1")
                continue
            op, shape = OPS[name.upper()]
            d = s = k = 0
            ext = None
            if len(args) != len(shape):
                raise ValueError(f"{name} takes {len(shape)} operand(s)")
            for kind, a in zip(shape, args):
                if kind == "d":
                    d = _reg(a)
                elif kind == "s":
                    s = _reg(a)
                elif kind == "k":
                    k = _val(a, labels, strict)
                    if not 0 <= k <= 15:
                        raise ValueError("count must be 0-15")
                elif kind == "i":
                    ext = _val(a, labels, strict)
                elif kind == "m":
                    base, ext = _mem(a, labels, strict)
                    if shape[0] == "d":
                        s = base          # LD/LDB rd, [rs+imm]
                    else:
                        d = base          # ST/STB rs, [rd+imm]
                elif kind == "r":
                    nxt = len(out) + 4
                    if re.fullmatch(r"[A-Za-z_.]\w*", a.strip()):
                        ext = _val(a, labels, strict) - nxt
                    else:
                        ext = _val(a, labels, strict)
            out += (op << 10 | d << 7 | s << 4 | k).to_bytes(2, "big")
            if any(c in EXT for c in shape):
                out += (ext & 0xFFFF).to_bytes(2, "big")
        except (KeyError, IndexError, ValueError) as e:
            raise ValueError(f"line {n}: {raw.strip()!r}: {e}") from None
    return bytes(out)


def assemble(text):
    labels = {}
    _pass(text, labels, strict=False)
    return _pass(text, labels, strict=True)

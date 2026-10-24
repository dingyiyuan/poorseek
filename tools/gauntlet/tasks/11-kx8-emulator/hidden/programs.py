"""The hidden KX-8 programs: (name, assembly, max_steps). Expected results are frozen in expected.json."""

FLAGS = "PUSHF\nPOP r7\nOUTN r7\n"          # print the flags word (C=8 V=4 N=2 Z=1)


def _arith():
    rows = [("LDI r1, 5", "LDI r2, 3", "SUB r1, r2"), ("LDI r1, 3", "LDI r2, 5", "SUB r1, r2"),
            ("LDI r1, 0x8000", "", "CMPI r1, 1"), ("LDI r1, 0x7FFF", "", "ADDI r1, 1"),
            ("LDI r1, 0xFFFF", "", "ADDI r1, 1"), ("", "", "AND r1, r1"), ("LDI r1, 7", "LDI r2, 7", "CMP r1, r2"),
            ("LDI r1, 0", "LDI r2, 1", "SUB r1, r2"), ("", "", "OR r1, r0"), ("LDI r1, 0x8000", "LDI r2, 0x8000", "ADD r1, r2"),
            ("LDI r1, 0x7FFF", "LDI r2, 0xFFFF", "SUB r1, r2"), ("LDI r1, 0x8000", "", "CMPI r1, 0x8000"),
            ("LDI r1, 0x1234", "", "NOT r1"), ("LDI r1, 0x00FF", "LDI r2, 0x0FF0", "XOR r1, r2")]
    return "".join(f"{a}\n{b}\n{c}\nOUTN r1\n{FLAGS}" for a, b, c in rows) + "HLT\n"


def _multiword():
    # (hi:lo) 32-bit add then subtract with carry chains; flags after each half
    return """
    LDI r1, 0xFFFF      ; a.lo
    LDI r2, 0x0001      ; a.hi
    LDI r3, 0x0001      ; b.lo
    LDI r4, 0x0002      ; b.hi
    ADD r1, r3
    """ + FLAGS + """
    ADC r2, r4
    """ + FLAGS + """
    OUTN r2
    OUTN r1
    LDI r1, 0x0000
    LDI r2, 0x0003
    LDI r3, 0x0001
    LDI r4, 0x0000
    SUB r1, r3          ; borrows -> C = 0
    """ + FLAGS + """
    SBC r2, r4          ; 3 - 0 - 1
    """ + FLAGS + """
    OUTN r2
    OUTN r1
    LDI r5, 10
    LDI r6, 3
    CMP r5, r6          ; no borrow -> C = 1
    SBC r5, r6          ; 10 - 3 - 0 = 7
    OUTN r5
    """ + FLAGS + """
    LDI r5, 0xFFFF
    LDI r6, 0
    CMPI r6, 1          ; 0 - 1 borrows -> C = 0
    ADC r5, r6          ; 0xFFFF + 0 + 0
    OUTN r5
    """ + FLAGS + """
    LDI r5, 0x7FFF
    LDI r6, 0
    CMPI r5, 0          ; C = 1
    ADC r5, r6          ; 0x7FFF + 0 + 1 -> signed overflow
    OUTN r5
    """ + FLAGS + """
    LDI r5, 0x8000
    LDI r6, 0
    CMPI r6, 1          ; C = 0
    SBC r5, r6          ; 0x8000 - 0 - 1 -> signed overflow, no borrow
    OUTN r5
    """ + FLAGS + """
    LDI r5, 0
    LDI r6, 0xFFFF
    CMPI r6, 1          ; C = 1
    SBC r5, r6          ; 0 - 0xFFFF - 0 -> borrows
    OUTN r5
    """ + FLAGS + "HLT\n"


def _div():
    pairs = [(7, 2), (-7, 2), (7, -2), (-7, -2), (-32768, -1), (1, -32768), (-1, 3), (-32768, 2), (32767, -1),
             (0, -5), (-9, 9), (100, 7)]
    s = ""
    for x, y in pairs:
        s += f"LDI r2, {x}\nLDI r4, {y}\nDIV r2, r4\nOUTN r2\nOUTN r3\n{FLAGS}"
    s += "LDI r7, -50\nLDI r4, 7\nDIV r7, r4\nOUTN r7\nOUTN r6\n"       # d = 7: remainder discarded
    s += "LDI r1, 99\nLDI r4, 7\nDIV r0, r4\nOUTN r0\nOUTN r1\n"          # d = 0: quotient discarded, r1 = 0 % 7
    return s + "HLT\n"


def _divzero():
    return """
    LDI r1, 1234
    LDI r2, 0
    LDI r5, 0xFFFF
    ADDI r5, 1          ; Z C set
    PUSH r1
    DIV r1, r2
    OUTN r1
    HLT
    """


def _branches():
    pairs = [(1, 2), (2, 1), (0xFFFF, 1), (1, 0xFFFF), (0x8000, 0x7FFF), (0x7FFF, 0x8000), (5, 5), (0, 0x8000)]
    s, n = "", 0
    for a, b in pairs:
        for br in ("BLT", "BGE", "BLO", "BHS", "BGT", "BLE", "BZ", "BNZ"):
            s += (f"LDI r1, {a}\nLDI r2, {b}\nLDI r5, 0\nCMP r1, r2\n{br} t{n}\nB j{n}\n"
                  f"t{n}: LDI r5, 1\nj{n}: OUTN r5\n")
            n += 1
    return s + "HLT\n"


def _shifts():
    s = ""
    for op, v, k, pre in [("SHL", 0x8001, 1, ""), ("SHL", 0x4000, 2, ""), ("SHL", 0x1234, 4, ""), ("SHR", 0x8001, 1, ""),
                          ("SHR", 0x8000, 15, ""), ("SAR", 0x8000, 15, ""), ("SAR", 0x8001, 1, ""),
                          ("SAR", 0x4002, 2, ""), ("SHL", 0x0000, 0, "CMPI r1, 1"), ("SHR", 0x8000, 0, "CMPI r6, 1"),
                          ("ROL", 0x8000, 1, "CMPI r6, 1"), ("ROL", 0x8000, 1, "CMPI r1, 0"), ("ROL", 0x0001, 15, "CMPI r1, 0"),
                          ("ROL", 0x1234, 0, "CMPI r6, 1"), ("ROL", 0xFFFF, 3, "CMPI r6, 1"), ("SHL", 0xFFFF, 15, "")]:
        s += f"LDI r6, 0\n{pre}\nLDI r1, {v}\n{op} r1, {k}\nOUTN r1\n{FLAGS}"
    return s + "HLT\n"


def _mul():
    return ("LDI r2, 300\nLDI r4, 300\nMUL r2, r4\nOUTN r2\nOUTN r3\n" + FLAGS +
            "LDI r2, 0xFFFF\nLDI r4, 0xFFFF\nMUL r2, r4\nOUTN r2\nOUTN r3\n" + FLAGS +
            "LDI r2, -1\nLDI r4, 2\nMUL r2, r4\nOUTN r2\nOUTN r3\n" + FLAGS +
            "LDI r7, 1000\nLDI r4, 1000\nLDI r1, 77\nMUL r7, r4\nOUTN r7\nOUTN r1\n" + FLAGS +
            "LDI r1, 5\nLDI r4, 0x9000\nLDI r6, 2\nMUL r0, r6\nOUTN r1\n" + FLAGS +
            "LDI r3, 0x4000\nMUL r3, r3\nOUTN r3\nOUTN r4\n" + FLAGS +
            "LDI r2, 0\nLDI r4, 9\nMUL r2, r4\n" + FLAGS + "HLT\n")


def _fib():
    return """
    LDI r1, 17
    CALL fib
    OUTN r2
    HLT
fib:                    ; r2 = fib(r1), clobbers r3; preserves r1
    CMPI r1, 2
    BLO base
    PUSH r1
    ADDI r1, -1
    CALL fib
    PUSH r2
    ADDI r1, -1
    CALL fib
    POP r3
    ADD r2, r3
    POP r1
    RET
base:
    MOV r2, r1
    RET
    """


def _selfmod():
    return """
    LDI r4, 5
loop:
patch: LDI r1, 100      ; the immediate of this LDI is rewritten each pass
    OUTN r1
    LD r2, [r0+patch+2]
    ADDI r2, 7
    ST r2, [r0+patch+2]
    ADDI r4, -1
    BNZ loop
    LDI r2, 0x0000      ; HLT word
    ST r2, [r0+gone]
    LDI r1, 1
gone: OUTN r1           ; replaced by HLT before it runs
    OUTN r1
    HLT
    """


def _memory():
    return """
    LDI r1, 0x1234
    LDI r2, buf
    ST r1, [r2+0]
    LDB r3, [r2+0]
    OUTN r3
    LDB r3, [r2+1]
    OUTN r3
    LDI r1, 0xABCD
    STB r1, [r2+3]
    LD r3, [r2+2]
    OUTN r3
    LDI r5, 0xFFFF
    LDI r1, 0x5A5A
    ST r1, [r5+0]       ; high byte -> 0xFFFF, low byte -> 0x0000 (this program's first byte)
    LDB r3, [r5+0]
    OUTN r3
    LDB r3, [r0+0]
    OUTN r3
    LD r3, [r5+0]
    OUTN r3
    LDI r6, 0x0010
    LD r3, [r6-16]      ; base + negative offset = address 0
    OUTN r3
    LDI r6, buf
    LD r3, [r6+0xFFFE]  ; wraps to buf-2
    OUTN r3
    HLT
    .word 0x7777
buf: .word 0, 0
    """


def _illegal():
    return """
    LDI r1, 3
    OUT r1
    ADDI r1, -3
    .word 0x1C00        ; op 0x07 is not defined
    HLT
    """


def _illegal_high():
    return "LDI r2, 66\nOUT r2\n.word 0xFFFF\nHLT\n"


def _limit():
    return "LDI r1, 0\nspin: ADDI r1, 1\nB spin\n"


def _r0():
    return ("LDI r0, 5\nOUTN r0\nLDI r1, 9\nPUSH r1\nPOP r0\nOUTN r0\nMOV r0, r1\nOUTN r0\nADD r0, r1\nOUTN r0\n" + FLAGS +
            "LDI r1, 0\nSUB r0, r1\n" + FLAGS + "CMPI r0, 1\n" + FLAGS + "ADDI r0, 0x8000\n" + FLAGS + "HLT\n")


def _jumptable():
    return """
    LDI r1, 0
next:
    LD r2, [r1+table]
    JMP r2
a:  LDI r3, 'a'
    B done
b:  LDI r3, 'b'
    B done
c:  LDI r3, 'c'
done:
    OUT r3
    ADDI r1, 2
    CMPI r1, 8
    BLO next
    LDI r3, end
    JMP r3
    OUTN r3
end: HLT
table: .word c, a, b, c
    """


def _output():
    return ("LDI r1, 0x141\nOUT r1\nLDI r1, 0x0A\nOUT r1\nLDI r1, -1\nOUTN r1\nLDI r1, 0x8000\nOUTN r1\n"
            "LDI r1, 32767\nOUTN r1\nLDI r1, 0x2D\nOUT r1\nHLT\n")


def _popf():
    return ("LDI r1, 0xFFF5\nPUSH r1\nPOPF\nBZ z1\nOUTN r0\nz1: BLT l1\nLDI r2, 7\nOUTN r2\nl1: BLO lo1\nLDI r2, 8\nOUTN r2\n"
            "lo1: LDI r1, 0x0008\nPUSH r1\nPOPF\nBHS hs\nOUTN r0\nhs: BGT gt\nOUTN r0\ngt: LDI r2, 1\nOUTN r2\n"
            "LDI r1, 0x0006\nPUSH r1\nPOPF\nBGE ge\nLDI r2, 2\nOUTN r2\nge: HLT\n")


def _sieve():
    # primes below 4000: count, sum (mod 65536) and the largest
    return """
    LDI r1, 2           ; i
outer:
    CMPI r1, 64         ; i*i < 4000 needs i < 64
    BHS scan
    LDB r2, [r1+flags]
    CMPI r2, 0
    BNZ skip
    MOV r3, r1
    MOV r4, r1
    MUL r3, r4          ; r3 = i*i (r4 = high word = 0)
mark:
    CMPI r3, 4000
    BHS skip
    LDI r5, 1
    STB r5, [r3+flags]
    ADD r3, r1
    B mark
skip:
    ADDI r1, 1
    B outer
scan:
    LDI r1, 2
    LDI r6, 0           ; count
    LDI r7, 0           ; sum
loop:
    CMPI r1, 4000
    BHS out
    LDB r2, [r1+flags]
    CMPI r2, 0
    BNZ nope
    ADDI r6, 1
    ADD r7, r1
    MOV r5, r1
nope:
    ADDI r1, 1
    B loop
out:
    OUTN r6
    OUTN r7
    OUTN r5
    HLT
flags: .byte 0
    """


def _stackwrap():
    return ("LDI r1, 0x1111\nPUSH r1\nLD r2, [r0+0xFFFE]\nOUTN r2\nPOP r3\nPOP r4\nOUTN r4\n"
            "LDI r5, 0xAC00\nST r5, [r0+0xFFFC]\nLDI r5, 0xFFFC\nCALL sub\nOUTN r5\nHLT\nsub: JMP r5\n")


def _collatz():
    # the longest Collatz chain for a start below 600, with 32-bit-safe steps tracked in 16 bits (values stay < 65536)
    return """
    LDI r1, 1           ; n
    LDI r6, 0           ; best length
    LDI r7, 0           ; best n
each:
    CMPI r1, 600
    BHS fin
    MOV r2, r1
    LDI r3, 0
step:
    CMPI r2, 1
    BZ done
    MOV r4, r2
    LDI r5, 1
    AND r4, r5
    BZ even
    MOV r4, r2
    SHL r4, 1
    ADD r2, r4
    ADDI r2, 1
    BHS ovf             ; carry out of the ADDI -> the value no longer fits; give up on this start
    ADDI r3, 1
    B step
even:
    SHR r2, 1
    ADDI r3, 1
    B step
ovf:
    LDI r3, 0
done:
    CMP r6, r3
    BHS keep
    MOV r6, r3
    MOV r7, r1
keep:
    ADDI r1, 1
    B each
fin:
    OUTN r7
    OUTN r6
    HLT
    """


PROGRAMS = [
    ("arith_flags", _arith(), 1_000_000),
    ("multiword_carry", _multiword(), 1_000_000),
    ("div_semantics", _div(), 1_000_000),
    ("divzero_fault", _divzero(), 1_000_000),
    ("branch_matrix", _branches(), 1_000_000),
    ("shifts_rotates", _shifts(), 1_000_000),
    ("mul_words", _mul(), 1_000_000),
    ("recursive_fib", _fib(), 1_000_000),
    ("self_modifying", _selfmod(), 1_000_000),
    ("memory_endian_wrap", _memory(), 1_000_000),
    ("illegal_op", _illegal(), 1_000_000),
    ("illegal_high_op", _illegal_high(), 1_000_000),
    ("step_limit", _limit(), 1001),
    ("step_limit_zero", _limit(), 0),
    ("r0_hardwired", _r0(), 1_000_000),
    ("jump_table", _jumptable(), 1_000_000),
    ("output_format", _output(), 1_000_000),
    ("popf_branches", _popf(), 1_000_000),
    ("sieve_4000", _sieve(), 2_000_000),
    ("stack_wrap", _stackwrap(), 1_000_000),
    ("collatz_600", _collatz(), 2_000_000),
]

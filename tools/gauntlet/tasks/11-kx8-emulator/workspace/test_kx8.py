from kx8 import run
from kx8asm import assemble


def test_hello():
    r = run(assemble("""
        LDI r1, msg
    next:
        LDB r2, [r1+0]
        CMPI r2, 0
        BZ done
        OUT r2
        ADDI r1, 1
        B next
    done:
        HLT
    msg: .ascii "hi\\n"
        .byte 0
    """))
    assert r["status"] == "HALT" and r["output"] == "hi\n"


def test_countdown_sum():
    r = run(assemble("""
        LDI r1, 10
        LDI r2, 0
    loop:
        ADD r2, r1
        ADDI r1, -1
        BNZ loop
        OUTN r2
        HLT
    """))
    assert r["output"] == "55\n"
    assert r["regs"][2] == 55 and r["regs"][1] == 0

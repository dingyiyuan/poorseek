# KX-8 — machine specification

Implement `run(image, max_steps=1_000_000)` in `kx8.py`. Every rule below is tested. Several rules are deliberately
**unlike** x86, ARM, 6502 or Python - read each one; do not assume a convention.

## State

- `mem`: 65,536 bytes, all 0; `image` (bytes) is copied in starting at address 0. All address arithmetic is mod 65536.
- **Words are BIG-endian**: the word at `a` is `mem[a] << 8 | mem[(a+1) & 0xFFFF]`; storing a word writes the high
  byte to `a` and the low byte to `a+1`.
- Registers `r0`..`r7`, 16 bits each, all 0 at start. **`r0` always reads 0; every write to `r0` is discarded**
  (POP into `r0` still moves `sp`; MUL/DIV with `d` = 0 discard the low word/quotient but still write `r1`).
- `sp` (16 bits) starts at **0**. PUSH: `sp = (sp - 2) & 0xFFFF`, then store the word at `sp`. POP: load the word at
  `sp`, then `sp = (sp + 2) & 0xFFFF`. (So the first push writes to 0xFFFE.)
- `pc` starts at 0. Flags `N Z C V` all start clear.

## Fetch / execute

1. Read the instruction word at `pc` **from memory, every time** (programs modify their own code); `pc += 2`.
2. If the opcode takes an extension word `imm`, read it at the new `pc`; `pc += 2`.
3. `next` = the `pc` after steps 1-2. Relative targets are `next + imm` (imm as signed 16-bit), mod 65536.
4. Execute. Instruction word layout: bits 15-10 `op` (6 bits), 9-7 `d`, 6-4 `s`, 3-0 `k`. Fields an instruction does
   not use are ignored. `rd`/`rs` below mean registers `r<d>`/`r<s>`.

Signed views are two's complement 16-bit. "Z N" means: Z = (16-bit result == 0), N = bit 15 of the result.

## Instructions (op in hex; `+imm` = takes an extension word)

| op | name | effect | flags |
|----|------|--------|-------|
| 00 | HLT | stop: status `HALT` | - |
| 01 | MOV rd, rs | rd = rs | - |
| 02 | LDI rd, imm (+imm) | rd = imm | - |
| 03 | LD rd, [rs+imm] (+imm) | rd = word at (rs + imm) | - |
| 04 | ST rs, [rd+imm] (+imm) | word at (**rd** + imm) = **rs** (note: `d` is the base, `s` the value) | - |
| 05 | LDB rd, [rs+imm] (+imm) | rd = byte at (rs + imm), zero-extended | - |
| 06 | STB rs, [rd+imm] (+imm) | byte at (rd + imm) = low byte of rs | - |
| 08 | ADD rd, rs | rd = rd + rs | Z N; C = unsigned carry out of bit 15; V = signed overflow |
| 09 | SUB rd, rs | rd = rd - rs | Z N; **C = 1 when NO borrow** (rd >= rs unsigned), C = 0 when it borrows; V = signed overflow |
| 0A | ADC rd, rs | rd = rd + rs + C | as ADD (carry out of the full 3-operand sum) |
| 0B | SBC rd, rs | rd = rd - rs - (1 - C) | as SUB: C = 1 iff rd - rs - (1-C) >= 0 computed unsigned; V = signed overflow of that difference |
| 0C | CMP rd, rs | flags of SUB rd, rs; rd unchanged | as SUB |
| 0D | ADDI rd, imm (+imm) | rd = rd + imm | as ADD |
| 0E | CMPI rd, imm (+imm) | flags of rd - imm | as SUB |
| 10 | AND rd, rs | rd &= rs | Z N; V = 0; C unchanged |
| 11 | OR rd, rs | rd \|= rs | Z N; V = 0; C unchanged |
| 12 | XOR rd, rs | rd ^= rs | Z N; V = 0; C unchanged |
| 13 | NOT rd | rd = ~rd | Z N; V = 0; C unchanged |
| 14 | SHL rd, k | shift left k (0-15) places | Z N; V = 0; C = the last bit shifted out; **k = 0: rd unchanged, C unchanged** (Z N still set) |
| 15 | SHR rd, k | logical shift right k | as SHL |
| 16 | SAR rd, k | arithmetic shift right k (copies bit 15) | as SHL |
| 17 | ROL rd, k | rotate the 17-bit value C:rd left k times (each step: new C = old bit 15, new bit 0 = old C) | Z N of rd; V = 0; C as rotated |
| 18 | MUL rd, rs | P = rd * rs **unsigned** 32-bit; rd = P & 0xFFFF; if d < 7, r(d+1) = P >> 16 (d = 7: the high word is discarded). Both operands are read before either write. | Z = (P == 0); N = bit 31 of P; C = V = (P >> 16 != 0) |
| 19 | DIV rd, rs | **signed**, quotient truncated **toward zero** (NOT floor); remainder = rd - q*rs (takes the dividend's sign); rd = q; if d < 7, r(d+1) = remainder. rs = 0: **fault `DIVZERO`**. rd = -32768, rs = -1: rd = 0x8000, remainder 0, V = 1. | Z N of q; V = 0 except the overflow case; C unchanged |
| 20 | B imm (+imm) | pc = next + imm | - |
| 21 | BZ (+imm) | branch if Z | - |
| 22 | BNZ (+imm) | branch if !Z | - |
| 23 | BLT (+imm) | branch if N != V (signed <) | - |
| 24 | BGE (+imm) | branch if N == V | - |
| 25 | BLO (+imm) | branch if **C == 0** (unsigned <, given SUB's no-borrow carry) | - |
| 26 | BHS (+imm) | branch if C == 1 (unsigned >=) | - |
| 27 | BGT (+imm) | branch if !Z and N == V | - |
| 28 | BLE (+imm) | branch if Z or N != V | - |
| 29 | JMP rs | pc = rs (absolute) | - |
| 2A | CALL imm (+imm) | push next; pc = next + imm | - |
| 2B | RET | pop pc | - |
| 2C | PUSH rs | push rs | - |
| 2D | POP rd | pop into rd | - |
| 2E | PUSHF | push the flags word: **bit 3 = C, bit 2 = V, bit 1 = N, bit 0 = Z**, other bits 0 | - |
| 2F | POPF | pop a word; set the flags from those bits (other bits ignored) | as popped |
| 30 | OUT rs | append chr(rs & 0xFF) to the output | - |
| 31 | OUTN rs | append rs as a **signed** decimal, then one `"\n"` | - |

Any other op: **fault `ILLEGAL`**.

## Ending a run

`run` returns a dict:

```
{"status": "HALT" | "LIMIT" | "DIVZERO" | "ILLEGAL",
 "pc": int, "sp": int, "regs": [r0..r7 as unsigned ints], "flags": str, "steps": int, "output": str}
```

- `steps` = instructions **completed** (HLT counts; a faulting instruction does not).
- `HALT`: `pc` = next (the address after the HLT).
- Fault (`DIVZERO`, `ILLEGAL`): the faulting instruction has **no effect** (no register, flag or sp change) and `pc` =
  its own address (where its instruction word starts).
- `LIMIT`: when `steps == max_steps` before a HLT or fault, stop; `pc` = the address of the next instruction to run.
- `flags`: four characters in the order `N Z C V`, the letter when set, `-` when clear, e.g. `"-ZC-"`.

`kx8asm.py` (provided, correct) assembles the text syntax the tests use; read it for the operand syntax.

//! DeepSeek V4.1 engram — the HOST half of the lookup.
//!
//! Engram is an n-gram keyed associative table added straight into the widened residual stream at
//! a few layers (`deepseek41.cpp` 37-440, `engram_index()` / `set_input()` / `build_engram()`).
//! Everything up to the point where the residual stream itself is needed is pure row arithmetic
//! over TOKEN IDS — no weights, no activations — so it runs here rather than in the graph:
//!
//! ```text
//!   ids   = hash(token window)                            -> n_bucket table rows
//!   emb   = gather + dequantize those rows                -> n_bucket * key_length f32
//!   kv    = engram_wkv . emb                              -> hc*n_embd + n_embd f32
//!   value = kv[hc*n_embd ..]                              -> n_embd f32 (shared by every copy)
//!   k'    = rmsnorm(kv[..hc*n_embd]) * engram_k * engram_q -> hc*n_embd f32
//! ```
//!
//! `k'` folds `engram_q` in, because the score is
//! `sum_d (rmsnorm(x)[d] * q_w[d]) * (rmsnorm(key)[d] * k_w[d])`, which is
//! `sum_d rmsnorm(x)[d] * (rmsnorm(key)[d] * k_w[d] * q_w[d])` — so the query-side weights belong
//! on the key, the host-side half. What is left for the graph is a weightless `RmsNorm` of the
//! stream, one `Op::QwenPleGate` (whose kernel already computes exactly `value * gate` with
//! `gate = sigmoid(sign(s) * sqrt(max(|s|, 1e-6)))` and `s = dot/sqrt(n_embd)`), and an `Add`.
//! Three ops that already exist on both backends, so engram needs no new operator.
//!
//! The table itself is never resident: one token touches `n_bucket` rows of `key_length` values,
//! and for this checkpoint a row is exactly ONE Q2_K block (256 values, 84 bytes).

use crate::Config;

/// One engram layer's hash map, borrowed from [`Config`]. Split out from `Config` so the row
/// arithmetic can be exercised without building a whole model config.
#[derive(Clone, Copy, Debug)]
pub struct EngramMap<'a> {
    /// This layer's slot in the per-layer arrays (`Config::engram_index`).
    pub slot: usize,
    pub n_gram: usize,
    pub n_head: usize,
    pub pad_id: u64,
    pub multipliers: &'a [u64],
    pub primes: &'a [u64],
    pub offsets: &'a [u64],
    pub token_map: &'a [u64],
}

impl<'a> EngramMap<'a> {
    /// Borrow layer `il`'s map, or `None` when the layer carries no engram.
    pub fn for_layer(cfg: &'a Config, il: usize) -> Option<Self> {
        let slot = cfg.engram_index(il)?;
        Some(Self {
            slot,
            n_gram: cfg.engram_max_ngram,
            n_head: cfg.engram_n_head,
            pad_id: cfg.engram_pad_id,
            multipliers: &cfg.engram_multipliers,
            primes: &cfg.engram_primes,
            offsets: &cfg.engram_offsets,
            token_map: &cfg.engram_token_map,
        })
    }

    /// Buckets one token fans out into: `(n_gram - 1) * n_head`.
    pub fn n_bucket(&self) -> usize {
        (self.n_gram - 1) * self.n_head
    }

    /// The table row every bucket of the window `[prev..., current]` selects.
    ///
    /// `prev` is oldest-first and holds `n_gram - 1` entries; a negative entry means "before the
    /// start of the sequence" and reads as `pad_id`, as does any token id outside the token map.
    /// The rolling hash is `ctx[0]*mult[0]` XOR-folded with `ctx[s]*mult[s]`, and only the buckets
    /// of position `s >= 1` are filled — position 0 has no bucket of its own, which is why
    /// `n_bucket` counts `n_gram - 1` positions.
    pub fn row_indices(&self, current: u32, prev: &[i32]) -> Vec<u32> {
        debug_assert_eq!(prev.len(), self.n_gram - 1);
        let map_of = |t: i64| -> u64 {
            if t < 0 || (t as usize) >= self.token_map.len() {
                self.pad_id
            } else {
                self.token_map[t as usize]
            }
        };

        let base = self.slot * self.n_gram;
        let mult = &self.multipliers[base..base + self.n_gram];

        let mut ctx = vec![0u64; self.n_gram];
        ctx[0] = map_of(current as i64);
        let mut blocked = false;
        for s in 1..self.n_gram {
            let t = if blocked {
                -1
            } else {
                prev[prev.len() - s] as i64
            };
            blocked = blocked || t < 0;
            ctx[s] = if blocked { self.pad_id } else { map_of(t) };
        }

        let bucket_base = self.slot * self.n_bucket();
        let mut rolling = ctx[0].wrapping_mul(mult[0]);
        let mut rows = vec![0u32; self.n_bucket()];
        for s in 1..self.n_gram {
            rolling ^= ctx[s].wrapping_mul(mult[s]);
            for h in 0..self.n_head {
                let b = (s - 1) * self.n_head + h;
                let prime = self.primes[bucket_base + b];
                debug_assert_ne!(prime, 0, "config rejects zero primes");
                rows[b] = (rolling % prime + self.offsets[bucket_base + b]) as u32;
            }
        }
        rows
    }
}

/// Elements in one Q2_K super-block (`ggml`'s `QK_K`).
pub const QK_K: usize = 256;
/// Bytes one Q2_K super-block occupies: 16 scale bytes, 64 quant bytes, `d`, `dmin`.
pub const Q2_K_BLOCK_BYTES: usize = 84;

/// `ggml`'s `dequantize_row_q2_K`, one block: `out[j] = d * scale[j/16] * q[j] - dmin * min[j/16]`.
///
/// Transcribed from `ggml-quants.c` so the row a table lookup lands on dequantizes exactly the way
/// the reference does — this is the only place the 60 GiB table is read.
pub fn dequant_q2k_block(block: &[u8], out: &mut [f32]) {
    debug_assert_eq!(block.len(), Q2_K_BLOCK_BYTES);
    debug_assert_eq!(out.len(), QK_K);

    let scales = &block[..16];
    let qs = &block[16..80];
    let d = f16_to_f32(u16::from_le_bytes([block[80], block[81]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[82], block[83]]));

    let mut is = 0usize;
    let mut o = 0usize;
    let mut q = 0usize;
    // Two 128-value halves; within each, four 2-bit planes. The plane's shift is `2*j` (0, 2, 4,
    // 6) and each plane RE-READS the same 32 quant bytes at that shift, which is why `q` advances
    // once per half rather than per plane (`ggml-quants.c::dequantize_row_q2_K`).
    for _ in 0..2 {
        for j in 0..4u32 {
            let shift = 2 * j;
            for half in 0..2 {
                let sc = scales[is];
                is += 1;
                let dl = d * f32::from(sc & 0x0F);
                let ml = dmin * f32::from(sc >> 4);
                let base = q + half * 16;
                for l in 0..16 {
                    let qv = i32::from((qs[base + l] >> shift) & 3);
                    out[o] = dl * qv as f32 - ml;
                    o += 1;
                }
            }
        }
        q += 32;
    }
}

/// Half-precision bits to `f32`, without pulling a dependency in for four bytes. Normals go
/// straight through the exponent rebias; a subnormal is `mant * 2^-24`, which f32 represents
/// exactly; `exp = 31` is inf/NaN.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = u32::from((bits >> 10) & 0x1F);
    let mant = u32::from(bits & 0x3FF);
    let neg = sign != 0;
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        let m = mant as f32 * 2f32.powi(-24);
        return if neg { -m } else { m };
    }
    if exp == 0x1F {
        let m = f32::from_bits(0x7F80_0000 | (mant << 13));
        return if neg { -m } else { m };
    }
    f32::from_bits(sign | ((exp + 127 - 15) << 23) | (mant << 13))
}

/// Gather `rows.len()` table rows and lay them out bucket-major: bucket `b` occupies
/// `[b*key_length, (b+1)*key_length)` — the order `build_inp_engram()` produces with its
/// `get_rows` + `reshape_2d(key_length*n_cols, nt)`.
///
/// `table` is the tensor's bytes with row `r` starting at `r * row_bytes`. Rows are copied, not
/// cached: one token reads `n_bucket * row_bytes` (2 KiB for this checkpoint).
pub fn gather_rows(table: &[u8], row_bytes: usize, rows: &[u32], key_length: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows.len() * key_length];
    let blocks = key_length.div_ceil(QK_K);
    debug_assert_eq!(row_bytes, blocks * Q2_K_BLOCK_BYTES, "row width must be whole Q2_K blocks");
    for (b, &row) in rows.iter().enumerate() {
        let start = row as usize * row_bytes;
        let src = &table[start..start + row_bytes];
        for blk in 0..blocks {
            let bytes = &src[blk * Q2_K_BLOCK_BYTES..(blk + 1) * Q2_K_BLOCK_BYTES];
            let dst = &mut out[b * key_length + blk * QK_K..b * key_length + (blk + 1) * QK_K];
            dequant_q2k_block(bytes, dst);
        }
    }
    out
}

/// `kv = engram_wkv · emb` — the one dense projection, `in_dim x out_dim` Q2_K weight rows.
///
/// Row-major `Q2_K` with `in_dim` values per output row, so a row is `in_dim/256` blocks. The
/// weight is 51 MiB per engram layer; callers that can afford it should dequantize once at load
/// and use [`project_f32`] instead, which this falls back to being equivalent to.
pub fn project_q2k(wkv: &[u8], in_dim: usize, emb: &[f32], out: &mut [f32]) {
    debug_assert_eq!(emb.len(), in_dim);
    let blocks = in_dim / QK_K;
    let row_bytes = blocks * Q2_K_BLOCK_BYTES;
    debug_assert_eq!(wkv.len(), out.len() * row_bytes);
    let mut deq = vec![0f32; QK_K];
    for (j, o) in out.iter_mut().enumerate() {
        let row = &wkv[j * row_bytes..(j + 1) * row_bytes];
        let mut acc = 0f32;
        for blk in 0..blocks {
            let bytes = &row[blk * Q2_K_BLOCK_BYTES..(blk + 1) * Q2_K_BLOCK_BYTES];
            dequant_q2k_block(bytes, &mut deq);
            let e = &emb[blk * QK_K..(blk + 1) * QK_K];
            for i in 0..QK_K {
                acc += deq[i] * e[i];
            }
        }
        *o = acc;
    }
}

/// `dst = kv · emb` against an already-dequantized (or natively f32) weight, same layout.
pub fn project_f32(w: &[f32], in_dim: usize, emb: &[f32], out: &mut [f32]) {
    debug_assert_eq!(emb.len(), in_dim);
    debug_assert_eq!(w.len(), out.len() * in_dim);
    for (j, o) in out.iter_mut().enumerate() {
        let row = &w[j * in_dim..(j + 1) * in_dim];
        let mut acc = 0f32;
        for i in 0..in_dim {
            acc += row[i] * emb[i];
        }
        *o = acc;
    }
}

/// Split `kv` into the stream half and the shared value, then fold the query-side weights in.
///
/// `key` is normalised per COPY (not jointly over the copies — `grouped_norm` in
/// `deepseek41.cpp` reshapes to `[n_embd, hc, nt]` and norms over `n_embd`), scaled first by
/// `engram_k` and then by `engram_q`. Both weight arrays are `[hc * n_embd]` flat in copy-major
/// order, which is `ggml`'s own order for the file's `[n_embd, hc]` tensors.
///
/// Returns the `[hc * n_embd]` baked key; `value` is the tail of `kv`, untouched.
pub fn fold_key_with_query(
    key: &mut [f32],
    k_w: &[f32],
    q_w: &[f32],
    n_embd: usize,
    hc: usize,
    eps: f32,
) {
    debug_assert_eq!(key.len(), hc * n_embd);
    debug_assert_eq!(k_w.len(), hc * n_embd);
    debug_assert_eq!(q_w.len(), hc * n_embd);
    for h in 0..hc {
        let row = &mut key[h * n_embd..(h + 1) * n_embd];
        let n = row.len() as f32;
        let ms = row.iter().map(|v| v * v).sum::<f32>() / n;
        let inv = 1.0f32 / (ms + eps).sqrt();
        for d in 0..n_embd {
            let w = k_w[h * n_embd + d] * q_w[h * n_embd + d];
            row[d] = row[d] * inv * w;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map<'a>(
        multipliers: &'a [u64],
        primes: &'a [u64],
        offsets: &'a [u64],
        token_map: &'a [u64],
    ) -> EngramMap<'a> {
        EngramMap {
            slot: 0,
            n_gram: 4,
            n_head: 2,
            pad_id: 7,
            multipliers,
            primes,
            offsets,
            token_map,
        }
    }

    #[test]
    fn hash_is_deterministic_and_in_range() {
        let mult = [3u64, 5, 7, 11];
        let primes = [101u64, 103, 107, 109, 113, 127];
        let offs = [0u64, 1, 2, 3, 4, 5];
        let tmap: Vec<u64> = (0..64).collect();
        let m = map(&mult, &primes, &offs, &tmap);

        let a = m.row_indices(41, &[1, 2, 3]);
        let b = m.row_indices(41, &[1, 2, 3]);
        assert_eq!(a, b, "same window must hash the same");
        assert_eq!(a.len(), 6, "(n_gram-1)*n_head buckets");
        for (i, &r) in a.iter().enumerate() {
            let limit = primes[i] + offs[i];
            assert!(u64::from(r) < limit, "row {i} = {r} must stay inside the table");
        }
        // A different current token must move at least one bucket.
        let c = m.row_indices(42, &[1, 2, 3]);
        assert_ne!(a, c);
    }

    #[test]
    fn sequence_start_reads_as_padding() {
        let mult = [3u64, 5, 7, 11];
        let primes = [101u64; 6];
        let offs = [0u64; 6];
        let tmap: Vec<u64> = (0..64).collect();
        let m = map(&mult, &primes, &offs, &tmap);

        // All predecessors unavailable: identical to hashing the pad id directly.
        let start = m.row_indices(5, &[-1, -1, -1]);
        // Once the window is blocked, later positions stay at pad, so this equals a window of
        // one real predecessor followed by two pads.
        let one = m.row_indices(5, &[-1, -1, 9]);
        assert_ne!(start, one);
        // And a full window of real tokens differs from both.
        let full = m.row_indices(5, &[9, 10, 11]);
        assert_ne!(full, one);
    }

    #[test]
    fn q2k_block_matches_reference_formula() {
        // d = 1.0 (0x3C00), dmin = 0.5 (0x3800); scales alternate (scale=1,min=0) and
        // (scale=0,min=1); quants all 2.
        let mut block = [0u8; Q2_K_BLOCK_BYTES];
        for b in 0..16 {
            block[b] = if b % 2 == 0 { 0x01 } else { 0x10 };
        }
        for b in 16..80 {
            // all-ones keeps every 2-bit field at 3 whatever the shift plane is
            block[b] = 0xFF;
        }
        block[80..82].copy_from_slice(&0x3C00u16.to_le_bytes());
        block[82..84].copy_from_slice(&0x3800u16.to_le_bytes());

        let mut out = [0f32; QK_K];
        dequant_q2k_block(&block, &mut out);
        for (j, &v) in out.iter().enumerate() {
            let group = j / 16;
            let expect = if group % 2 == 0 {
                // scale 1, min 0 -> 1.0 * 3 - 0.0
                3.0
            } else {
                // scale 0, min 1 -> 0.0 * 3 - 0.5 * 1.0
                -0.5
            };
            assert!((v - expect).abs() < 1e-6, "element {j}: {v} != {expect}");
        }
    }

    #[test]
    fn key_fold_normalises_per_copy() {
        let (ne, hc) = (4usize, 2usize);
        let mut key = vec![0f32; hc * ne];
        // copy 0: [3, 4, 0, 0] -> rms 2.5 ; copy 1: [0, 0, 1, 1] -> rms sqrt(0.5)
        key[0..4].copy_from_slice(&[3.0, 4.0, 0.0, 0.0]);
        key[4..8].copy_from_slice(&[0.0, 0.0, 1.0, 1.0]);
        let k_w = vec![1f32; hc * ne];
        let q_w = vec![2f32; hc * ne];
        fold_key_with_query(&mut key, &k_w, &q_w, ne, hc, 0.0);
        // rms(3,4,0,0) = sqrt((9+16)/4) = 2.5 -> 3/2.5*2 = 2.4, 4/2.5*2 = 3.2
        assert!((key[0] - 2.4).abs() < 1e-5, "{}", key[0]);
        assert!((key[1] - 3.2).abs() < 1e-5, "{}", key[1]);
        // rms(0,0,1,1) = sqrt(2/4) = 0.7071 -> 1/0.7071*2 = 2.8284
        assert!((key[6] - 2.8284).abs() < 1e-4, "{}", key[6]);
    }

    #[test]
    fn q2k_block_matches_the_shared_dequantizer() {
        // The repo's `infr_gguf` dequant is the shipped Q2_K oracle. The row gather must agree with
        // it element for element — and the quant bytes here vary with the index so a wrong plane
        // shift cannot slip through the way an all-`0b10` block would let it.
        let mut block = [0u8; Q2_K_BLOCK_BYTES];
        for (b, v) in block[..16].iter_mut().enumerate() {
            *v = (b as u8 & 0x0F) | (((b as u8) % 3) << 4);
        }
        for (b, v) in block[16..80].iter_mut().enumerate() {
            *v = (b as u8).wrapping_mul(37) ^ 0x5A;
        }
        block[80..82].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
        block[82..84].copy_from_slice(&0x3800u16.to_le_bytes()); // dmin = 0.5

        let mut got = [0f32; QK_K];
        dequant_q2k_block(&block, &mut got);
        let want =
            infr_gguf::dequant::dequant_block(infr_core::DType::Q2K, &block).expect("oracle");
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-6,
                "element {i}: {g} vs the shared dequantizer's {w}"
            );
        }
    }

    #[test]
    fn f16_roundtrip_covers_the_encodings_q2k_uses() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0x3800), 0.5);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        // smallest positive subnormal: 2^-24
        assert!((f16_to_f32(0x0001) - 5.960_464_5e-8).abs() < 1e-14);
    }
}

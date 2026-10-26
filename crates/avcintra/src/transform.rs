//! Integer transforms, quantisation and the normative scaling and inverse transforms (§8.5) for
//! flat scaling matrices, with qP′ (QP + QpBdOffset) up to 63.

use crate::tables::{DEQUANT4, QUANT4, pos_class4};

/// Forward 4x4 core transform of a residual block (raster order).
#[inline]
pub fn fdct4(d: &[i32; 16]) -> [i32; 16] {
    let mut t = [0i32; 16];
    for i in 0..4 {
        let (a, b, c, e) = (d[i * 4], d[i * 4 + 1], d[i * 4 + 2], d[i * 4 + 3]);
        let (s03, d03, s12, d12) = (a + e, a - e, b + c, b - c);
        t[i * 4] = s03 + s12;
        t[i * 4 + 1] = 2 * d03 + d12;
        t[i * 4 + 2] = s03 - s12;
        t[i * 4 + 3] = d03 - 2 * d12;
    }
    let mut o = [0i32; 16];
    for i in 0..4 {
        let (a, b, c, e) = (t[i], t[4 + i], t[8 + i], t[12 + i]);
        let (s03, d03, s12, d12) = (a + e, a - e, b + c, b - c);
        o[i] = s03 + s12;
        o[4 + i] = 2 * d03 + d12;
        o[8 + i] = s03 - s12;
        o[12 + i] = d03 - 2 * d12;
    }
    o
}

/// Normative inverse 4x4 transform (§8.5.12.2) of scaled coefficients `d` (raster); returns the
/// residual (after the final `+ 32 >> 6`).
#[inline]
pub fn idct4(d: &[i32; 16]) -> [i32; 16] {
    let mut f = [0i32; 16];
    for i in 0..4 {
        let (d0, d1, d2, d3) = (d[i * 4], d[i * 4 + 1], d[i * 4 + 2], d[i * 4 + 3]);
        let (e, ff, g, h) = (d0 + d2, d0 - d2, (d1 >> 1) - d3, d1 + (d3 >> 1));
        f[i * 4] = e + h;
        f[i * 4 + 1] = ff + g;
        f[i * 4 + 2] = ff - g;
        f[i * 4 + 3] = e - h;
    }
    let mut r = [0i32; 16];
    for j in 0..4 {
        let (f0, f1, f2, f3) = (f[j], f[4 + j], f[8 + j], f[12 + j]);
        let (e, ff, g, h) = (f0 + f2, f0 - f2, (f1 >> 1) - f3, f1 + (f3 >> 1));
        r[j] = (e + h + 32) >> 6;
        r[4 + j] = (ff + g + 32) >> 6;
        r[8 + j] = (ff - g + 32) >> 6;
        r[12 + j] = (e - h + 32) >> 6;
    }
    r
}

/// Unnormalised 4x4 Hadamard (the Intra16x16 luma DC transform, and SATD).
#[inline]
pub fn hadamard4(d: &[i32; 16]) -> [i32; 16] {
    let mut t = [0i32; 16];
    for i in 0..4 {
        let (a, b, c, e) = (d[i * 4], d[i * 4 + 1], d[i * 4 + 2], d[i * 4 + 3]);
        t[i * 4] = a + b + c + e;
        t[i * 4 + 1] = a + b - c - e;
        t[i * 4 + 2] = a - b - c + e;
        t[i * 4 + 3] = a - b + c - e;
    }
    let mut o = [0i32; 16];
    for i in 0..4 {
        let (a, b, c, e) = (t[i], t[4 + i], t[8 + i], t[12 + i]);
        o[i] = a + b + c + e;
        o[4 + i] = a + b - c - e;
        o[8 + i] = a - b - c + e;
        o[12 + i] = a - b + c - e;
    }
    o
}

/// Sum of absolute Hadamard-transformed differences of a 4x4 residual, halved.
#[inline]
pub fn satd4(d: &[i32; 16]) -> u32 {
    (hadamard4(d).iter().map(|v| v.unsigned_abs()).sum::<u32>() + 1) >> 1
}

/// Quantisation rounding as a fraction of a step, in 1/256: about a third for intra.
const ROUND256: i64 = 85;

/// A component's 4x4 scaling (weightScale4x4, raster) with the forward multipliers that match
/// it: MF·16/w for every qP % 6 and position.
pub struct Scale {
    pub w: [i32; 16],
    mf: [[i64; 16]; 6],
}

impl Scale {
    pub fn new(w: [i32; 16]) -> Scale {
        let mf = std::array::from_fn(|m| std::array::from_fn(|i| (QUANT4[m][pos_class4(i)] * 16 + i64::from(w[i]) / 2) / i64::from(w[i])));
        Scale { w, mf }
    }
    /// LevelScale4x4(m, i, j) (equation 8-313).
    fn level_scale(&self, m: usize, i: usize) -> i32 {
        self.w[i] * DEQUANT4[m][pos_class4(i)]
    }
}

/// Quantise a 4x4 block of transform coefficients (raster) in place into levels at qP′ `qp`;
/// `skip_dc` leaves c[0] untouched (Intra16x16 and chroma AC). Returns the non-zero count.
#[inline]
pub fn quant4(c: &mut [i32; 16], s: &Scale, qp: i32, skip_dc: bool) -> u32 {
    let qbits = 15 + (qp / 6) as u32;
    let f = ((1i64 << qbits) * ROUND256) >> 8;
    let mf = &s.mf[(qp % 6) as usize];
    let mut nz = 0;
    for i in usize::from(skip_dc)..16 {
        let v = c[i];
        let a = ((i64::from(v.unsigned_abs()) * mf[i] + f) >> qbits) as i32;
        c[i] = if v < 0 { -a } else { a };
        nz += u32::from(a != 0);
    }
    nz
}

/// Scale 4x4 levels (raster) as the decoder does (§8.5.12.1).
#[inline]
pub fn dequant4(c: &mut [i32; 16], s: &Scale, qp: i32, skip_dc: bool) {
    let (m, q6) = ((qp % 6) as usize, qp / 6);
    for i in usize::from(skip_dc)..16 {
        let ls = s.level_scale(m, i);
        c[i] = if qp >= 24 { (c[i] * ls) << (q6 - 4) } else { (c[i] * ls + (1 << (3 - q6))) >> (4 - q6) };
    }
}

/// Intra16x16 luma DC: the 4x4 raster of the blocks' DC coefficients → levels (raster).
pub fn quant_luma_dc(dc: &[i32; 16], s: &Scale, qp: i32) -> [i32; 16] {
    let h = hadamard4(dc);
    let qbits = 16 + (qp / 6) as u32;
    let f = ((1i64 << qbits) * ROUND256) >> 8;
    let mf = s.mf[(qp % 6) as usize][0];
    let mut out = [0i32; 16];
    for i in 0..16 {
        let v = h[i] >> 1;
        let a = ((i64::from(v.unsigned_abs()) * mf + f) >> qbits) as i32;
        out[i] = if v < 0 { -a } else { a };
    }
    out
}

/// Intra16x16 luma DC scaling and transform (§8.5.10): levels (raster) → dcY (raster).
pub fn dequant_luma_dc(c: &[i32; 16], s: &Scale, qp: i32) -> [i32; 16] {
    let f = hadamard4(c);
    let ls = s.level_scale((qp % 6) as usize, 0);
    let q6 = qp / 6;
    let mut o = [0i32; 16];
    for i in 0..16 {
        o[i] = if qp >= 36 { (f[i] * ls) << (q6 - 6) } else { (f[i] * ls + (1 << (5 - q6))) >> (6 - q6) };
    }
    o
}

/// The 2x4 chroma DC transform (equation 8-325, and its forward counterpart: the matrices are
/// symmetric): `c` as 4 rows of 2.
fn cdc422(c: &[[i32; 2]; 4]) -> [[i32; 2]; 4] {
    // rows of the 4-point matrix of 8-325
    const A: [[i32; 4]; 4] = [[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, -1, 1], [1, -1, 1, -1]];
    let mut t = [[0i32; 2]; 4];
    for i in 0..4 {
        for j in 0..2 {
            t[i][j] = (0..4).map(|k| A[i][k] * c[k][j]).sum();
        }
    }
    let mut o = [[0i32; 2]; 4];
    for i in 0..4 {
        o[i][0] = t[i][0] + t[i][1];
        o[i][1] = t[i][0] - t[i][1];
    }
    o
}

/// 4:2:2 chroma DC at QP′C `qpc`: the blocks' DC coefficients (4 rows of 2) → levels (4 rows
/// of 2), quantised at qPDC = QP′C + 3 like the decoder scales them.
pub fn quant_chroma_dc422(dc: &[[i32; 2]; 4], s: &Scale, qpc: i32) -> [[i32; 2]; 4] {
    let h = cdc422(dc);
    let q = qpc + 3;
    let qbits = 16 + (q / 6) as u32;
    let f = ((1i64 << qbits) * ROUND256) >> 8;
    let mf = s.mf[(q % 6) as usize][0];
    let mut out = [[0i32; 2]; 4];
    for i in 0..4 {
        for j in 0..2 {
            let v = h[i][j];
            let a = ((i64::from(v.unsigned_abs()) * mf + f) >> qbits) as i32;
            out[i][j] = if v < 0 { -a } else { a };
        }
    }
    out
}

/// 4:2:2 chroma DC transform and scaling (§8.5.11): levels (4 rows of 2) → dcC.
pub fn dequant_chroma_dc422(c: &[[i32; 2]; 4], s: &Scale, qpc: i32) -> [[i32; 2]; 4] {
    let f = cdc422(c);
    let q = qpc + 3;
    let ls = s.level_scale((q % 6) as usize, 0);
    let q6 = q / 6;
    let mut o = [[0i32; 2]; 4];
    for i in 0..4 {
        for j in 0..2 {
            o[i][j] = if q >= 36 { (f[i][j] * ls) << (q6 - 6) } else { (f[i][j] * ls + (1 << (5 - q6))) >> (6 - q6) };
        }
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat residual in every 4x4 block of an 8x16 chroma block comes back through the DC
    /// path alone, at every QP.
    #[test]
    fn chroma_dc_422_round_trips() {
        for qp in 0..52 {
            for r in [-300, -40, -3, 7, 55, 511] {
                // every block's DC coefficient of a flat residual r
                let dc = [[16 * r; 2]; 4];
                let sc = Scale::new(crate::tables::weights(&crate::tables::CHROMA_4X4));
                let lv = quant_chroma_dc422(&dc, &sc, qp);
                let d = dequant_chroma_dc422(&lv, &sc, qp);
                let mut blk = [0i32; 16];
                blk[0] = d[0][0];
                let res = idct4(&blk);
                let step = 0.625 * 2f64.powf(f64::from(qp + 3) / 6.0) / 8.0;
                assert!((f64::from(res[5] - r)).abs() <= step.max(1.0) + 1.0, "qp {qp} r {r}: {}", res[5]);
            }
        }
    }

    #[test]
    fn luma_dc_round_trips() {
        for qp in 0..64 {
            let r = 123;
            let dc = [16 * r; 16];
            let sc = Scale::new(crate::tables::weights(&crate::tables::DEFAULT_4X4_INTRA));
            let lv = quant_luma_dc(&dc, &sc, qp);
            let d = dequant_luma_dc(&lv, &sc, qp);
            let mut blk = [0i32; 16];
            blk[0] = d[0];
            let res = idct4(&blk);
            let step = 0.625 * 2f64.powf(f64::from(qp) / 6.0) * 6.0 / 16.0;
            assert!((f64::from(res[0] - r)).abs() <= step.max(1.0) + 1.0, "qp {qp}: {}", res[0]);
        }
    }

    #[test]
    fn ac_round_trips() {
        let d: [i32; 16] = std::array::from_fn(|i| ((i * 37) % 23) as i32 * 9 - 90);
        let sc = Scale::new([16; 16]);
        for qp in [12, 20, 30, 40] {
            let mut c = fdct4(&d);
            quant4(&mut c, &sc, qp, false);
            dequant4(&mut c, &sc, qp, false);
            let r = idct4(&c);
            let err: i32 = r.iter().zip(&d).map(|(a, b)| (a - b).abs()).max().unwrap_or(0);
            let step = 0.625 * 2f64.powf(f64::from(qp) / 6.0);
            assert!(f64::from(err) <= step * 2.0 + 1.0, "qp {qp}: {err}");
        }
    }
}

//! 8×8 DCT (H.262 Annex A): the forward transform for coding, and the inverse transform the
//! reconstruction uses — the same separable `f32` evaluation as FilmCraft's MPEG-2 decoder, so
//! its pictures and the encoder's references agree.

use std::sync::OnceLock;

/// `M[x][u] = C(u)/2 · cos((2x+1)uπ/16)`, `C(0) = 1/√2`, `C(u>0) = 1`.
fn basis() -> &'static [[f32; 8]; 8] {
    static B: OnceLock<[[f32; 8]; 8]> = OnceLock::new();
    B.get_or_init(|| {
        let mut m = [[0f32; 8]; 8];
        for (x, row) in m.iter_mut().enumerate() {
            for (u, v) in row.iter_mut().enumerate() {
                let c = if u == 0 { std::f64::consts::FRAC_1_SQRT_2 } else { 1.0 };
                *v = (c / 2.0 * (((2 * x + 1) * u) as f64 * std::f64::consts::PI / 16.0).cos()) as f32;
            }
        }
        m
    })
}

/// Forward DCT of a block of samples or residuals (raster): F[v][u] = Σ M[y][v] M[x][u] f[y][x].
pub fn fdct(f: &[i32; 64]) -> [f32; 64] {
    let m = basis();
    let mut t = [[0f32; 8]; 8];
    for y in 0..8 {
        for u in 0..8 {
            t[y][u] = (0..8).map(|x| m[x][u] * f[y * 8 + x] as f32).sum();
        }
    }
    let mut out = [0f32; 64];
    for v in 0..8 {
        for u in 0..8 {
            out[v * 8 + u] = (0..8).map(|y| m[y][v] * t[y][u]).sum();
        }
    }
    out
}

/// In-place IDCT of dequantised coefficients (raster), rounded and saturated to [−256, 255].
pub fn idct(blk: &mut [i32; 64]) {
    if blk[1..].iter().all(|&c| c == 0) {
        let v = ((blk[0] + 4) >> 3).clamp(-256, 255);
        blk.fill(v);
        return;
    }
    let m = basis();
    // The DC term F[0][0]/8 is added exactly (so ties round as in the DC-only path).
    let dc = blk[0] as f32 * 0.125;
    blk[0] = 0;
    let mut tmp = [[0f32; 8]; 8];
    for v in 0..8 {
        let r = &blk[v * 8..v * 8 + 8];
        if r.iter().all(|&c| c == 0) {
            continue;
        }
        let f: [f32; 8] = std::array::from_fn(|u| r[u] as f32);
        for x in 0..4 {
            let mx = &m[x];
            let e = mx[0] * f[0] + mx[2] * f[2] + mx[4] * f[4] + mx[6] * f[6];
            let o = mx[1] * f[1] + mx[3] * f[3] + mx[5] * f[5] + mx[7] * f[7];
            tmp[v][x] = e + o;
            tmp[v][7 - x] = e - o;
        }
    }
    for y in 0..4 {
        let my = &m[y];
        let mut e = [0f32; 8];
        let mut o = [0f32; 8];
        for x in 0..8 {
            e[x] = my[0] * tmp[0][x] + my[2] * tmp[2][x] + my[4] * tmp[4][x] + my[6] * tmp[6][x];
            o[x] = my[1] * tmp[1][x] + my[3] * tmp[3][x] + my[5] * tmp[5][x] + my[7] * tmp[7][x];
        }
        for x in 0..8 {
            blk[y * 8 + x] = ((dc + (e[x] + o[x]) + 0.5).floor() as i32).clamp(-256, 255);
            blk[(7 - y) * 8 + x] = ((dc + (e[x] - o[x]) + 0.5).floor() as i32).clamp(-256, 255);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips() {
        let f: [i32; 64] = std::array::from_fn(|i| ((i * 37) % 51) as i32 - 25);
        let c = fdct(&f);
        let mut b: [i32; 64] = std::array::from_fn(|i| c[i].round() as i32);
        idct(&mut b);
        assert!(b.iter().zip(&f).all(|(a, b)| (a - b).abs() <= 1));
        let flat = [100i32; 64];
        assert!((fdct(&flat)[0] - 800.0).abs() < 0.01, "F00 is 8 × the mean");
    }
}

//! Coding one frame picture of interlaced 4:2:2 video (H.262 §6.2.4–§6.2.6, §7): motion search
//! (frame and field vectors, half-sample), the choice of each macroblock's prediction and DCT
//! type, quantisation with the non-linear scale, and the slices — one per macroblock row, coded
//! in parallel.
//!
//! A picture is planned once (motion, predictions, transform coefficients, activity) and then
//! coded at as many quantiser scales as the rate control asks; the reconstruction, for the
//! pictures others predict from, is made from the coding finally kept.

use filmcraft_bitstream::BitWriter;

use crate::dct::{fdct, idct};
use crate::tables::{ALT_SCAN, Code, DEFAULT_INTRA, NON_LINEAR_Q, q_code, tables};

/// Picture size in samples (1080 lines coded as 68 macroblock rows) and macroblocks.
pub const W: usize = 1920;
pub const H: usize = 1088;
pub const CW: usize = W / 2;
pub const MB_W: usize = W / 16;
pub const MB_H: usize = H / 16;

/// A frame of 8-bit 4:2:2 samples, 1920x1088.
#[derive(Clone)]
pub struct Pic {
    pub y: Vec<u8>,
    pub c: [Vec<u8>; 2],
}

impl Pic {
    pub fn blank() -> Pic {
        Pic { y: vec![16; W * H], c: [vec![128; CW * H], vec![128; CW * H]] }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PType {
    I,
    P,
    B,
}

impl PType {
    pub fn code(self) -> u32 {
        match self {
            PType::I => 1,
            PType::P => 2,
            PType::B => 3,
        }
    }
}

/// intra_dc_precision 2 (10 bits): the DC is quantised by 2.
pub const DC_PRECISION: u32 = 2;
const DC_MULT: i32 = 8 >> DC_PRECISION;
const DC_RESET: i32 = 1 << (7 + DC_PRECISION);

/// macroblock_type flags (Tables B.2–B.4).
const MB_QUANT: u8 = 1;
const MB_FWD: u8 = 2;
const MB_BWD: u8 = 4;
const MB_PATTERN: u8 = 8;
const MB_INTRA: u8 = 16;

/// The macroblock_type code for `flags` in a picture of type `t`.
fn mb_type_code(t: PType, flags: u8) -> Code {
    let s = match (t, flags) {
        (PType::I, MB_INTRA) => "1",
        (PType::I, _) => "01",
        (PType::P, f) if f == MB_FWD | MB_PATTERN => "1",
        (PType::P, MB_PATTERN) => "01",
        (PType::P, MB_FWD) => "001",
        (PType::P, MB_INTRA) => "00011",
        (PType::P, f) if f == MB_QUANT | MB_FWD | MB_PATTERN => "00010",
        (PType::P, f) if f == MB_QUANT | MB_PATTERN => "00001",
        (PType::P, _) => "000001",
        (PType::B, f) if f == MB_FWD | MB_BWD => "10",
        (PType::B, f) if f == MB_FWD | MB_BWD | MB_PATTERN => "11",
        (PType::B, MB_BWD) => "010",
        (PType::B, f) if f == MB_BWD | MB_PATTERN => "011",
        (PType::B, MB_FWD) => "0010",
        (PType::B, f) if f == MB_FWD | MB_PATTERN => "0011",
        (PType::B, MB_INTRA) => "00011",
        (PType::B, f) if f == MB_QUANT | MB_FWD | MB_BWD | MB_PATTERN => "00010",
        (PType::B, f) if f == MB_QUANT | MB_FWD | MB_PATTERN => "000011",
        (PType::B, f) if f == MB_QUANT | MB_BWD | MB_PATTERN => "000010",
        (PType::B, _) => "000001",
    };
    let mut c = Code::default();
    for b in s.bytes() {
        c.bits = (c.bits << 1) | u32::from(b - b'0');
        c.len += 1;
    }
    c
}

/// How a macroblock is predicted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pred {
    Intra,
    /// Frame prediction: forward and/or backward vector (half samples, frame lines).
    Frame { fwd: Option<[i32; 2]>, bwd: Option<[i32; 2]> },
    /// Field prediction from one direction (0 forward, 1 backward): for the top and the bottom
    /// field of the macroblock, the reference field and the vector (field lines).
    Field { dir: usize, sel: [usize; 2], mv: [[i32; 2]; 2] },
}

/// A macroblock's plan: its prediction, DCT type, coefficients and activity.
pub struct MbPlan {
    pred: Pred,
    field_dct: bool,
    /// Transform coefficients of the eight blocks (Y0..Y3, Cb0, Cr0, Cb1, Cr1).
    coef: [[f32; 64]; 8],
    /// Prediction samples (luma 16x16, chroma 8x16 each), for the reconstruction.
    pred_y: [u8; 256],
    pred_c: [[u8; 128]; 2],
    /// Normalised activity (TM5): the quantiser scale is multiplied by it.
    act: f64,
}

impl MbPlan {
    /// TM5 activity of the macroblock.
    pub fn activity(&self) -> f64 {
        self.act
    }
}

/// Everything about the picture its slices share.
pub struct Setup<'a> {
    pub ptype: PType,
    pub src: &'a Pic,
    pub fwd: Option<&'a Pic>,
    pub bwd: Option<&'a Pic>,
    /// f_code for (forward, backward).
    pub f_code: [u32; 2],
}

// ---------------------------------------------------------------- prediction

/// A reference plane, whole frame or one field.
#[derive(Clone, Copy)]
struct Plane<'a> {
    data: &'a [u8],
    off: usize,
    stride: usize,
    w: usize,
    h: usize,
}

fn plane(p: &Pic, c: usize, field: Option<usize>) -> Plane<'_> {
    let (data, w) = if c == 0 { (&p.y[..], W) } else { (&p.c[c - 1][..], CW) };
    match field {
        None => Plane { data, off: 0, stride: w, w, h: H },
        Some(f) => Plane { data, off: f * w, stride: 2 * w, w, h: H / 2 },
    }
}

/// The `bw`×`bh` prediction at half-sample vector `mv` from block position (`x`, `y`) into
/// `dst` rows `first + k·step` (stride `ds`), averaging with what is there when `avg` — the
/// decoder's half-sample rules (§7.6.4).
#[allow(clippy::too_many_arguments)]
fn predict(dst: &mut [u8], first: usize, step: usize, ds: usize, src: Plane, x: i32, y: i32, mv: [i32; 2], bw: usize, bh: usize, avg: bool) {
    let (px, py) = (x + (mv[0] >> 1), y + (mv[1] >> 1));
    let (hx, hy) = (mv[0] & 1 != 0, mv[1] & 1 != 0);
    let get = |xx: i32, yy: i32| -> u16 {
        let xx = xx.clamp(0, src.w as i32 - 1) as usize;
        let yy = yy.clamp(0, src.h as i32 - 1) as usize;
        u16::from(src.data[src.off + yy * src.stride + xx])
    };
    for r in 0..bh {
        let yy = py + r as i32;
        let at = (first + r * step) * ds;
        for i in 0..bw {
            let xx = px + i as i32;
            let v = match (hx, hy) {
                (false, false) => get(xx, yy),
                (true, false) => (get(xx, yy) + get(xx + 1, yy) + 1) >> 1,
                (false, true) => (get(xx, yy) + get(xx, yy + 1) + 1) >> 1,
                (true, true) => (get(xx, yy) + get(xx + 1, yy) + get(xx, yy + 1) + get(xx + 1, yy + 1) + 2) >> 2,
            } as u8;
            let o = &mut dst[at + i];
            *o = if avg { ((u16::from(*o) + u16::from(v) + 1) >> 1) as u8 } else { v };
        }
    }
}

/// The prediction of macroblock (`mbx`, `mby`) by `pred` from the references.
fn form_prediction(s: &Setup, pred: &Pred, mbx: usize, mby: usize, y: &mut [u8; 256], c: &mut [[u8; 128]; 2]) {
    let refs = [s.fwd, s.bwd];
    let (x0, y0) = ((mbx * 16) as i32, (mby * 16) as i32);
    let cx0 = (mbx * 8) as i32;
    let chroma_mv = |mv: [i32; 2]| [mv[0] / 2, mv[1]];
    match *pred {
        Pred::Intra => {}
        Pred::Frame { fwd, bwd } => {
            let mut avg = false;
            for (d, mv) in [fwd, bwd].into_iter().enumerate() {
                let (Some(mv), Some(r)) = (mv, refs[d].or(refs[1 - d])) else { continue };
                predict(y, 0, 1, 16, plane(r, 0, None), x0, y0, mv, 16, 16, avg);
                for k in 0..2 {
                    predict(&mut c[k], 0, 1, 8, plane(r, 1 + k, None), cx0, y0, chroma_mv(mv), 8, 16, avg);
                }
                avg = true;
            }
        }
        Pred::Field { dir, sel, mv } => {
            let Some(r) = refs[dir].or(refs[1 - dir]) else { return };
            for f in 0..2 {
                // this field's 8 lines of the macroblock, at field line mby·8
                predict(y, f, 2, 16, plane(r, 0, Some(sel[f])), x0, (mby * 8) as i32, mv[f], 16, 8, false);
                for k in 0..2 {
                    predict(&mut c[k], f, 2, 8, plane(r, 1 + k, Some(sel[f])), cx0, (mby * 8) as i32, chroma_mv(mv[f]), 8, 8, false);
                }
            }
        }
    }
}

// ---------------------------------------------------------------- motion search

/// Sum of absolute differences of the macroblock's luma (or one field of it, `field`: lines of
/// that parity) against a prediction.
fn sad(src: &[u8; 256], pred: &[u8; 256], field: Option<usize>) -> u32 {
    let mut s = 0u32;
    for r in 0..16 {
        if field.is_some_and(|f| r % 2 != f) {
            continue;
        }
        for i in 0..16 {
            s += u32::from(src[r * 16 + i].abs_diff(pred[r * 16 + i]));
        }
    }
    s
}

/// Integer-sample SAD of the 16-wide block at (`x`, `y`) of `src` against `r` at (`rx`, `ry`),
/// `rows` rows of each, `step` apart in `r`'s lines.
#[allow(clippy::too_many_arguments)]
fn sad_at(src: &[u8; 256], rows: impl Iterator<Item = usize>, r: Plane, rx: i32, ry: i32, best: u32) -> u32 {
    let mut s = 0u32;
    for (k, row) in rows.enumerate() {
        let at = r.off + (ry as usize + k) * r.stride + rx as usize;
        let a = &src[row * 16..row * 16 + 16];
        let b = &r.data[at..at + 16];
        s += a.iter().zip(b).map(|(p, q)| u32::from(p.abs_diff(*q))).sum::<u32>();
        if s >= best {
            return s;
        }
    }
    s
}

/// Search range of a vector component in half samples for f_code `f`: [−16·2^(f−1), 16·2^(f−1) − 1].
fn mv_range(f: u32) -> (i32, i32) {
    let f = 1i32 << (f - 1);
    (-16 * f, 16 * f - 1)
}

/// The best frame vector (half samples) for the macroblock from `r`, and its SAD: integer search
/// on a coarse grid then finer around the best, then half samples.
fn search_frame(src: &[u8; 256], r: &Pic, mbx: usize, mby: usize, range: (i32, i32), seeds: &[[i32; 2]]) -> ([i32; 2], u32) {
    let p = plane(r, 0, None);
    let (x0, y0) = ((mbx * 16) as i32, (mby * 16) as i32);
    // integer vectors inside the picture (one sample spare for half samples) and the f_code range
    let (xl, xh) = ((range.0 / 2).max(-x0), ((range.1 - 1) / 2).min(W as i32 - 17 - x0));
    let (yl, yh) = ((range.0 / 2).max(-y0), ((range.1 - 1) / 2).min(H as i32 - 17 - y0));
    let eval = |dx: i32, dy: i32, best: u32| sad_at(src, 0..16, p, x0 + dx, y0 + dy, best);
    let mut best = ([0i32; 2], u32::MAX);
    let try_at = |dx: i32, dy: i32, best: &mut ([i32; 2], u32)| {
        if dx < xl || dx > xh || dy < yl || dy > yh {
            return;
        }
        let s = eval(dx, dy, best.1);
        if s < best.1 {
            *best = ([dx, dy], s);
        }
    };
    try_at(0, 0, &mut best);
    for s in seeds {
        try_at(s[0] >> 1, s[1] >> 1, &mut best);
    }
    // coarse: every 4 samples over ±48 by ±24, then every sample around the best
    for dy in (-24..=24).step_by(4) {
        for dx in (-48..=48).step_by(4) {
            try_at(dx, dy, &mut best);
        }
    }
    let c = best.0;
    for dy in -3..=3 {
        for dx in -3..=3 {
            try_at(c[0] + dx, c[1] + dy, &mut best);
        }
    }
    // half samples around the best integer vector
    let mut pred = [0u8; 256];
    let mut hb = ([best.0[0] * 2, best.0[1] * 2], u32::MAX);
    for hy in -1..=1 {
        for hx in -1..=1 {
            let mv = [best.0[0] * 2 + hx, best.0[1] * 2 + hy];
            if mv[0] < range.0 || mv[0] > range.1 || mv[1] < range.0 || mv[1] > range.1 {
                continue;
            }
            // half-sample reach stays inside
            if x0 + (mv[0] >> 1) < 0 || x0 + (mv[0] >> 1) + 16 + (mv[0] & 1) > W as i32 || y0 + (mv[1] >> 1) < 0 || y0 + (mv[1] >> 1) + 16 + (mv[1] & 1) > H as i32 {
                continue;
            }
            predict(&mut pred, 0, 1, 16, p, x0, y0, mv, 16, 16, false);
            let s = sad(src, &pred, None);
            if s < hb.1 {
                hb = (mv, s);
            }
        }
    }
    hb
}

/// Field vectors for the macroblock's two fields from `r`'s two fields, around the frame vector
/// `around` (half samples, frame lines): for each field, (reference field, vector, SAD).
fn search_fields(src: &[u8; 256], r: &Pic, mbx: usize, mby: usize, range: (i32, i32), around: [i32; 2]) -> ([usize; 2], [[i32; 2]; 2], u32) {
    let (x0, fy0) = ((mbx * 16) as i32, (mby * 8) as i32);
    let mut sel = [0usize; 2];
    let mut mvs = [[0i32; 2]; 2];
    let mut total = 0u32;
    let mut pred = [0u8; 256];
    for f in 0..2 {
        let mut best = (0usize, [0i32; 2], u32::MAX);
        for rf in 0..2 {
            let p = plane(r, 0, Some(rf));
            // integer search around the frame vector, its vertical part in field lines
            let (cx, cy) = (around[0] >> 1, around[1] >> 2);
            let mut ib = ([cx, cy], u32::MAX);
            for dy in -2..=2 {
                for dx in -2..=2 {
                    let (vx, vy) = (cx + dx, cy + dy);
                    if 2 * vx < range.0 || 2 * vx > range.1 || 2 * vy < range.0 || 2 * vy > range.1 {
                        continue;
                    }
                    if x0 + vx < 0 || x0 + vx + 17 > W as i32 || fy0 + vy < 0 || fy0 + vy + 9 > (H / 2) as i32 {
                        continue;
                    }
                    let s = sad_at(src, (f..16).step_by(2), p, x0 + vx, fy0 + vy, ib.1);
                    if s < ib.1 {
                        ib = ([vx, vy], s);
                    }
                }
            }
            if ib.1 == u32::MAX {
                continue;
            }
            for hy in -1..=1 {
                for hx in -1..=1 {
                    let mv = [ib.0[0] * 2 + hx, ib.0[1] * 2 + hy];
                    if mv[0] < range.0 || mv[0] > range.1 || mv[1] < range.0 || mv[1] > range.1 {
                        continue;
                    }
                    if x0 + (mv[0] >> 1) < 0 || x0 + (mv[0] >> 1) + 16 + (mv[0] & 1) > W as i32 || fy0 + (mv[1] >> 1) < 0 || fy0 + (mv[1] >> 1) + 8 + (mv[1] & 1) > (H / 2) as i32 {
                        continue;
                    }
                    predict(&mut pred, f, 2, 16, p, x0, fy0, mv, 16, 8, false);
                    let s = sad(src, &pred, Some(f));
                    if s < best.2 {
                        best = (rf, mv, s);
                    }
                }
            }
        }
        sel[f] = best.0;
        mvs[f] = best.1;
        total = total.saturating_add(best.2);
    }
    (sel, mvs, total)
}

// ---------------------------------------------------------------- planning

fn luma_of(p: &Pic, mbx: usize, mby: usize) -> [u8; 256] {
    std::array::from_fn(|i| p.y[(mby * 16 + i / 16) * W + mbx * 16 + i % 16])
}

fn chroma_of(p: &Pic, k: usize, mbx: usize, mby: usize) -> [u8; 128] {
    std::array::from_fn(|i| p.c[k][(mby * 16 + i / 8) * CW + mbx * 8 + i % 8])
}

/// Sum of absolute deviations from the mean of the four 8x8 luma blocks (frame or field).
fn intra_cost(y: &[u8; 256]) -> u32 {
    let mut cost = 0u32;
    for b in 0..4 {
        let (bx, by) = ((b % 2) * 8, (b / 2) * 8);
        let sum: u32 = (0..64).map(|i| u32::from(y[(by + i / 8) * 16 + bx + i % 8])).sum();
        let mean = (sum + 32) / 64;
        cost += (0..64).map(|i| u32::from(y[(by + i / 8) * 16 + bx + i % 8]).abs_diff(mean)).sum::<u32>();
    }
    cost
}

/// TM5 spatial activity: 1 + the smallest variance of the eight 8x8 luma blocks (frame and
/// field arrangements).
fn activity(y: &[u8; 256]) -> f64 {
    let mut least = f64::MAX;
    for field in [false, true] {
        for b in 0..4 {
            let (bx, k) = ((b % 2) * 8, b / 2);
            let row = |r: usize| if field { k + 2 * r } else { k * 8 + r };
            let v: Vec<f64> = (0..64).map(|i| f64::from(y[row(i / 8) * 16 + bx + i % 8])).collect();
            let mean = v.iter().sum::<f64>() / 64.0;
            least = least.min(v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / 64.0);
        }
    }
    1.0 + least
}

/// Field DCT when the fields differ less line to line than the frame does.
fn prefers_field_dct(d: &[i32; 256]) -> bool {
    let (mut frame, mut field) = (0i64, 0i64);
    for r in 0..14 {
        for i in 0..16 {
            frame += i64::from((d[r * 16 + i] - d[(r + 1) * 16 + i]).abs());
            field += i64::from((d[r * 16 + i] - d[(r + 2) * 16 + i]).abs());
        }
    }
    field < frame
}

/// The 8 blocks of a macroblock's samples (or residual): luma 16x16 and chroma 8x16 each, frame
/// or field arranged.
fn blocks(y: &[i32; 256], c: &[[i32; 128]; 2], field: bool) -> [[i32; 64]; 8] {
    let mut out = [[0i32; 64]; 8];
    for b in 0..4 {
        let x0 = (b & 1) * 8;
        for r in 0..8 {
            let line = if field { (b >> 1) + 2 * r } else { (b >> 1) * 8 + r };
            out[b][r * 8..r * 8 + 8].copy_from_slice(&y[line * 16 + x0..line * 16 + x0 + 8]);
        }
    }
    for i in 4..8 {
        let (k, cc) = ((i - 4) >> 1, (i - 4) & 1);
        for r in 0..8 {
            let line = if field { k + 2 * r } else { k * 8 + r };
            out[i][r * 8..r * 8 + 8].copy_from_slice(&c[cc][line * 8..line * 8 + 8]);
        }
    }
    out
}

/// Plan macroblock row `mby`: predictions, DCT types, coefficients and activities.
pub fn plan_row(s: &Setup, mby: usize) -> Vec<MbPlan> {
    let mut out = Vec::with_capacity(MB_W);
    let mut last_mv: [[i32; 2]; 2] = [[0; 2]; 2];
    for mbx in 0..MB_W {
        let src = luma_of(s.src, mbx, mby);
        let srcc = [chroma_of(s.src, 0, mbx, mby), chroma_of(s.src, 1, mbx, mby)];
        let intra = intra_cost(&src);
        let mut pred = Pred::Intra;
        let mut pred_y = [0u8; 256];
        let mut pred_c = [[0u8; 128]; 2];
        if s.ptype != PType::I {
            let mut cands: Vec<(u32, Pred)> = Vec::new();
            let mut frame_best = [None, None];
            for d in 0..2 {
                let Some(r) = [s.fwd, s.bwd][d] else { continue };
                if s.ptype == PType::P && d == 1 {
                    continue;
                }
                let range = mv_range(s.f_code[d]);
                let (mv, sad_f) = search_frame(&src, r, mbx, mby, range, &[last_mv[d]]);
                frame_best[d] = Some(mv);
                let p = if d == 0 { Pred::Frame { fwd: Some(mv), bwd: None } } else { Pred::Frame { fwd: None, bwd: Some(mv) } };
                // a little in favour of the zero vector (cheaper to code, P skips)
                let bias = if mv == [0, 0] { 0 } else { 48 };
                cands.push((sad_f + bias, p));
                let (sel, mvs, sad_fld) = search_fields(&src, r, mbx, mby, range, mv);
                if sad_fld < u32::MAX {
                    cands.push((sad_fld + 96, Pred::Field { dir: d, sel, mv: mvs }));
                }
            }
            if let (Some(f), Some(b)) = (frame_best[0], frame_best[1]) {
                let p = Pred::Frame { fwd: Some(f), bwd: Some(b) };
                let mut py = [0u8; 256];
                let mut pc = [[0u8; 128]; 2];
                form_prediction(s, &p, mbx, mby, &mut py, &mut pc);
                cands.push((sad(&src, &py, None) + 64, p));
            }
            if let Some(&(cost, p)) = cands.iter().min_by_key(|c| c.0)
                && cost < intra + 512
            {
                pred = p;
                form_prediction(s, &p, mbx, mby, &mut pred_y, &mut pred_c);
                match p {
                    Pred::Frame { fwd, bwd } => {
                        if let Some(v) = fwd {
                            last_mv[0] = v;
                        }
                        if let Some(v) = bwd {
                            last_mv[1] = v;
                        }
                    }
                    Pred::Field { dir, mv, .. } => last_mv[dir] = [mv[0][0], mv[0][1] * 2],
                    Pred::Intra => {}
                }
            }
        }
        let (dy, dc): ([i32; 256], [[i32; 128]; 2]) = if pred == Pred::Intra {
            (std::array::from_fn(|i| i32::from(src[i])), std::array::from_fn(|k| std::array::from_fn(|i| i32::from(srcc[k][i]))))
        } else {
            (
                std::array::from_fn(|i| i32::from(src[i]) - i32::from(pred_y[i])),
                std::array::from_fn(|k| std::array::from_fn(|i| i32::from(srcc[k][i]) - i32::from(pred_c[k][i]))),
            )
        };
        let field_dct = prefers_field_dct(&dy);
        let bl = blocks(&dy, &dc, field_dct);
        let coef = std::array::from_fn(|b| fdct(&bl[b]));
        out.push(MbPlan { pred, field_dct, coef, pred_y, pred_c, act: activity(&src) });
    }
    out
}

// ---------------------------------------------------------------- coding

/// Intra or non-intra quantisation of a block (raster coefficients → levels in raster).
fn quantise(c: &[f32; 64], intra: bool, qs: i32) -> [i32; 64] {
    let mut out = [0i32; 64];
    for i in 0..64 {
        if intra && i == 0 {
            out[0] = ((c[0] / DC_MULT as f32).round() as i32).clamp(0, (1 << (8 + DC_PRECISION)) - 1);
            continue;
        }
        let w = if intra { DEFAULT_INTRA[i] } else { 16 } as f32;
        let step = w * qs as f32 / 16.0;
        let a = c[i].abs() / step;
        // intra: rounding with a small dead zone; non-intra: the levels' reconstruction sits in
        // the middle of their interval, so truncate
        let l = if intra { (a + 0.375).floor() } else { (a - 0.1).floor().max(0.0) } as i32;
        out[i] = if c[i] < 0.0 { -l } else { l }.clamp(-2047, 2047);
    }
    out
}

/// The decoder's dequantisation with mismatch control (§7.4.2–§7.4.4), then the IDCT.
fn reconstruct(lv: &[i32; 64], intra: bool, qs: i32) -> [i32; 64] {
    let mut f = [0i32; 64];
    let mut sum = 0i32;
    for i in 0..64 {
        let l = lv[i];
        f[i] = if intra && i == 0 {
            l * DC_MULT
        } else if l == 0 {
            0
        } else {
            let w = if intra { DEFAULT_INTRA[i] } else { 16 };
            let k = if intra { 0 } else { l.signum() };
            ((2 * l + k) * w * qs / 32).clamp(-2048, 2047)
        };
        sum += f[i];
    }
    if sum & 1 == 0 {
        f[63] ^= 1;
    }
    idct(&mut f);
    f
}

/// The coded slices of a picture and their total size in bits.
pub struct Coded {
    pub slices: Vec<Vec<u8>>,
    pub bits: usize,
    /// Mean quantiser scale used (for the rate control's complexity).
    pub mean_q: f64,
}

/// What coding one macroblock row decided, for the reconstruction.
pub struct RowCoding {
    /// Per macroblock: (quantiser scale, levels of the eight blocks, coded block pattern).
    mbs: Vec<(i32, [[i32; 64]; 8], u8)>,
}

/// Code macroblock row `mby` (one slice) at base quantiser scale `q`, `mean_act` the picture's
/// mean activity.
pub fn code_row(s: &Setup, mby: usize, plan: &[MbPlan], q: f64, mean_act: f64) -> (Vec<u8>, RowCoding, f64) {
    let t = tables();
    let mut w = BitWriter::new();
    let qcodes: Vec<usize> = plan
        .iter()
        .map(|m| {
            let n = (2.0 * m.act + mean_act) / (m.act + 2.0 * mean_act);
            q_code(q * n).clamp(1, 31)
        })
        .collect();
    // slice header: the first macroblock's scale
    let mut cur_code = qcodes[0];
    w.write_bits(0x0000_0100 | (mby as u32 + 1), 32);
    w.write_bits(cur_code as u32, 5);
    w.write_bit(false); // extra_bit_slice
    let mut dc_pred = [DC_RESET; 3];
    let mut pmv = [[[0i32; 2]; 2]; 2];
    let mut skipped_run = 0usize;
    let mut rows = RowCoding { mbs: Vec::with_capacity(MB_W) };
    let mut qsum = 0f64;
    for (mbx, m) in plan.iter().enumerate() {
        let intra = m.pred == Pred::Intra;
        let want = qcodes[mbx];
        let qs_try = NON_LINEAR_Q[want];
        let lv_try: [[i32; 64]; 8] = std::array::from_fn(|b| quantise(&m.coef[b], intra, qs_try));
        let cbp_try = cbp_of(&lv_try, intra);
        // the scale can change only with a coded pattern (or intra)
        let (code, lv, cbp) = if want != cur_code && (intra || cbp_try != 0) {
            (want, lv_try, cbp_try)
        } else if want == cur_code {
            (cur_code, lv_try, cbp_try)
        } else {
            let qs = NON_LINEAR_Q[cur_code];
            let lv: [[i32; 64]; 8] = std::array::from_fn(|b| quantise(&m.coef[b], intra, qs));
            let cbp = cbp_of(&lv, intra);
            (cur_code, lv, cbp)
        };
        let qs = NON_LINEAR_Q[code];
        qsum += f64::from(qs);
        let quant_change = code != cur_code;
        // a P macroblock predicted at the zero frame vector with nothing to code is skipped
        // (never the first or the last of the slice)
        let zero_p = s.ptype == PType::P && m.pred == (Pred::Frame { fwd: Some([0, 0]), bwd: None });
        if zero_p && cbp == 0 && mbx > 0 && mbx + 1 < MB_W {
            skipped_run += 1;
            pmv = [[[0; 2]; 2]; 2];
            dc_pred = [DC_RESET; 3];
            rows.mbs.push((qs, lv, 0));
            continue;
        }
        // macroblock_address_increment
        let mut inc = skipped_run + 1;
        while inc > 33 {
            write(&mut w, t.mba[34]);
            inc -= 33;
        }
        write(&mut w, t.mba[inc]);
        skipped_run = 0;
        let pattern = !intra && cbp != 0;
        let mut flags = if intra { MB_INTRA } else { 0 };
        if pattern {
            flags |= MB_PATTERN;
        }
        let (fwd, bwd) = match m.pred {
            Pred::Intra => (false, false),
            Pred::Frame { fwd, bwd } => (fwd.is_some(), bwd.is_some()),
            Pred::Field { dir, .. } => (dir == 0, dir == 1),
        };
        // P: a zero frame vector with a coded pattern goes as "No MC"
        let no_mc = zero_p && pattern;
        if fwd && !no_mc {
            flags |= MB_FWD;
        }
        if bwd {
            flags |= MB_BWD;
        }
        if quant_change {
            flags |= MB_QUANT;
            cur_code = code;
        }
        write(&mut w, mb_type_code(s.ptype, flags));
        if flags & (MB_FWD | MB_BWD) != 0 {
            // frame_motion_type: 01 field, 10 frame
            w.write_bits(if matches!(m.pred, Pred::Field { .. }) { 1 } else { 2 }, 2);
        }
        if intra || pattern {
            w.write_bit(m.field_dct);
        }
        if flags & MB_QUANT != 0 {
            w.write_bits(code as u32, 5);
        }
        // motion vectors
        match m.pred {
            Pred::Intra => pmv = [[[0; 2]; 2]; 2],
            Pred::Frame { fwd: f, bwd: b } => {
                for (d, v) in [f, b].into_iter().enumerate() {
                    let Some(v) = v else { continue };
                    if d == 0 && no_mc {
                        continue;
                    }
                    for c in 0..2 {
                        motion(&mut w, v[c] - pmv[0][d][c], s.f_code[d]);
                        pmv[0][d][c] = v[c];
                    }
                    pmv[1][d] = pmv[0][d];
                }
                if no_mc {
                    pmv = [[[0; 2]; 2]; 2];
                }
            }
            Pred::Field { dir, sel, mv } => {
                for r in 0..2 {
                    w.write_bit(sel[r] == 1); // motion_vertical_field_select
                    motion(&mut w, mv[r][0] - pmv[r][dir][0], s.f_code[dir]);
                    pmv[r][dir][0] = mv[r][0];
                    // vertical: the predictor in field units is PMV / 2
                    motion(&mut w, mv[r][1] - (pmv[r][dir][1] >> 1), s.f_code[dir]);
                    pmv[r][dir][1] = mv[r][1] * 2;
                }
            }
        }
        if pattern {
            write(&mut w, t.cbp[usize::from(cbp >> 2)]);
            w.write_bits(u32::from(cbp & 3), 2);
        }
        if !intra {
            dc_pred = [DC_RESET; 3];
        }
        for b in 0..8 {
            if cbp & (0x80 >> b) == 0 {
                continue;
            }
            let cc = if b < 4 { 0 } else { 1 + ((b - 4) & 1) };
            block(&mut w, &lv[b], intra, cc == 0, &mut dc_pred[cc]);
        }
        rows.mbs.push((qs, lv, cbp));
    }
    // stuffing-free end: the slice is byte aligned before the next start code
    w.align_zero();
    let mean_q = qsum / MB_W as f64;
    (w.finish(), rows, mean_q)
}

fn cbp_of(lv: &[[i32; 64]; 8], intra: bool) -> u8 {
    if intra {
        return 0xFF;
    }
    let mut cbp = 0u8;
    for b in 0..8 {
        if lv[b].iter().any(|&v| v != 0) {
            cbp |= 0x80 >> b;
        }
    }
    cbp
}

fn write(w: &mut BitWriter, c: Code) {
    w.write_bits(c.bits, u32::from(c.len));
}

/// One motion vector component difference (§6.2.5.2.1), brought into the f_code range.
fn motion(w: &mut BitWriter, delta: i32, f_code: u32) {
    let t = tables();
    let r_size = f_code - 1;
    let f = 1i32 << r_size;
    let (lo, hi) = (-16 * f, 16 * f - 1);
    let mut d = delta;
    if d < lo {
        d += 32 * f;
    } else if d > hi {
        d -= 32 * f;
    }
    if d == 0 || f == 1 {
        write(w, t.motion[(d + 16) as usize]);
        return;
    }
    let a = d.abs() - 1;
    let code = a / f + 1;
    let residual = a % f;
    write(w, t.motion[(code * d.signum() + 16) as usize]);
    w.write_bits(residual as u32, r_size);
}

/// One block's coefficients (§6.2.6): intra DC as a difference, then run/level codes in the
/// alternate scan, escapes beyond the tables, end of block.
fn block(w: &mut BitWriter, lv: &[i32; 64], intra: bool, luma: bool, dc_pred: &mut i32) {
    let t = tables();
    if intra {
        let diff = lv[0] - *dc_pred;
        *dc_pred = lv[0];
        let size = 32 - diff.unsigned_abs().leading_zeros();
        write(w, if luma { t.dc_luma[size as usize] } else { t.dc_chroma[size as usize] });
        if size > 0 {
            let v = if diff > 0 { diff } else { diff + (1 << size) - 1 };
            w.write_bits(v as u32, size);
        }
    }
    // intra blocks use table one (intra_vlc_format 1), the others table zero
    let one = intra;
    let mut run = 0usize;
    let mut first = !intra;
    for k in usize::from(intra)..64 {
        let l = lv[ALT_SCAN[k]];
        if l == 0 {
            run += 1;
            continue;
        }
        let a = l.unsigned_abs();
        if first && run == 0 && a == 1 {
            // "1s": the first coefficient of a non-intra block
            w.write_bit(true);
            w.write_bit(l < 0);
        } else if let Some(c) = t.run_level(one, run, a) {
            write(w, c);
            w.write_bit(l < 0);
        } else {
            write(w, t.escape);
            w.write_bits(run as u32, 6);
            w.write_bits((l & 0xFFF) as u32, 12);
        }
        first = false;
        run = 0;
    }
    write(w, t.eob[usize::from(one)]);
}

/// The reconstruction of macroblock row `mby` as the decoder makes it: 16 luma lines and 16 of
/// each chroma component.
pub fn reconstruct_row(plan: &[MbPlan], coding: &RowCoding) -> ([u8; W * 16], [[u8; CW * 16]; 2]) {
    let mut y = [0u8; W * 16];
    let mut c = [[0u8; CW * 16]; 2];
    for (mbx, (m, (qs, lv, cbp))) in plan.iter().zip(&coding.mbs).enumerate() {
        let intra = m.pred == Pred::Intra;
        let res: [[i32; 64]; 8] = std::array::from_fn(|b| if cbp & (0x80 >> b) != 0 { reconstruct(&lv[b], intra, *qs) } else { [0; 64] });
        for b in 0..8 {
            for r in 0..8 {
                for i in 0..8 {
                    let v = res[b][r * 8 + i];
                    if b < 4 {
                        let line = if m.field_dct { (b >> 1) + 2 * r } else { (b >> 1) * 8 + r };
                        let px = (b & 1) * 8 + i;
                        let base = if intra { 0 } else { i32::from(m.pred_y[line * 16 + px]) };
                        y[line * W + mbx * 16 + px] = (base + v).clamp(0, 255) as u8;
                    } else {
                        let (k, cc) = ((b - 4) >> 1, (b - 4) & 1);
                        let line = if m.field_dct { k + 2 * r } else { k * 8 + r };
                        let base = if intra { 0 } else { i32::from(m.pred_c[cc][line * 8 + i]) };
                        c[cc][line * CW + mbx * 8 + i] = (base + v).clamp(0, 255) as u8;
                    }
                }
            }
        }
    }
    (y, c)
}
